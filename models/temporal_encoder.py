import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention for temporal modeling."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class BiLSTMTemporalEncoder(nn.Module):
    """Bidirectional LSTM encoder with optional self-attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_transformer: bool = False,
        fusion: str = "concat",
        roi_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional LSTM")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_transformer = use_transformer
        self.input_dim = input_dim

        valid_fusion = {"concat"}
        if fusion not in valid_fusion:
            raise ValueError(f"Unsupported fusion mode '{fusion}'. Available: {sorted(valid_fusion)}")
        self.fusion = fusion

        valid_pooling = {"mean", "max"}
        if roi_pooling not in valid_pooling:
            raise ValueError(f"Unsupported ROI pooling '{roi_pooling}'. Available: {sorted(valid_pooling)}")
        self.roi_pooling = roi_pooling

        # Initialize LSTM with input_size matching the hidden_dim since we project to it
        self.lstm = nn.LSTM(
            input_size=hidden_dim,  # Match the projected dimension
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Add projection layer to handle the input dimension mismatch
        # First, dynamically compute the actual input dimension
        self.actual_input_dim = input_dim  # This will be updated in the first forward pass
        
        # Use a placeholder for the input projection
        self.input_proj = nn.Sequential(
            nn.Linear(1, 1),  # Dummy layer to be replaced
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if hidden_dim > 1 else nn.Identity()
        )
        
        # Flag to track if we've initialized the projection layer
        self._proj_initialized = False

        if use_transformer:
            self.self_attn = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            self.attn_layer_norm = nn.LayerNorm(hidden_dim)
            self.attn_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def reset_state(self) -> None:
        """Reset cached hidden states between videos."""
        self._hidden_state = None

    def detach_state(self) -> None:
        """Detach cached hidden state from the current computation graph."""
        if self._hidden_state is not None:
            h, c = self._hidden_state
            self._hidden_state = (h.detach(), c.detach())

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        roi_embeddings: Optional[torch.Tensor] = None,
        reset_state: bool = False,
        detach_state: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if reset_state:
            self.reset_state()
        if detach_state:
            self.detach_state()

        inputs = self._prepare_inputs(x, roi_embeddings)
        batch_size, seq_len, _ = inputs.size()
        packed = None
        perm_idx = None
        original_lengths = lengths.clone() if lengths is not None else None

        if lengths is not None:
            lengths, perm_idx = lengths.sort(descending=True)
            inputs = inputs[perm_idx]
            packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=True)

        if hidden is None:
            if self._hidden_state is not None:
                hidden = tuple(h.to(inputs.device) for h in self._hidden_state)
            else:
                h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=inputs.device)
                c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2, device=inputs.device)
                hidden = (h0, c0)
        else:
            hidden = tuple(h.to(inputs.device) for h in hidden)

        if perm_idx is not None:
            hidden = (hidden[0][:, perm_idx], hidden[1][:, perm_idx])

        lstm_out, hidden = self.lstm(packed if packed is not None else inputs, hidden)

        # Cache detached hidden state for optional reuse
        self._hidden_state = (hidden[0].detach(), hidden[1].detach())

        if isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence):
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

        if perm_idx is not None:
            _, unperm_idx = perm_idx.sort()
            lstm_out = lstm_out[unperm_idx]
            hidden = (hidden[0][:, unperm_idx], hidden[1][:, unperm_idx])
            lengths = original_lengths
        else:
            lengths = original_lengths

        lstm_out = self.layer_norm(lstm_out)
        attn_weights = None

        if self.use_transformer:
            attn_mask = None
            key_padding_mask = None

            if lengths is not None:
                max_len = seq_len
                attn_mask = torch.triu(torch.ones(max_len, max_len, device=lstm_out.device) * float("-inf"), diagonal=1)
                key_padding_mask = torch.arange(max_len, device=lstm_out.device).expand(batch_size, max_len) >= lengths.unsqueeze(1).to(lstm_out.device)

            attn_out, attn_weights = self.self_attn(
                query=lstm_out,
                key=lstm_out,
                value=lstm_out,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            lstm_out = self.attn_layer_norm(lstm_out + self.attn_dropout(attn_out))

        output = self.dropout(lstm_out)
        
        # Get the device from the output tensor
        device = output.device
        
        # Ensure we have valid lengths
        if lengths is None:
            lengths = torch.tensor([output.size(1)] * output.size(0), device=device)
        
        # Return a dictionary with the expected keys for the training loop
        return {
            'features': output,  # Main output features for the decoder
            'frame_features': output,  # Alias for backward compatibility
            'outputs': output,   # Another common alias
            'hidden': hidden,    # Hidden states for potential use
            'attn_weights': attn_weights,  # Attention weights if using transformer
            'video_lengths': lengths,
            'lengths': lengths  # Alias for video_lengths
        }

    def _prepare_inputs(
        self,
        visual_embeddings: torch.Tensor,
        roi_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Initialize the projection layer on first forward pass
        if not self._proj_initialized:
            self.actual_input_dim = visual_embeddings.size(-1)
            if self.actual_input_dim != self.input_dim:
                # Replace the first layer of input_proj with the correct dimension
                self.input_proj[0] = nn.Linear(self.actual_input_dim, self.hidden_dim)
                nn.init.xavier_uniform_(self.input_proj[0].weight)
                nn.init.constant_(self.input_proj[0].bias, 0.0)
                logger = logging.getLogger('video_captioning')
                logger.info(f"Initialized input projection: {self.actual_input_dim} -> {self.hidden_dim}")
            self._proj_initialized = True
        
        # Project visual embeddings to match expected dimension
        projected_visual = self.input_proj(visual_embeddings)
        
        if roi_embeddings is None:
            inputs = projected_visual
        else:
            roi_features = self._pool_roi_embeddings(roi_embeddings)
            if self.fusion == "concat":
                # Project ROI features to match visual feature dimension
                if roi_features.size(-1) != projected_visual.size(-1):
                    # Create a separate projection for ROI features if needed
                    if not hasattr(self, 'roi_proj'):
                        self.roi_proj = nn.Linear(roi_features.size(-1), projected_visual.size(-1)).to(roi_features.device)
                        nn.init.xavier_uniform_(self.roi_proj.weight)
                        nn.init.constant_(self.roi_proj.bias, 0.0)
                    roi_features = self.roi_proj(roi_features)
                inputs = torch.cat([projected_visual, roi_features], dim=-1)
            else:  # pragma: no cover - guarded by validation in __init__
                raise RuntimeError(f"Unsupported fusion mode '{self.fusion}'")

        if inputs.dim() != 3:
            raise ValueError("Expected fused inputs to have shape (batch, seq_len, feature_dim)")
        
        # Project to the expected hidden dimension if needed
        if inputs.size(-1) != self.hidden_dim:
            if not hasattr(self, 'final_proj'):
                self.final_proj = nn.Linear(inputs.size(-1), self.hidden_dim).to(inputs.device)
                nn.init.xavier_uniform_(self.final_proj.weight)
                nn.init.constant_(self.final_proj.bias, 0.0)
                logger = logging.getLogger('video_captioning')
                logger.info(f"Projecting features from {inputs.size(-1)} to {self.hidden_dim}")
            inputs = self.final_proj(inputs)

        return inputs

    def _pool_roi_embeddings(self, roi_embeddings: torch.Tensor) -> torch.Tensor:
        if roi_embeddings.dim() == 4:
            if self.roi_pooling == "mean":
                pooled = roi_embeddings.mean(dim=2)
            elif self.roi_pooling == "max":
                pooled = roi_embeddings.max(dim=2).values
            else:  # pragma: no cover - guarded
                raise RuntimeError(f"Unsupported ROI pooling '{self.roi_pooling}'")
        elif roi_embeddings.dim() == 3:
            pooled = roi_embeddings
        else:
            raise ValueError("ROI embeddings must have shape (batch, seq, features) or (batch, seq, regions, features)")

        return pooled
