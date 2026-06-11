from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingModule(nn.Module):
    """Align decoder states with grounded visual features to score object relevance."""

    def __init__(
        self,
        decoder_dim: int,
        object_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim)
        self.object_proj = nn.Linear(object_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.decoder_proj.weight)
        nn.init.zeros_(self.decoder_proj.bias)
        nn.init.xavier_uniform_(self.object_proj.weight)
        nn.init.zeros_(self.object_proj.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        object_features: torch.Tensor,
        *,
        attention_weights: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute grounding logits for each generated token.

        Args:
            decoder_hidden: Decoder hidden states ``[B, L_caption, H_dec]``.
            object_features: ROI-augmented frame features ``[B, L_video, D_obj]``.
            attention_weights: Optional attention maps ``[B, L_caption, L_video]`` used to
                pool object features per token. When omitted, the mean of valid frames is used.
            video_lengths: Optional per-video frame counts for masking ``[B]``.

        Returns:
            Grounding logits ``[B, L_caption]`` suitable for BCE loss with logits.
        """
        if object_features.dim() != 3:
            raise ValueError("object_features must have shape [B, L_video, D_obj]")

        batch_size, video_len, _ = object_features.size()
        caption_len = decoder_hidden.size(1)

        if attention_weights is not None:
            if attention_weights.dim() != 3:
                raise ValueError("attention_weights must have shape [B, L_caption, L_video]")
            if attention_weights.size(2) != video_len:
                raise ValueError("Attention weights length must match video feature steps")
            pooled_objects = torch.bmm(attention_weights, object_features)
        else:
            pooled_objects = object_features.mean(dim=1, keepdim=True).expand(batch_size, caption_len, -1)

        if video_lengths is not None:
            mask = (
                torch.arange(video_len, device=object_features.device)
                .unsqueeze(0)
                .expand(batch_size, video_len)
            )
            valid = mask < video_lengths.unsqueeze(1)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            masked_features = object_features * valid.unsqueeze(-1)
            fallback = masked_features.sum(dim=1, keepdim=True) / denom
            pooled_objects = torch.where(
                valid.sum(dim=1, keepdim=True).unsqueeze(-1) == 0,
                fallback.expand_as(pooled_objects),
                pooled_objects,
            )

        fused = torch.tanh(
            self.decoder_proj(decoder_hidden) + self.object_proj(pooled_objects)
        )
        logits = self.output(self.dropout(fused)).squeeze(-1)
        return logits
