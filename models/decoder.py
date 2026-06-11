import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, CoverageAttention


class HierarchicalDecoder(nn.Module):
    """Hierarchical caption decoder supporting coverage attention and sampling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_coverage: bool = True,
        max_seq_length: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.use_coverage = use_coverage
        self.max_seq_length = max_seq_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        attention_dim = hidden_dim
        self.attention = Attention(hidden_dim, encoder_dim, attention_dim)
        self.coverage_attention = CoverageAttention(hidden_dim, encoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim + encoder_dim, hidden_dim, bias=False), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(hidden_dim + encoder_dim, hidden_dim, bias=False), nn.Tanh())
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0.0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        return h, c

    def forward_step(
        self,
        word: torch.Tensor,
        prev_hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        word_embed = self.embedding(word)

        if self.use_coverage and coverage is not None:
            context, alpha, coverage = self.coverage_attention(prev_hidden[0], encoder_outputs, coverage, mask)
        else:
            context, alpha = self.attention(prev_hidden[0], encoder_outputs, mask)
            if self.use_coverage:
                coverage = alpha.unsqueeze(2)

        lstm_input = torch.cat([word_embed, context], dim=1)
        h, c = self.lstm(lstm_input, prev_hidden)
        gate = self.gate(torch.cat([h, context], dim=1))
        transformed = self.transform(torch.cat([h, context], dim=1))
        h = gate * h + (1.0 - gate) * transformed
        scores = self.fc(self.dropout(h))

        return scores, (h, c), alpha, coverage

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        num_pixels = encoder_outputs.size(1)
        max_len = max_length or self.max_seq_length

        hidden = self.init_hidden_state(batch_size)
        word = torch.ones(batch_size, dtype=torch.long, device=self.device)
        coverage = torch.zeros(batch_size, num_pixels, 1, device=self.device) if self.use_coverage else None
        mask = None

        predictions: List[torch.Tensor] = []
        alphas: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        if caption_lengths is not None:
            mask = torch.ones(batch_size, num_pixels, device=self.device)
            for i, length in enumerate(caption_lengths):
                if length < num_pixels:
                    mask[i, length:] = 0
            if max_length is None:
                max_len = int(caption_lengths.max().item())

        for t in range(max_len):
            scores, hidden, alpha, coverage = self.forward_step(word, hidden, encoder_outputs, coverage, mask)
            predictions.append(scores)
            alphas.append(alpha)
            hidden_states.append(hidden[0])

            if captions is not None and t + 1 < captions.size(1) and random.random() < teacher_forcing_ratio:
                word = captions[:, t + 1]
            else:
                word = scores.argmax(dim=1)

        predictions = torch.stack(predictions, dim=1)
        alphas = torch.stack(alphas, dim=1)
        hidden_tensor = torch.stack(hidden_states, dim=1)

        return {"predictions": predictions, "alphas": alphas, "hidden_states": hidden_tensor}

    def sample(
        self,
        encoder_outputs: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: int = 2,
    ) -> Dict[str, torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        num_pixels = encoder_outputs.size(1)
        max_len = max_length or self.max_seq_length

        hidden = self.init_hidden_state(batch_size)
        word = torch.ones(batch_size, dtype=torch.long, device=self.device)
        coverage = torch.zeros(batch_size, num_pixels, 1, device=self.device) if self.use_coverage else None

        sequences = [word.unsqueeze(1)]
        alphas = []

        for _ in range(max_len):
            scores, hidden, alpha, coverage = self.forward_step(word, hidden, encoder_outputs, coverage)
            scores = scores / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                scores = self._top_k_filter(scores, top_k)
            if top_p < 1.0:
                scores = self._top_p_filter(scores, top_p)

            probs = F.softmax(scores, dim=-1)
            word = torch.multinomial(probs, num_samples=1).squeeze(1)
            sequences.append(word.unsqueeze(1))
            alphas.append(alpha)

            if (word == eos_token_id).all():
                break

        sequence = torch.cat(sequences, dim=1)
        attn_weights = torch.stack(alphas, dim=1) if alphas else torch.zeros(batch_size, 0, num_pixels, device=self.device)

        return {"sequence": sequence, "alphas": attn_weights}

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > top_p
        mask[:, 0] = False
        sorted_logits[mask] = float("-inf")

        original_logits = torch.full_like(logits, float("-inf"))
        original_logits.scatter_(1, sorted_indices, sorted_logits)
        return original_logits
