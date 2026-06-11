import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Attention(nn.Module):
    """Additive attention for decoder focus over encoder outputs."""

    def __init__(self, hidden_dim: int, encoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights over encoder outputs."""
        att1 = self.encoder_att(encoder_outputs)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)

        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))

        alpha = self.softmax(att)
        context = (encoder_outputs * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class CoverageAttention(Attention):
    """Attention mechanism augmented with coverage to discourage repetition."""

    def __init__(self, hidden_dim: int, encoder_dim: int, attention_dim: int) -> None:
        super().__init__(hidden_dim, encoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        coverage: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        att1 = self.encoder_att(encoder_outputs)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att3 = self.coverage_att(coverage)
        att = self.full_att(self.relu(att1 + att2 + att3)).squeeze(2)

        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))

        alpha = self.softmax(att)
        coverage = coverage + alpha.unsqueeze(2)
        context = (encoder_outputs * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha, coverage
