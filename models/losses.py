from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    ce_loss: torch.Tensor
    coverage_loss: torch.Tensor
    grounding_loss: torch.Tensor


class CaptionLoss(nn.Module):
    """Composite loss for caption generation with coverage and grounding terms."""

    def __init__(
        self,
        vocab_pad_idx: int,
        coverage_weight: float = 1.0,
        grounding_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_pad_idx = vocab_pad_idx
        self.coverage_weight = coverage_weight
        self.grounding_weight = grounding_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        caption_lengths: torch.Tensor,
        attn_alphas: torch.Tensor,
        grounding_scores: Optional[torch.Tensor] = None,
        grounding_targets: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Compute total loss with optional coverage and grounding components."""
        ce_loss = self._cross_entropy(predictions, targets, caption_lengths)
        coverage_loss = self._coverage(attn_alphas, caption_lengths)

        grounding_loss = torch.zeros_like(ce_loss)
        if grounding_scores is not None and grounding_targets is not None:
            grounding_loss = self._grounding(grounding_scores, grounding_targets)

        total = ce_loss + self.coverage_weight * coverage_loss + self.grounding_weight * grounding_loss
        return LossOutput(total=total, ce_loss=ce_loss, coverage_loss=coverage_loss, grounding_loss=grounding_loss)

    def _cross_entropy(
        self, predictions: torch.Tensor, targets: torch.Tensor, caption_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy with optional label smoothing and padding mask."""
        batch_size, max_len, vocab_size = predictions.size()
        predictions = predictions.view(batch_size * max_len, vocab_size)
        targets = targets[:, :max_len].reshape(-1)

        if self.label_smoothing > 0:
            smooth_dist = torch.full_like(predictions, fill_value=self.label_smoothing / (vocab_size - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(predictions, dim=-1)
            loss = -(smooth_dist * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(
                predictions,
                targets,
                ignore_index=self.vocab_pad_idx,
                reduction="none",
            )

        mask = targets != self.vocab_pad_idx
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        return loss

    @staticmethod
    def _coverage(attn_alphas: torch.Tensor, caption_lengths: torch.Tensor) -> torch.Tensor:
        """Coverage loss encouraging attention diversity."""
        # attn_alphas: [B, T, N]
        coverage = torch.zeros_like(attn_alphas[:, 0, :])
        cov_loss = 0.0

        for t in range(attn_alphas.size(1)):
            alpha = attn_alphas[:, t, :]
            cov_loss = cov_loss + torch.min(alpha, coverage).sum(dim=1)
            coverage = coverage + alpha

        mask = (caption_lengths > 0).float()
        cov_loss = (cov_loss * mask).mean()
        return cov_loss

    @staticmethod
    def _grounding(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Grounding loss aligns caption tokens with detected objects."""
        return F.binary_cross_entropy_with_logits(scores, targets.float())
