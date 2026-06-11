from __future__ import annotations

from typing import Dict

import torch


def build_optimizer(
    model_params: Dict[str, torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """Create optimizer with given hyperparameters."""
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model_params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 50,
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Construct learning-rate scheduler."""
    if scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if scheduler_name.lower() == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(1, warmup_epochs),
            total_iters=epochs,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
