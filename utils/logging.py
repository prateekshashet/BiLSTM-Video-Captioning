import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter:
    """Keeps track of values and computes running averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricLogger:
    """Simple metric logger storing averages for TensorBoard-like reporting."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        message = ", ".join(f"{prefix}{k}: {v:.4f}" for k, v in metrics.items())
        logging.info(message)

    def get_metrics(self) -> Dict[str, float]:
        return {k: meter.avg for k, meter in self.meters.items()}


def create_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("video_captioning")


def setup_logger(output_dir: str, name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger that writes to both console and file.
    
    Args:
        output_dir: Directory to save the log file
        name: Optional name for the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create formatters and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def setup_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Set up TensorBoard writer.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        
    Returns:
        TensorBoard SummaryWriter instance
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    global_step: int,
    prefix: str = "",
    print_metrics: bool = True
) -> None:
    """
    Log metrics to TensorBoard and optionally print them.
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary of metric names and values
        global_step: Current step/epoch
        prefix: Optional prefix for metric names
        print_metrics: Whether to print metrics to console
    """
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}{name}", value, global_step)
    
    if print_metrics:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logging.info(f"{prefix[:-1].capitalize() if prefix else 'Metrics'}: {metrics_str}")


def log_grad_norm(model: torch.nn.Module, writer: SummaryWriter, global_step: int) -> None:
    """Log gradient norms for model parameters."""
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if not parameters:
        return
    
    device = parameters[0].grad.device
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2).to(device)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    writer.add_scalar("grad_norm", total_norm, global_step)
