import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import create_dataloader
from data.transforms import create_test_transforms, create_train_transforms, create_val_transforms
from detection import YOLODetector
from features.cnn_encoder import CNNEncoder
from models import BiLSTMTemporalEncoder, CaptionLoss, HierarchicalDecoder
from training.optimizer import build_optimizer, build_scheduler
from utils.config import load_config
from utils.logging import AverageMeter, MetricLogger, create_logger
from utils.cleanup import cleanup_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM-based dense video captioning model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Configuration file path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Training device")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: Dict, device: torch.device) -> Dict[str, nn.Module]:
    cnn_encoder = CNNEncoder(
        backbone=config["model"].get("cnn_backbone", "resnet50"),
        pretrained=True,
        train_backbone=config["model"].get("train_backbone", False),
    ).to(device)

    temporal_encoder = BiLSTMTemporalEncoder(
        input_dim=cnn_encoder.temporal_input_dim,
        hidden_dim=config["model"].get("hidden_size", 512),
        num_layers=config["model"].get("num_layers", 2),
        num_heads=config["model"].get("num_heads", 8),
        dropout=config["model"].get("dropout", 0.3),
        use_transformer=config["model"].get("use_transformer", False),
    ).to(device)

    decoder = HierarchicalDecoder(
        vocab_size=config["model"].get("vocab_size", 10000),
        embed_dim=config["model"].get("embed_dim", 512),
        hidden_dim=config["model"].get("hidden_size", 512),
        encoder_dim=config["model"].get("hidden_size", 512),
        dropout=config["model"].get("dropout", 0.3),
        use_coverage=config["training"].get("use_coverage", True),
        max_seq_length=config["model"].get("max_seq_length", 60),
        device=device,
    ).to(device)

    return {
        "cnn_encoder": cnn_encoder,
        "temporal_encoder": temporal_encoder,
        "decoder": decoder,
    }


def build_detector(config: Dict, device: torch.device) -> YOLODetector:
    yolo_cfg = config.get("yolo", {})
    weights_path = yolo_cfg.get("weights") or None

    detector = YOLODetector(
        model_name=yolo_cfg.get("model", "yolov8m"),
        device=device,
        conf_threshold=yolo_cfg.get("confidence_threshold", 0.4),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
        cache=yolo_cfg.get("cache", False),
        weights_path=weights_path,
    )
    detector.reset_state()
    detector.model.eval()
    return detector


def extract_batch_detections(
    videos: torch.Tensor,
    lengths: torch.Tensor,
    detector: YOLODetector,
    norm_stats: Dict[str, torch.Tensor],
) -> List[List[List[Dict]]]:
    batch_size, max_steps = videos.size(0), videos.size(1)
    lengths_cpu = lengths.detach().cpu()

    mean = norm_stats["mean"].view(3, 1, 1)
    std = norm_stats["std"].view(3, 1, 1)

    batch_detections: List[List[List[Dict]]] = []

    with torch.no_grad():
        for b in range(batch_size):
            detector.reset_state()
            video_detections: List[List[Dict]] = [[] for _ in range(max_steps)]
            seq_len = lengths_cpu[b].item()

            for t in range(seq_len):
                frame = videos[b, t].detach().cpu()
                frame_denorm = frame * std + mean
                frame_rgb = frame_denorm.permute(1, 2, 0).clamp(0.0, 1.0)
                frame_uint8 = (frame_rgb * 255.0).to(torch.uint8).numpy()

                detections = detector.detect_frame(frame_uint8, frame_idx=t)
                video_detections[t] = [
                    {
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                    }
                    for det in detections
                ]

            batch_detections.append(video_detections)
            detector.reset_state()
            cleanup_memory()

    return batch_detections


def get_teacher_forcing_ratio(epoch: int, schedule_cfg: Dict) -> float:
    start_prob = schedule_cfg.get("start_prob", 1.0)
    end_prob = schedule_cfg.get("end_prob", 0.5)
    start_epoch = schedule_cfg.get("start_epoch", 0)
    end_epoch = schedule_cfg.get("end_epoch", start_epoch)

    if epoch <= start_epoch:
        return start_prob
    if epoch >= end_epoch:
        return end_prob

    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return start_prob + (end_prob - start_prob) * progress


def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    models: Dict[str, nn.Module],
    criterion: CaptionLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: Dict,
    detector: YOLODetector,
    norm_stats: Dict[str, torch.Tensor],
    logger: MetricLogger,
) -> None:
    for model in models.values():
        model.train()

    losses = AverageMeter()

    teacher_ratio = get_teacher_forcing_ratio(epoch, config["training"].get("scheduled_sampling", {}))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
        videos = batch["videos"].to(device)
        video_lengths = batch["video_lengths"].to(device)
        captions = batch.get("captions")
        caption_lengths = batch.get("caption_lengths")
        pad_token_id = batch.get("pad_token_id", 0)

        if captions is not None:
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)

        models["temporal_encoder"].reset_state()

        # Extract detections per video and compute grounded visual features
        batch_detections = extract_batch_detections(videos, video_lengths, detector, norm_stats)

        encoder_inputs = models["cnn_encoder"](videos, detections=batch_detections)
        frame_features = encoder_inputs["combined_features"]  # [B, T, D]

        # Encode temporally
        temporal_out = models["temporal_encoder"](frame_features, lengths=video_lengths)
        encoder_outputs = temporal_out["output"]

        # Decode captions
        outputs = models["decoder"](
            encoder_outputs=encoder_outputs,
            captions=captions,
            caption_lengths=caption_lengths,
            teacher_forcing_ratio=teacher_ratio,
        )

        # Compute loss
        loss_out = criterion(
            predictions=outputs["predictions"],
            targets=captions,
            caption_lengths=caption_lengths,
            attn_alphas=outputs["alphas"],
        )

        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for model in models.values() for p in model.parameters() if p.requires_grad],
            config["training"].get("clip_grad_norm", 1.0),
        )
        optimizer.step()

        models["temporal_encoder"].detach_state()

        losses.update(loss_out.total.item(), videos.size(0))
        logger.update(loss=loss_out.total.item())

        del batch_detections
        cleanup_memory()

    scheduler.step()


def validate(
    dataloader: DataLoader,
    models: Dict[str, nn.Module],
    criterion: CaptionLoss,
    device: torch.device,
    config: Dict,
    detector: YOLODetector,
    norm_stats: Dict[str, torch.Tensor],
    logger: MetricLogger,
) -> Dict[str, float]:
    for model in models.values():
        model.eval()

    losses = AverageMeter()
    metrics = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            videos = batch["videos"].to(device)
            video_lengths = batch["video_lengths"].to(device)
            captions = batch.get("captions")
            caption_lengths = batch.get("caption_lengths")

            if captions is None:
                continue

            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            models["temporal_encoder"].reset_state()

            batch_detections = extract_batch_detections(videos, video_lengths, detector, norm_stats)

            encoder_inputs = models["cnn_encoder"](videos, detections=batch_detections)
            frame_features = encoder_inputs["combined_features"]
            temporal_out = models["temporal_encoder"](frame_features, lengths=video_lengths)
            encoder_outputs = temporal_out["output"]

            outputs = models["decoder"](
                encoder_outputs=encoder_outputs,
                captions=captions,
                caption_lengths=caption_lengths,
                teacher_forcing_ratio=0.0,
            )

            loss_out = criterion(
                predictions=outputs["predictions"],
                targets=captions,
                caption_lengths=caption_lengths,
                attn_alphas=outputs["alphas"],
            )

            losses.update(loss_out.total.item(), videos.size(0))
            del batch_detections
            cleanup_memory()

    metrics["loss"] = losses.avg
    logger.log_metrics(metrics, prefix="val/")
    return metrics


def load_checkpoint(checkpoint_path: str, models: Dict[str, nn.Module], optimizer: torch.optim.Optimizer = None, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict:
    """Load model checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint['model_state']:
            model.load_state_dict(checkpoint['model_state'][name])
    
    # Load optimizer and scheduler states if provided
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }

def save_checkpoint(epoch: int, models: Dict[str, nn.Module], optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler, best_val_loss: float,
                   output_dir: str, filename: str) -> str:
    """Save model checkpoint and return the saved path."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state': {name: model.state_dict() for name, model in models.items()},
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    
    return checkpoint_path

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = create_logger(os.path.join(args.output_dir, "train.log"))
    metric_logger = MetricLogger()

    norm_stats = {
        "mean": torch.tensor(config["data"].get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32),
        "std": torch.tensor(config["data"].get("std", [0.229, 0.224, 0.225]), dtype=torch.float32),
    }

    # Initialize starting epoch and best validation loss
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Check if we're resuming from a checkpoint
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")

    # Dataloaders
    train_transforms = create_train_transforms(
        mean=config["data"].get("mean", [0.485, 0.456, 0.406]),
        std=config["data"].get("std", [0.229, 0.224, 0.225]),
    )
    val_transforms = create_val_transforms(
        mean=config["data"].get("mean", [0.485, 0.456, 0.406]),
        std=config["data"].get("std", [0.229, 0.224, 0.225]),
    )

    train_loader = create_dataloader(
        root_dir=config["data"]["root_dir"],
        batch_size=config["data"].get("batch_size", 4),
        num_workers=config["data"].get("num_workers", 2),
        num_frames=config["data"].get("num_frames", 30),
        sampling=config["data"].get("frame_sampling", "uniform"),
        transform=train_transforms,
        annotations_path=config["data"].get("train_annotations"),
        pad_token_id=config["model"].get("pad_token_id", 0),
    )

    val_loader = create_dataloader(
        root_dir=config["data"].get("val_root_dir", config["data"]["root_dir"]),
        batch_size=config["data"].get("batch_size", 4),
        num_workers=config["data"].get("num_workers", 2),
        num_frames=config["data"].get("num_frames", 30),
        sampling=config["data"].get("frame_sampling", "uniform"),
        transform=val_transforms,
        annotations_path=config["data"].get("val_annotations"),
        pad_token_id=config["model"].get("pad_token_id", 0),
    )

    models = build_model(config, device)
    detector = build_detector(config, device)

    criterion = CaptionLoss(
        vocab_pad_idx=config["model"].get("pad_token_id", 0),
        coverage_weight=config["training"].get("coverage_weight", 0.5),
        grounding_weight=config["training"].get("grounding_weight", 0.5),
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    ).to(device)

    optimizer = build_optimizer(
        model_params=[
            {"params": model.parameters()} for model in models.values()
            if any(param.requires_grad for param in model.parameters())
        ],
        learning_rate=config["training"].get("learning_rate", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-5),
        optimizer_name=config["training"].get("optimizer", "adamw"),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"].get("scheduler", "cosine"),
        epochs=config["training"].get("epochs", 50),
        warmup_epochs=config["training"].get("warmup_epochs", 5),
    )

    # Load checkpoint if resuming
    if args.resume and os.path.isfile(args.resume):
        # Build models first
        models = build_model(config, device)
        detector = build_detector(config, device)
        optimizer = build_optimizer(
            model_params=[
                {"params": model.parameters()} for model in models.values()
                if any(param.requires_grad for param in model.parameters())
            ],
            learning_rate=config["training"].get("learning_rate", 1e-4),
            weight_decay=config["training"].get("weight_decay", 1e-5),
            optimizer_name=config["training"].get("optimizer", "adamw"),
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=config["training"].get("scheduler", "cosine"),
            epochs=config["training"].get("epochs", 50),
            warmup_epochs=config["training"].get("warmup_epochs", 5),
        )
        
        # Load checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        for name, model in models.items():
            if name in checkpoint['model_state']:
                model.load_state_dict(checkpoint['model_state'][name])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {start_epoch-1} with best val loss {best_val_loss:.4f}")
    else:
        # Initialize new models if not resuming
        models = build_model(config, device)
        detector = build_detector(config, device)
        optimizer = build_optimizer(
            model_params=[
                {"params": model.parameters()} for model in models.values()
                if any(param.requires_grad for param in model.parameters())
            ],
            learning_rate=config["training"].get("learning_rate", 1e-4),
            weight_decay=config["training"].get("weight_decay", 1e-5),
            optimizer_name=config["training"].get("optimizer", "adamw"),
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=config["training"].get("scheduler", "cosine"),
            epochs=config["training"].get("epochs", 50),
            warmup_epochs=config["training"].get("warmup_epochs", 5),
        )
        logger.info("Initialized new models")

    for epoch in range(start_epoch, config["training"].get("epochs", 50) + 1):
        train_one_epoch(
            epoch,
            train_loader,
            models,
            criterion,
            optimizer,
            scheduler,
            device,
            config,
            detector,
            norm_stats,
            metric_logger,
        )

        metrics = validate(
            val_loader,
            models,
            criterion,
            device,
            config,
            detector,
            norm_stats,
            metric_logger,
        )

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            checkpoint_path = os.path.join(args.output_dir, f"best_model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": {name: model.state_dict() for name, model in models.items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Also save a latest checkpoint
            latest_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state": {name: model.state_dict() for name, model in models.items()},
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, latest_path)

        cleanup_memory()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
