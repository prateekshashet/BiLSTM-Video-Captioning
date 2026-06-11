import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import torch.nn.functional as F
import gc
import sys
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any, Union

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import BiLSTMTemporalEncoder as TemporalEncoder, HierarchicalDecoder, GroundingModule
from models.losses import CaptionLoss
from data.dataset import VideoCaptioningDataset, collate_fn
from utils.metrics import compute_bleu, compute_cider


def setup_logger(output_dir: str, name: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger that writes to both console and file.
    
    Args:
        output_dir: Directory to save the log file
        name: Optional name for the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or 'video_captioning')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

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
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    
    if print_metrics and logging.getLogger('video_captioning').isEnabledFor(logging.INFO):
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logging.getLogger('video_captioning').info(f"{prefix[:-1].capitalize() if prefix else 'Metrics'}: {metrics_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train video captioning model")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device for training')
    parser.add_argument('--subset', type=int, default=0,
                        help='Use a subset of the dataset (for debugging)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_models(config: Dict[str, Any], vocab_size: int, device: torch.device) -> Dict[str, nn.Module]:
    """Initialize all model components."""
    models = {}

    # Compute encoder output dimension: if encoder is bidirectional, outputs = hidden_dim * 2
    hidden_dim = config['model'].get('hidden_dim', config['model'].get('hidden_size'))
    bidirectional = bool(config['model'].get('bidirectional', False))
    encoder_dim = int(hidden_dim * (2 if bidirectional else 1))

    # Initialize encoder
    models['encoder'] = TemporalEncoder(
        input_dim=config['model']['input_dim'],
        hidden_dim=hidden_dim,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    # Initialize decoder with parameters matching the class definition
    models['decoder'] = HierarchicalDecoder(
        vocab_size=vocab_size,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=hidden_dim,
        encoder_dim=encoder_dim,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_coverage=bool(config['training'].get('coverage_weight', 0.1) > 0),
        max_seq_length=config.get('max_seq_length', 100),
        device=device
    ).to(device)

    # Initialize grounding module with correct parameter names
    models['grounding'] = GroundingModule(
        decoder_dim=hidden_dim,  # Changed from hidden_dim to decoder_dim
        object_dim=config['model']['object_feat_dim'],  # Changed from object_feat_dim to object_dim
        hidden_dim=hidden_dim,  # Added hidden_dim with same value as decoder_dim
        dropout=config['model']['dropout']
    ).to(device)

    return models



def create_data_loaders(config: Dict[str, Any], subset: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    # Define transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),  # Resize to a standard size
        torchvision.transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        torchvision.transforms.Normalize(
            mean=config['data']['mean'],
            std=config['data']['std']
        )
    ])
    
    # Create datasets
    train_dataset = VideoCaptioningDataset(
        root_dir=config['data']['root_dir'],
        annotations_path=config['data']['train_annotations'],
        num_frames=config['data']['num_frames'],
        sampling=config['data']['frame_sampling'],
        temporal_stride=config['data']['temporal_stride'],
        transform=transform
    )
    
    val_dataset = VideoCaptioningDataset(
        root_dir=config['data']['root_dir'],
        annotations_path=config['data']['val_annotations'],
        num_frames=config['data']['num_frames'],
        sampling='uniform',  # Always use uniform for validation
        temporal_stride=config['data']['temporal_stride'],
        transform=transform
    )
    
    # Use subset if specified (for debugging)
    if subset > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(subset, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(subset // 2, len(val_dataset))))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(
    epoch: int,
    models: Dict[str, nn.Module],
    train_loader: DataLoader,
    criterion: CaptionLoss,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    logger: Any,
    writer: Any,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
    global_step: int = 0
) -> Tuple[float, int]:
    """Train for one epoch."""
    models['encoder'].train()
    models['decoder'].train()
    models['grounding'].train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cov_loss = 0.0
    total_ground_loss = 0.0
    
    # Calculate scheduled sampling ratio
    if epoch < config['training']['scheduled_sampling']['start_epoch']:
        teacher_forcing_ratio = 1.0
    elif epoch > config['training']['scheduled_sampling']['end_epoch']:
        teacher_forcing_ratio = config['training']['scheduled_sampling']['final_ratio']
    else:
        # Linear decay
        progress = (epoch - config['training']['scheduled_sampling']['start_epoch']) / \
                  (config['training']['scheduled_sampling']['end_epoch'] - config['training']['scheduled_sampling']['start_epoch'])
        teacher_forcing_ratio = 1.0 - progress * (1.0 - config['training']['scheduled_sampling']['final_ratio'])
    
    # Training loop
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        videos = batch['videos'].to(device, non_blocking=True)
        captions = batch['captions'].to(device, non_blocking=True)
        caption_lengths = batch['caption_lengths'].to(device, non_blocking=True)
        video_ids = batch['video_ids']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        with autocast(enabled=config['training'].get('mixed_precision', False)):
            # Encode video
            encoder_outputs = models['encoder'](videos)
            
            # Decode captions
            decoder_outputs = models['decoder'](
                encoder_outputs=encoder_outputs,
                captions=captions,
                caption_lengths=caption_lengths,
                teacher_forcing_ratio=teacher_forcing_ratio,
                device=device
            )
            
            # Compute grounding logits
            grounding_logits = models['grounding'](
                decoder_hidden=decoder_outputs['hidden_states'],
                object_features=encoder_outputs['object_features'],
                attention_weights=decoder_outputs.get('alphas'),
                video_lengths=encoder_outputs['video_lengths']
            )
            
            # Compute loss
            loss_output = criterion(
                predictions=decoder_outputs['predictions'],
                targets=captions,
                caption_lengths=caption_lengths,
                attn_alphas=decoder_outputs.get('alphas'),
                grounding_scores=grounding_logits,
                grounding_targets=None  # TODO: Add grounding targets
            )
            
            loss = loss_output['total']
        
        # Backward pass
        if config['training'].get('mixed_precision', False):
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training'].get('max_grad_norm', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for model in models.values() for p in model.parameters()],
                    config['training']['max_grad_norm']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config['training'].get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for model in models.values() for p in model.parameters()],
                    config['training']['max_grad_norm']
                )
            
            optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        total_ce_loss += loss_output['ce_loss'].item()
        total_cov_loss += loss_output['coverage_loss'].item() if loss_output['coverage_loss'] is not None else 0.0
        total_ground_loss += loss_output['grounding_loss'].item() if loss_output['grounding_loss'] is not None else 0.0
        
        # Log metrics
        if batch_idx % config['logging']['log_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to console
            log_str = (
                f"Epoch: {epoch} | "
                f"Batch: {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"CE: {loss_output['ce_loss'].item():.4f} | "
                f"Cov: {loss_output['coverage_loss'].item() if loss_output['coverage_loss'] is not None else 0.0:.4f} | "
                f"Ground: {loss_output['grounding_loss'].item() if loss_output['grounding_loss'] is not None else 0.0:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"TF Ratio: {teacher_forcing_ratio:.2f}"
            )
            logger.info(log_str)
            
            # Log to tensorboard
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/ce_loss', loss_output['ce_loss'].item(), global_step)
            if loss_output['coverage_loss'] is not None:
                writer.add_scalar('train/coverage_loss', loss_output['coverage_loss'].item(), global_step)
            if loss_output['grounding_loss'] is not None:
                writer.add_scalar('train/grounding_loss', loss_output['grounding_loss'].item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)
            writer.add_scalar('train/teacher_forcing_ratio', teacher_forcing_ratio, global_step)
        
        global_step += 1
        
        # Free memory
        del videos, captions, caption_lengths, encoder_outputs, decoder_outputs, loss_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate epoch metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_cov_loss = total_cov_loss / num_batches
    avg_ground_loss = total_ground_loss / num_batches
    
    # Log epoch summary
    logger.info(f"Epoch {epoch} Train Summary:")
    logger.info(f"  Loss: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | Cov: {avg_cov_loss:.4f} | Ground: {avg_ground_loss:.4f}")
    
    return avg_loss, global_step


def validate(
    epoch: int,
    models: Dict[str, nn.Module],
    val_loader: DataLoader,
    criterion: CaptionLoss,
    device: torch.device,
    logger: Any,
    writer: Any,
    config: Dict[str, Any],
    global_step: int
) -> Tuple[float, float]:
    """Validate the model."""
    models['encoder'].eval()
    models['decoder'].eval()
    models['grounding'].eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_cov_loss = 0.0
    total_ground_loss = 0.0
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            videos = batch['videos'].to(device, non_blocking=True)
            captions = batch['captions'].to(device, non_blocking=True)
            caption_lengths = batch['caption_lengths'].to(device, non_blocking=True)
            video_ids = batch['video_ids']
            
            # Forward pass
            with autocast(enabled=config['training'].get('mixed_precision', False)):
                # Encode video
                encoder_outputs = models['encoder'](videos)
                
                # Decode captions (teacher forcing ratio = 0 for validation)
                decoder_outputs = models['decoder'](
                    encoder_outputs=encoder_outputs,
                    captions=captions,
                    caption_lengths=caption_lengths,
                    teacher_forcing_ratio=0.0,
                    device=device
                )
                
                # Compute grounding logits
                grounding_logits = models['grounding'](
                    decoder_hidden=decoder_outputs['hidden_states'],
                    object_features=encoder_outputs['object_features'],
                    attention_weights=decoder_outputs.get('alphas'),
                    video_lengths=encoder_outputs['video_lengths']
                )
                
                # Compute loss
                loss_output = criterion(
                    predictions=decoder_outputs['predictions'],
                    targets=captions,
                    caption_lengths=caption_lengths,
                    attn_alphas=decoder_outputs.get('alphas'),
                    grounding_scores=grounding_logits,
                    grounding_targets=None  # TODO: Add grounding targets
                )
                
                loss = loss_output['total']
            
            # Update metrics
            total_loss += loss.item()
            total_ce_loss += loss_output['ce_loss'].item()
            total_cov_loss += loss_output['coverage_loss'].item() if loss_output['coverage_loss'] is not None else 0.0
            total_ground_loss += loss_output['grounding_loss'].item() if loss_output['grounding_loss'] is not None else 0.0
            
            # Decode predictions
            _, preds = torch.max(decoder_outputs['predictions'], dim=2)  # (batch_size, max_len)
            
            # Convert predictions and references to tokens
            for i in range(preds.size(0)):
                # Get valid tokens (remove padding and EOS tokens)
                pred = preds[i].cpu().numpy()
                ref = captions[i].cpu().numpy()
                
                # Remove padding and EOS tokens
                pred = [token for token in pred if token not in [0, 2]]  # 0=PAD, 2=EOS
                ref = [token for token in ref if token not in [0, 2]]  # 0=PAD, 2=EOS
                
                all_predictions.append(pred)
                all_references.append([ref])  # Note: references is a list of lists for each prediction
            
            # Free memory
            del videos, captions, caption_lengths, encoder_outputs, decoder_outputs, loss_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate metrics
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_cov_loss = total_cov_loss / num_batches
    avg_ground_loss = total_ground_loss / num_batches
    
    # Compute BLEU and CIDEr scores
    bleu4 = compute_bleu(all_references, all_predictions)
    cider = compute_cider(all_references, all_predictions)
    
    # Log validation summary
    logger.info(f"Epoch {epoch} Validation Summary:")
    logger.info(f"  Loss: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | Cov: {avg_cov_loss:.4f} | Ground: {avg_ground_loss:.4f}")
    logger.info(f"  BLEU-4: {bleu4:.4f} | CIDEr: {cider:.4f}")
    
    # Log to tensorboard
    writer.add_scalar('val/loss', avg_loss, global_step)
    writer.add_scalar('val/ce_loss', avg_ce_loss, global_step)
    if avg_cov_loss > 0:
        writer.add_scalar('val/coverage_loss', avg_cov_loss, global_step)
    if avg_ground_loss > 0:
        writer.add_scalar('val/grounding_loss', avg_ground_loss, global_step)
    writer.add_scalar('val/bleu4', bleu4, global_step)
    writer.add_scalar('val/cider', cider, global_step)
    
    return avg_loss, cider


def save_checkpoint(
    epoch: int,
    models: Dict[str, nn.Module],
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_val_loss: float,
    best_val_cider: float,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
) -> None:
    """Save model checkpoint."""
    state = {
        'epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'best_val_cider': best_val_cider,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    
    # Add model states
    for name, model in models.items():
        state[name] = model.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    # If this is the best model, save it separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    models: Dict[str, nn.Module],
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Tuple[int, float, float]:
    """Load model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint:
            model.load_state_dict(checkpoint[name])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Return epoch and best metrics
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('best_val_loss', float('inf')),
        checkpoint.get('best_val_cider', 0.0)
    )


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    device = torch.device(args.device)
    
    # Setup logging
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = setup_logger(log_dir, 'video_captioning')
    writer = setup_tensorboard(log_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Logging to directory: {os.path.abspath(log_dir)}")
    logger.info(f"Configuration: {config}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use sanity_check.output_dir from config or default to 'outputs' if not specified
    base_output_dir = config.get('sanity_check', {}).get('output_dir', 'outputs')
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup tensorboard
    writer = setup_tensorboard(logs_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, subset=args.subset)

    # Infer visual feature dimensionality from a sample batch so the temporal encoder
    # receives inputs with a matching input_dim.
    sample_batch = next(iter(train_loader))
    sample_videos = sample_batch["videos"]  # shape: (batch, seq_len, feature_dim)
    feature_dim = sample_videos.size(-1)
    config['model']['input_dim'] = int(feature_dim)

    # Get vocabulary size from dataset
    vocab_size = train_loader.dataset.vocab_size if hasattr(train_loader.dataset, 'vocab_size') else config['model']['vocab_size']
    
    # Build models
    models = build_models(config, vocab_size, device)
    
    # Create loss function
    criterion = CaptionLoss(
        vocab_pad_idx=0,  # Assuming 0 is the padding index
        coverage_weight=config['training'].get('coverage_weight', 0.0),
        grounding_weight=config['training'].get('grounding_weight', 0.0),
        label_smoothing=config['training'].get('label_smoothing', 0.0)
    )
    
    # Create optimizer
    params = []
    for model in models.values():
        params.extend([p for p in model.parameters() if p.requires_grad])
    
    # Convert string values to float if needed
    learning_rate = float(config['training']['learning_rate']) if isinstance(config['training']['learning_rate'], str) else config['training']['learning_rate']
    weight_decay = float(config['training']['weight_decay']) if isinstance(config['training']['weight_decay'], str) else config['training']['weight_decay']
    
    optimizer = optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    if config['training']['lr_scheduler']['type'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr_scheduler']['min_lr']
        )
    elif config['training']['lr_scheduler']['type'] == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['training']['lr_scheduler']['step_size'],
            gamma=config['training']['lr_scheduler']['gamma']
        )
    else:
        scheduler = None
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_cider = 0.0
    
    if args.resume:
        try:
            logger.info(f"Loading checkpoint from '{args.resume}'")
            start_epoch, best_val_loss, best_val_cider = load_checkpoint(
                args.resume, models, optimizer, scheduler)
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting from scratch")
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config['training'].get('mixed_precision', False))
    
    # Training loop
    global_step = 0
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train for one epoch
        train_loss, global_step = train_epoch(
            epoch=epoch,
            models=models,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler if config['training']['lr_scheduler']['type'] == 'cosine' else None,
            device=device,
            logger=logger,
            writer=writer,
            config=config,
            scaler=scaler,
            global_step=global_step
        )
        
        # Step the scheduler if using step scheduler
        if config['training']['lr_scheduler']['type'] == 'step':
            scheduler.step()
        
        # Validate
        val_loss, val_cider = validate(
            epoch=epoch,
            models=models,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            writer=writer,
            config=config,
            global_step=global_step
        )
        
        # Check if this is the best model based on validation CIDEr
        is_best = val_cider > best_val_cider
        if is_best:
            best_val_cider = val_cider
            best_val_loss = min(best_val_loss, val_loss)
        
        # Save checkpoint
        save_checkpoint(
            epoch=epoch,
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            best_val_cider=best_val_cider,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename=f'checkpoint_epoch{epoch:03d}.pth.tar'
        )
        
        # Save latest checkpoint
        save_checkpoint(
            epoch=epoch,
            models=models,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            best_val_cider=best_val_cider,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename='checkpoint_latest.pth.tar'
        )
        
        # Log best metrics
        logger.info(f"Best Val Loss: {best_val_loss:.4f} | Best Val CIDEr: {best_val_cider:.4f}")
    
    # Close tensorboard writer
    writer.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
