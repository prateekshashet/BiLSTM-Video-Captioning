# train/train_caption.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import yaml
import argparse
from datetime import datetime

# Project imports
from models import BiLSTMTemporalEncoder as TemporalEncoder
from models.losses import CaptionLoss
from data.dataset import VideoCaptioningDataset, collate_fn

def setup_logging(output_dir):
    """Set up logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('train')

def build_models(config, vocab_size, device):
    """Initialize the models."""
    encoder = TemporalEncoder(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model'].get('num_layers', 1),
        dropout=config['model'].get('dropout', 0.3),
        bidirectional=config['model'].get('bidirectional', False)
    ).to(device)
    
    decoder = nn.Linear(
        config['model']['hidden_dim'],
        vocab_size
    ).to(device)
    
    return {'encoder': encoder, 'decoder': decoder}

def train_epoch(models, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    models['encoder'].train()
    models['decoder'].train()
    
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        frames = batch['frames'].to(device)
        captions = batch['captions'].to(device)
        lengths = batch['lengths'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        features = models['encoder'](frames)
        outputs = models['decoder'](features)
        
        # Calculate loss
        loss = criterion(
            predictions=outputs,
            targets=captions,
            caption_lengths=lengths,
            attn_alphas=None,  # Not using attention in this simplified version
            grounding_scores=None,
            grounding_targets=None
        )
        
        # Backward pass and optimize
        loss.total.backward()
        optimizer.step()
        
        total_loss += loss.total.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--subset', type=int, default=0, 
                      help='Use a subset of the dataset (for quick testing)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device)
    
    # Setup logging
    output_dir = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    train_dataset = VideoCaptioningDataset(
        root_dir=config['data']['root_dir'],
        annotations=config['data']['train_annotations'],
        num_frames=config['data'].get('num_frames', 8),
        frame_sampling=config['data'].get('frame_sampling', 'uniform')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_fn
    )
    
    # Build models
    models = build_models(config, vocab_size=5000, device=device)  # Assuming vocab_size=5000
    
    # Loss function and optimizer
    criterion = CaptionLoss(
        vocab_pad_idx=0,
        coverage_weight=config['training'].get('coverage_weight', 0.1),
        grounding_weight=config['training'].get('grounding_weight', 0.0)
    )
    
    optimizer = optim.Adam(
        [p for model in models.values() for p in model.parameters()],
        lr=config['training'].get('learning_rate', 0.0001)
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(models, train_loader, criterion, optimizer, device)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
