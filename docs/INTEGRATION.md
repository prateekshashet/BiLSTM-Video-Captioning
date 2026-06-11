# Video Captioning Training Pipeline

This document provides instructions for running the video captioning training pipeline, including setup, training, and evaluation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pilot Run](#pilot-run)
6. [Full Training](#full-training)
7. [Expected Outputs](#expected-outputs)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- CUDA 11.1+ (for GPU training)
- Other dependencies in `requirements.txt`

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd video-captioning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and prepare the dataset according to `data/README.md`

## Training

### Configuration

The training configuration is specified in `configs/default_config.yaml`. Key parameters include:

```yaml
data:
  root_dir: "data/msvd"  # Path to dataset
  train_annotations: "data/msvd/annotations/train.json"
  val_annotations: "data/msvd/annotations/val.json"
  batch_size: 32
  num_workers: 4
  frame_sampling: "motion"  # or "uniform"
  num_frames: 30
  
model:
  input_dim: 2048  # Dimension of CNN features
  hidden_dim: 512
  embed_dim: 512
  num_layers: 2
  dropout: 0.4
  bidirectional: True
  
training:
  epochs: 80
  learning_rate: 1e-4
  weight_decay: 1e-5
  max_grad_norm: 5.0
  mixed_precision: True
  coverage_weight: 0.1
  grounding_weight: 0.3
  
  lr_scheduler:
    type: "cosine"  # or "step"
    min_lr: 1e-6
    step_size: 5
    gamma: 0.1
    
  scheduled_sampling:
    start_epoch: 0
    end_epoch: 40
    final_ratio: 0.5
    
logging:
  log_interval: 10
  val_interval: 1
  save_interval: 1
```

### Starting Training

To start training with default configuration:

```bash
python train/train_caption.py --config configs/default_config.yaml --device cuda:0
```

Command-line arguments:

- `--config`: Path to config file (default: 'configs/default_config.yaml')
- `--device`: Device for training ('cuda:0', 'cuda:1', 'cpu', etc.)
- `--subset`: Use a subset of the dataset (for debugging)
- `--epochs`: Number of training epochs
- `--seed`: Random seed
- `--resume`: Path to checkpoint to resume training

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=results/runs
```

Key metrics:
- `train/loss`: Total training loss
- `train/ce_loss`: Cross-entropy loss
- `train/coverage_loss`: Coverage loss
- `train/grounding_loss`: Grounding loss
- `val/loss`: Total validation loss
- `val/bleu4`: BLEU-4 score
- `val/cider`: CIDEr score

## Evaluation

To evaluate a trained model:

```bash
python eval/evaluate.py --config configs/default_config.yaml --checkpoint results/checkpoints/model_best.pth.tar --device cuda:0
```

## Pilot Run

To run a quick test with a small subset of the data:

```bash
python train/train_caption.py --config configs/default_config.yaml --device cuda:0 --subset 20 --epochs 3
```

This will:
1. Use only 20 training samples and 10 validation samples
2. Train for 3 epochs
3. Save checkpoints and logs to `results/run_<timestamp>/`

## Full Training

For full training on the entire dataset:

```bash
python train/train_caption.py --config configs/default_config.yaml --device cuda:0 --epochs 80
```

## Expected Outputs

After training, the following files will be generated in the output directory (`results/run_<timestamp>/`):

```
results/
  run_20231117_1530/
    checkpoints/
      checkpoint_epoch001.pth.tar
      checkpoint_epoch002.pth.tar
      ...
      checkpoint_latest.pth.tar
      model_best.pth.tar
    logs/
      events.out.tfevents...
    config.yaml
    train.log
    val_samples_epoch001.json
    val_samples_epoch002.json
    ...
    attention_maps/
      sample_001_epoch001.png
      sample_002_epoch001.png
      ...
```

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `batch_size` in the config
- Enable gradient checkpointing
- Use mixed precision training
- Reduce model size (hidden_dim, num_layers)

### Training is Slow

- Increase `num_workers` in DataLoader
- Use a larger batch size if possible
- Enable mixed precision training
- Use a smaller model

### Poor Performance

- Check learning rate (try lower values)
- Increase model capacity
- Add more training data
- Adjust loss weights (coverage_weight, grounding_weight)
- Check for overfitting (large gap between train and val metrics)
