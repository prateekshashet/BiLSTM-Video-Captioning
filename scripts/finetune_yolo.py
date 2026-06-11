import os
import sys
import argparse
import shutil
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
from utils.config import load_config


def get_auto_device() -> str:
    """Automatically select the best available device."""
    import torch
    if torch.cuda.is_available():
        return "0"  # Default to first CUDA device
    return "cpu"  # Fall back to CPU

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on annotated frames")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=get_auto_device(), 
                        help="Device to use: 'cpu', '0' (for CUDA device 0), etc. Defaults to auto-detect")
    parser.add_argument("--override_weights", type=str, default="", help="Optional path to start from custom weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    yolo_cfg = cfg.get("yolo", {})
    finetune_cfg = yolo_cfg.get("finetune", {})

    if not finetune_cfg.get("enabled", False):
        raise SystemExit("Finetune config is disabled. Enable yolo.finetune.enabled in the config file.")

    data_yaml = finetune_cfg.get("annotations")
    if data_yaml is None:
        raise SystemExit("Missing yolo.finetune.annotations path in config")

    frames_dir = finetune_cfg.get("frames_dir")
    if frames_dir is None:
        raise SystemExit("Missing yolo.finetune.frames_dir in config")

    epochs = finetune_cfg.get("epochs", 20)
    lr = finetune_cfg.get("learning_rate", 1e-4)
    batch_size = finetune_cfg.get("batch_size", 8)
    patience = finetune_cfg.get("patience", 5)
    img_size = finetune_cfg.get("img_size", 640)
    output_dir = Path(finetune_cfg.get("output_dir", "checkpoints/yolo"))
    best_weights = Path(finetune_cfg.get("best_weights", output_dir / "yolo_best.pt"))

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = yolo_cfg.get("model", "yolov8m")
    weights_override = args.override_weights or yolo_cfg.get("weights", "")
    
    # If no weights are specified, use the base model name (will download if needed)
    if not weights_override:
        print(f"Using base model: {model_name}")
        model = YOLO(f"{model_name}.pt")
    else:
        # If weights are specified, verify they exist
        weights_path = Path(weights_override)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        print(f"Using weights from: {weights_path}")
        model = YOLO(str(weights_path))

    # Ensure all numeric values are of correct type
    # Set device to None to let YOLO auto-detect, or 'cpu' if explicitly requested
    device = args.device.lower() if args.device.lower() != 'cuda' else None
    
    train_kwargs = {
        "data": str(data_yaml),  # Ensure path is string
        "epochs": int(epochs),
        "lr0": float(lr),
        "batch": int(batch_size),
        "imgsz": int(img_size),
        "patience": int(patience),
        "device": device,  # Let YOLO handle device selection
        "project": str(output_dir),
        "name": "finetune",
        "cache": False,
        "exist_ok": True,
        "verbose": True  # Add verbose output
    }

    model.train(**{k: v for k, v in train_kwargs.items() if v is not None})

    candidate = output_dir / "finetune" / "weights" / "best.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"Expected best weights at {candidate}")

    best_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, best_weights)
    print(f"Saved fine-tuned weights to {best_weights}")


if __name__ == "__main__":
    main()
