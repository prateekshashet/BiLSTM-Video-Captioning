import os
import cv2
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov8m.pt', conf_thres=0.4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize YOLO detector.
        
        Args:
            model_name: Path to YOLO weights or model name
            conf_thres: Confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.conf_thres = conf_thres
        self.model = YOLO(model_name).to(device)
        self.model.conf = conf_thres
        self.class_names = self.model.names
        
    def detect_folder(self, image_dir, output_dir, save_txt=True, save_conf=True):
        """Run detection on all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save detection results
            save_txt: Whether to save detections in YOLO format
            save_conf: Whether to save confidence scores in the output files
            
        Returns:
            List of detection results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            # Run inference
            result = self.model(img_path, verbose=False)[0]
            
            # Save detections to file
            if save_txt:
                self._save_detections(result, output_dir / f"{img_path.stem}.txt")
                
            results.append(result)
            
        return results
    
    def _save_detections(self, result, output_path):
        """Save detections to a text file in YOLO format."""
        if result.boxes is None:
            # No detections
            with open(output_path, 'w') as f:
                pass
            return
            
        with open(output_path, 'w') as f:
            for box in result.boxes:
                # Convert from xyxy to xywhn (normalized)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img_h, img_w = result.orig_shape
                
                # Calculate normalized center coordinates and dimensions
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Write class_id, x_center, y_center, width, height, confidence
                line = f"{int(box.cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {box.conf.item():.6f}\n"
                f.write(line)

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO detection on extracted frames')
    parser.add_argument('--images_dir', type=str, default='yolo_finetune/images',
                        help='Base directory containing train/val folders with images')
    parser.add_argument('--output_dir', type=str, default='yolo_finetune/detections',
                        help='Base directory to save detection results')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Path to YOLO weights or model name')
    parser.add_argument('--conf_thres', type=float, default=0.4,
                        help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Using model: {args.model}")
    print(f"Confidence threshold: {args.conf_thres}")
    
    # Initialize detector
    detector = YOLODetector(
        model_name=args.model,
        conf_thres=args.conf_thres,
        device=args.device
    )
    
    # Process train and val sets
    for split in ['train', 'val']:
        image_dir = Path(args.images_dir) / split
        output_dir = Path(args.output_dir) / split
        
        print(f"\nProcessing {split} set...")
        print(f"Input directory: {image_dir}")
        print(f"Output directory: {output_dir}")
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist, skipping...")
            continue
            
        # Run detection
        detector.detect_folder(
            image_dir=image_dir,
            output_dir=output_dir,
            save_txt=True,
            save_conf=True
        )
    
    print("\nDetection complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
