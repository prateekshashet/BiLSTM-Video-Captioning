import os
import json
from pathlib import Path
from typing import List, Dict, Any
import random

def find_video_files(video_dir: str) -> List[str]:
    """Find all video files in the specified directory."""
    video_extensions = {'.mp4', '.mov', '.mkv', '.avi'}
    video_files = []
    
    for root, _, files in os.walk(video_dir):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(file)
    
    # Sort files to ensure consistent ordering
    video_files.sort()
    return video_files

def create_annotation_entry(video_file: str, video_id: int) -> Dict[str, Any]:
    """Create an annotation entry for a single video."""
    video_name = Path(video_file).stem
    return {
        "id": f"video{video_id}",
        "file": video_file,
        "caption": f"Placeholder caption for {video_name}."
    }

def split_train_val(video_files: List[str], train_ratio: float = 0.8) -> tuple:
    """Split video files into training and validation sets."""
    # Sort files to ensure consistent splitting
    video_files_sorted = sorted(video_files)
    split_idx = int(len(video_files_sorted) * train_ratio)
    return video_files_sorted[:split_idx], video_files_sorted[split_idx:]

def create_annotations(video_dir: str, output_dir: str) -> None:
    """Create train and validation annotation files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = find_video_files(video_dir)
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    # Split into train and validation sets
    train_files, val_files = split_train_val(video_files)
    
    # Create annotation entries
    train_annotations = {
        "videos": [create_annotation_entry(f, i) for i, f in enumerate(train_files, 1)]
    }
    
    val_annotations = {
        "videos": [create_annotation_entry(f, i + len(train_files)) for i, f in enumerate(val_files, 1)]
    }
    
    # Write to files
    train_path = os.path.join(output_dir, "train_annotations.json")
    val_path = os.path.join(output_dir, "val_annotations.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    # Print summary
    print(f"Created annotation files in {output_dir}")
    print(f"Number of training samples: {len(train_annotations['videos'])}")
    print(f"Number of validation samples: {len(val_annotations['videos'])}")
    
    print("\nFirst 2 training examples:")
    for i, video in enumerate(train_annotations["videos"][:2], 1):
        print(f"{i}. ID: {video['id']}, File: {video['file']}, Caption: {video['caption']}")
    
    print("\nFirst 2 validation examples:")
    for i, video in enumerate(val_annotations["videos"][:2], 1):
        print(f"{i}. ID: {video['id']}, File: {video['file']}, Caption: {video['caption']}")

if __name__ == "__main__":
    # Paths
    video_dir = "training videos"  # Relative to the script location
    output_dir = "annotations"
    
    # Create annotations
    create_annotations(video_dir, output_dir)
