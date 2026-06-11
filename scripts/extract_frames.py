import os
import cv2
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir, frame_interval=30):
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        frame_interval: Extract one frame every N frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Extracting frames from {video_path.name}:")
    print(f"  FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's the first frame or at the specified interval
        if frame_count % frame_interval == 0:
            frame_file = output_dir / f"{video_path.stem}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_count += 1
            
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...", end='\r')
    
    cap.release()
    print(f"\nExtracted {saved_count} frames to {output_dir}")
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos in a directory')
    parser.add_argument('--videos_dir', type=str, default='training_videos',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='yolo_finetune/images',
                        help='Base directory to save extracted frames')
    parser.add_argument('--frame_interval', type=int, default=30,
                        help='Extract one frame every N frames')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Ratio of training data (rest will be validation)')
    
    args = parser.parse_args()
    
    # Create output directories
    train_dir = Path(args.output_dir) / 'train'
    val_dir = Path(args.output_dir) / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(args.videos_dir).glob(f'*{ext}'))
    
    if not video_files:
        print(f"No video files found in {args.videos_dir}. Please add some video files first.")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    total_frames = 0
    for i, video_path in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {video_path.name}")
        
        # Determine if this video's frames go to train or val
        if i / len(video_files) < args.split_ratio:
            output_dir = train_dir
        else:
            output_dir = val_dir
        
        # Extract frames
        total_frames += extract_frames(video_path, output_dir, args.frame_interval)
    
    print(f"\nDone! Extracted a total of {total_frames} frames.")
    print(f"Training frames: {len(list(train_dir.glob('*.jpg')))}")
    print(f"Validation frames: {len(list(val_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    main()
