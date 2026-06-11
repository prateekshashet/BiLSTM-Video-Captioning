import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def detect_objects_in_video(video_path, output_path, model_name='yolov8m.pt', conf_thres=0.4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run YOLOv8 object detection on a video and save the results.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save output video
        model_name (str): YOLOv8 model name or path to weights
        conf_thres (float): Confidence threshold for detections
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    # Load YOLOv8 model
    model = YOLO(model_name)
    model.to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO detection
            results = model(frame, conf=conf_thres, verbose=False)
            
            # Get the first (and only) result
            result = results[0]
            
            # Render detections on frame
            frame_with_detections = result.plot()
            
            # Write frame to output video
            out.write(frame_with_detections)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinished processing {frame_count} frames.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLOv8 object detection on a video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save output video')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='YOLOv8 model name or path to weights')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run inference on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run detection
    detect_objects_in_video(
        video_path=args.video,
        output_path=str(output_path),
        model_name=args.model,
        conf_thres=args.conf,
        device=args.device
    )
