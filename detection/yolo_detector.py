import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import cv2
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml


class Tracker:
    """Simple IoU-based tracker to avoid cross-video state leakage."""

    def __init__(self, max_track_history: int = 10) -> None:
        self.max_track_history = max_track_history
        self.reset()

    def reset(self) -> None:
        self.track_history: Dict[int, Dict[str, Union[int, float, List[Dict[str, Union[int, float]]]]]] = {}
        self.next_track_id: int = 0

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        for track_id in list(self.track_history.keys()):
            if frame_idx - self.track_history[track_id]['last_seen'] > self.max_track_history:
                del self.track_history[track_id]

        for det in detections:
            bbox = det['bbox']
            best_iou = 0.3
            best_match = None
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            det_rect = [x1, y1, w, h]

            for track_id, track in self.track_history.items():
                if track['class_id'] != det['class_id']:
                    continue

                track_rect = track['last_bbox']
                iou = YOLODetector._calculate_iou(det_rect, track_rect)
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id

            if best_match is not None:
                track_id = best_match
                self.track_history[track_id].update({
                    'last_bbox': det_rect,
                    'last_seen': frame_idx,
                    'history': self.track_history[track_id]['history'] + [{
                        'frame_idx': frame_idx,
                        'bbox': det_rect,
                        'confidence': det['confidence']
                    }]
                })
            else:
                track_id = self.next_track_id
                self.track_history[track_id] = {
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'last_bbox': det_rect,
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'history': [{
                        'frame_idx': frame_idx,
                        'bbox': det_rect,
                        'confidence': det['confidence']
                    }]
                }
                self.next_track_id += 1

            det['track_id'] = track_id

        return detections

class YOLODetector:
    """
    YOLO-based object detector for video frames with tracking support.
    Handles object detection, tracking, and temporal smoothing.
    """
    
    def __init__(self, 
                 model_name: str = 'yolov8x',
                 device: Optional[torch.device] = None,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 cache: bool = False,
                 weights_path: Optional[Union[str, Path]] = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8x', 'yolov5s')
            device: Device to run the model on (cuda/cpu)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_name = model_name
        self.cache = cache
        self.weights_path = Path(weights_path) if weights_path is not None else None
        
        # Initialize YOLO model
        self._init_model()
        
        # For tracking objects across frames
        self.max_track_history = 10  # Number of frames to keep in track history
        self.tracker = Tracker(self.max_track_history)
    
    def _init_model(self):
        """Initialize YOLO model based on the specified version."""
        try:
            if 'yolov8' in self.model_name.lower():
                from ultralytics import YOLO
                weights = self.weights_path if self.weights_path is not None else f'{self.model_name}.pt'
                self.model = YOLO(str(weights)).to(self.device)
                self.model.overrides['cache'] = self.cache
                self.model_type = 'yolov8'
            else:  # Default to YOLOv5
                if self.weights_path is not None:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.weights_path))
                else:
                    self.model = torch.hub.load('ultralytics/yolov5', 
                                              self.model_name, 
                                              pretrained=True)
                self.model = self.model.to(self.device)
                self.model_type = 'yolov5'
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'optimizer'):  # pragma: no cover
                    self.model.model.optimizer = None
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'cache'):
                    self.model.model.cache = self.cache
                if hasattr(self.model, 'cache'):
                    self.model.cache = self.cache
                
            self.model.eval()
            print(f"Loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            raise ImportError(f"Failed to initialize YOLO model: {str(e)}")

    def reset_state(self) -> None:
        """Reset tracking history and any internal buffers between videos."""
        self.tracker = Tracker(self.max_track_history)
    
    @torch.no_grad()
    def detect_frame(self, 
                    frame: np.ndarray, 
                    frame_idx: int = 0) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame (H, W, 3) in RGB format
            frame_idx: Frame index for tracking
            
        Returns:
            List of detection dictionaries, each containing:
                - bbox: [x1, y1, x2, y2] in absolute coordinates
                - confidence: Detection confidence
                - class_id: Class ID
                - class_name: Class name
                - track_id: Assigned track ID (if tracking)
        """
        # Convert frame to RGB if needed
        if frame.shape[-1] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Run inference
        if self.model_type == 'yolov8':
            results = self.model(frame, 
                               conf=self.conf_threshold, 
                               iou=self.iou_threshold,
                               verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(class_id),
                        'class_name': self.model.names[int(class_id)],
                    })
        else:  # YOLOv5
            results = self.model(frame[..., ::-1])  # YOLOv5 expects BGR
            pred = results.xyxy[0].cpu().numpy()
            
            detections = []
            for *xyxy, conf, cls in pred:
                if conf >= self.conf_threshold:
                    detections.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.model.names[int(cls)],
                    })
        
        # Update tracking information
        detections = self.tracker.update(detections, frame_idx)

        return detections
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return max(0.0, min(1.0, iou))
    
    def process_video(self, 
                     video_path: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     show: bool = False) -> Dict:
        """
        Process a video file and detect objects in each frame.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save output visualizations
            show: Whether to display the results
            
        Returns:
            Dictionary with detection results per frame
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
            
        # Reset state per video to avoid leakage
        self.reset_state()

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if saving output
        if output_dir is not None:
            output_path = output_dir / f"{video_path.stem}_detections.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process each frame
        results = {}
        frame_idx = 0
        
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect objects
                detections = self.detect_frame(frame_rgb, frame_idx)
                results[frame_idx] = detections
                
                # Draw detections
                vis_frame = self.visualize_detections(frame_rgb.copy(), detections)
                
                # Convert back to BGR for display/saving
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                
                # Save frame
                if output_dir is not None:
                    out.write(vis_frame)
                
                # Show frame
                if show:
                    cv2.imshow('Detections', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        if output_dir is not None:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        return results
    
    @staticmethod
    def visualize_detections(frame: np.ndarray, 
                           detections: List[Dict],
                           show_labels: bool = True,
                           show_conf: bool = True) -> np.ndarray:
        """
        Draw detections on a frame.
        
        Args:
            frame: Input frame (RGB)
            detections: List of detection dictionaries
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            
        Returns:
            Frame with drawn detections (RGB)
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'][:4])
            class_name = det['class_name']
            conf = det['confidence']
            track_id = det.get('track_id', -1)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = []
            if show_labels:
                label.append(f"{class_name}")
            if show_conf:
                label.append(f"{conf:.2f}")
            if track_id >= 0:
                label.append(f"ID:{track_id}")
                
            if label:
                label = " ".join(label)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(vis_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_frame
