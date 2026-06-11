import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class VideoCaptioningDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform=None,
        num_frames: int = 30,
        sampling: str = "uniform",
        temporal_stride: int = 1,
        max_frames: int = 300,
        annotations_path: Optional[Union[str, Path]] = None,
        caption_key: str = "tokens",
        text_key: str = "text",
        pad_token_id: int = 0,
    ):
        """
        Args:
            root_dir: Directory containing video files
            transform: Transform to apply to frames
            num_frames: Number of frames to sample per video
            sampling: Sampling strategy ('uniform', 'keyframe', 'motion')
            temporal_stride: Stride for frame sampling
            max_frames: Maximum number of frames to process per video
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_frames = num_frames
        self.sampling = sampling
        self.temporal_stride = temporal_stride
        self.max_frames = max_frames
        
        # Find all video files
        self.video_files = list(self.root_dir.glob("*.mp4")) + list(self.root_dir.glob("*.avi"))
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {root_dir}")

        self.annotations: Dict[str, Dict[str, Union[str, List[int]]]] = {}
        self.caption_key = caption_key
        self.text_key = text_key
        self.pad_token_id = pad_token_id

        if annotations_path is not None:
            annotations_path = Path(annotations_path)
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
            with annotations_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.annotations = data
            else:
                raise ValueError("Annotation file must contain a JSON object mapping video IDs to captions")

    def __len__(self) -> int:
        return len(self.video_files)

    def _sample_frames_uniform(self, total_frames: int) -> List[int]:
        """Sample frames uniformly from video"""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            indices += [indices[-1]] * (self.num_frames - len(indices))
            return indices[:self.num_frames]
        
        step = total_frames / self.num_frames
        indices = [int(i * step) for i in range(self.num_frames)]
        return indices

    def _sample_frames_keyframe(self, cap: cv2.VideoCapture) -> List[int]:
        """Sample keyframes using OpenCV's built-in method"""
        # This is a placeholder - in practice you'd use a more sophisticated method
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._sample_frames_uniform(total_frames)

    def _sample_frames_motion(self, cap: cv2.VideoCapture) -> List[int]:
        """Sample frames based on motion detection"""
        # This is a simplified version - consider using optical flow in practice
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.num_frames)
        return list(range(0, total_frames, step))[:self.num_frames]

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select sampling strategy
        if self.sampling == "keyframe":
            frame_indices = self._sample_frames_keyframe(cap)
        elif self.sampling == "motion":
            frame_indices = self._sample_frames_motion(cap)
        else:  # uniform
            frame_indices = self._sample_frames_uniform(total_frames)

        # Read frames
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not read any frames from {video_path}")

        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames into tensor: [T, C, H, W]
        frames = torch.stack(frames)
        
        sample = {
            "video": frames,
            "video_path": str(video_path),
            "frame_indices": frame_indices,
        }

        video_id = video_path.stem
        if self.annotations:
            # Check if we have a 'videos' key in annotations (new format)
            if 'videos' in self.annotations:
                # Find the annotation for this video
                annotation = None
                for video in self.annotations['videos']:
                    if video.get('id') == video_id or video.get('file') == video_path.name:
                        annotation = video
                        break
                
                if annotation is None:
                    # Skip this video if no annotation is found
                    return None
                
                tokens = annotation.get('caption')
                caption_text = annotation.get('caption')
                
                if tokens is None:
                    # Skip if no caption is found
                    return None
                    
                # Convert caption to tokens if needed (simple space-based tokenization as fallback)
                if isinstance(tokens, str):
                    tokens = tokens.split()
                
                sample["caption_tokens"] = torch.tensor([len(tokens)], dtype=torch.long)  # Dummy token
                sample["caption_text"] = caption_text
                sample["pad_token_id"] = self.pad_token_id
            else:
                # Old format (direct mapping)
                annotation = self.annotations.get(video_id)
                if annotation is None:
                    # Skip if no annotation is found
                    return None

                if isinstance(annotation, dict):
                    tokens = annotation.get(self.caption_key)
                    caption_text = annotation.get(self.text_key, "")
                else:
                    tokens = annotation
                    caption_text = ""

                if tokens is None:
                    # Skip if no tokens are found
                    return None

                sample["caption_tokens"] = torch.tensor(tokens if isinstance(tokens, list) else [tokens], dtype=torch.long)
                sample["caption_text"] = caption_text
                sample["pad_token_id"] = self.pad_token_id

        return sample

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable length sequences.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary containing batched data with padding
    """
    # Filter out None values (videos that were skipped)
    batch = [b for b in batch if b is not None]
    if not batch:  # If batch is empty after filtering
        return {}
        
    # Handle video frames
    videos = [item["video"] for item in batch]
    video_lengths = [video.shape[0] for video in videos]
    max_length = max(video_lengths) if video_lengths else 0

    # Prepare video tensor with shape (batch, seq_len, C, H, W) and pad in time dimension
    batched_videos = torch.zeros(
        len(batch),
        max_length,
        *videos[0].shape[1:],  # C, H, W
        dtype=videos[0].dtype,
    )

    for i, video in enumerate(videos):
        batched_videos[i, : video.shape[0]] = video

    # Reshape to (batch * seq_len, C, H, W) for processing with CNN
    batch_size, seq_len, C, H, W = batched_videos.shape
    batched_videos = batched_videos.view(-1, C, H, W)
    
    # Apply any transformations that might be needed (e.g., normalization)
    # This is a placeholder - in practice, you'd apply your CNN here
    # For now, we'll just flatten the spatial dimensions to create features
    batched_videos = batched_videos.permute(0, 2, 3, 1)  # (N, H, W, C)
    batched_videos = batched_videos.reshape(batch_size, seq_len, -1)  # (batch, seq_len, H*W*C)
    
    # Handle captions if they exist
    if "caption_tokens" in batch[0]:
        captions = [item["caption_tokens"] for item in batch]
        caption_lengths = torch.tensor([len(cap) for cap in captions], dtype=torch.long)
        max_caption_length = max(caption_lengths) if len(caption_lengths) > 0 else 0
        
        # Pad captions
        padded_captions = []
        for cap in captions:
            padding = max_caption_length - len(cap)
            if padding > 0:
                padded_cap = torch.cat([
                    cap,
                    torch.full((padding,), batch[0]["pad_token_id"], dtype=torch.long)
                ])
                padded_captions.append(padded_cap)
            else:
                padded_captions.append(cap)
        
        batched_captions = torch.stack(padded_captions) if padded_captions else None
    else:
        batched_captions = None
        caption_lengths = None
    
    # Create batch dictionary
    batch_dict = {
        "videos": batched_videos,
        "video_lengths": torch.tensor(video_lengths, dtype=torch.long) if video_lengths else None,
        "video_paths": [item["video_path"] for item in batch],
        "video_ids": [Path(item["video_path"]).stem for item in batch],
        "frame_indices": [item["frame_indices"] for item in batch],
    }
    
    if batched_captions is not None:
        batch_dict.update({
            "captions": batched_captions,
            "caption_lengths": caption_lengths,
            "caption_texts": [item.get("caption_text", "") for item in batch]
        })
    
    return batch_dict

def create_dataloader(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    num_frames: int = 30,
    sampling: str = "uniform",
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for video captioning.
    
    Args:
        root_dir: Directory containing video files
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        num_frames: Number of frames to sample per video
        sampling: Sampling strategy ('uniform', 'keyframe', 'motion')
        **kwargs: Additional arguments to pass to VideoCaptioningDataset
        
    Returns:
        Configured DataLoader instance
    """
    dataset = VideoCaptioningDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        sampling=sampling,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
