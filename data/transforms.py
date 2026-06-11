import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import List, Tuple, Optional, Union
import cv2

class VideoTransform:
    """Base class for video transformations."""
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class RandomHorizontalFlip(VideoTransform):
    """Randomly flip the frames horizontally with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return torch.flip(frames, dims=[-1])  # Flip width dimension
        return frames

class RandomCrop(VideoTransform):
    """Randomly crop frames to a specified size."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, h, w = frames.shape
        th, tw = self.size
        
        if w == tw and h == th:
            return frames
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return frames[..., i:i+th, j:j+tw]

class CenterCrop(VideoTransform):
    """Crop the center of the frames."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, h, w = frames.shape
        th, tw = self.size
        
        i = (h - th) // 2
        j = (w - tw) // 2
        
        return frames[..., i:i+th, j:j+tw]

class Normalize(VideoTransform):
    """Normalize frames with mean and standard deviation."""
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        self.inplace = inplace

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            frames = frames.clone()
            
        frames = (frames - self.mean) / self.std
        return frames

class ToTensor(VideoTransform):
    """Convert frames to tensor and scale to [0, 1]."""
    def __call__(self, frames: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            # Convert HWC to CHW and scale to [0, 1]
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        return frames

class Compose:
    """Compose several transforms together."""
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            frames = t(frames)
        return frames

def create_train_transforms(
    crop_size: Tuple[int, int] = (224, 224),
    hflip_p: float = 0.5,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> callable:
    """Create training transforms with data augmentation."""
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(p=hflip_p),
        RandomCrop(crop_size),
        Normalize(mean=mean, std=std)
    ])

def create_val_transforms(
    crop_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> callable:
    """Create validation transforms (no augmentation)."""
    return Compose([
        ToTensor(),
        CenterCrop(crop_size),
        Normalize(mean=mean, std=std)
    ])

def create_test_transforms(
    crop_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> callable:
    """Create test transforms (same as validation)."""
    return create_val_transforms(crop_size, mean, std)
