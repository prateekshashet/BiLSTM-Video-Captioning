# YOLO Fine-tuning Dataset Setup

## Directory Structure
```
yolo_finetune/
├── annotations.yaml    # Dataset configuration
├── images/
│   ├── train/         # Training images
│   └── val/           # Validation images
└── labels/
    ├── train/         # Training labels in YOLO format
    └── val/           # Validation labels in YOLO format
```

## Setup Instructions

1. **Prepare your dataset**:
   - Place your training images in `yolo_finetune/images/train/`
   - Place your validation images in `yolo_finetune/images/val/`
   - Place corresponding label files in `yolo_finetune/labels/train/` and `yolo_finetune/labels/val/`

2. **Update `annotations.yaml`**:
   - Update the `train` and `val` paths if you used different directory names
   - Update `nc` with your number of classes
   - Update the `names` list with your class names

3. **Label Format**:
   - Each image should have a corresponding `.txt` file with the same name
   - Each line in the label file should be in YOLO format: `class_id x_center y_center width height`
   - All values should be normalized (0-1)

4. **Run the fine-tuning script**:
   ```bash
   python scripts/finetune_yolo.py --config configs/default_config.yaml
   ```

## Example Label File
For an image with two objects:
```
0 0.5 0.5 0.2 0.3  # class 0, center at (0.5, 0.5), width=0.2, height=0.3
1 0.7 0.3 0.1 0.1  # class 1, center at (0.7, 0.3), width=0.1, height=0.1
```
