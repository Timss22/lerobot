# YOLO Bounding Box Integration Guide for SmolVLA

This guide explains how to use YOLO bounding boxes as input to SmolVLA for tomato picking tasks.

## Overview

The YOLO integration allows SmolVLA to receive bounding box coordinates from YOLO detections as part of the state vector. This enables the model to see where tomatoes are located in the camera images.

## Architecture

- **YOLO Processor**: Runs YOLO on camera images to extract bounding boxes
- **State Concatenation**: Bounding boxes are concatenated to the existing state vector
- **Auto-Dimension Detection**: State dimension is automatically adjusted to accommodate bounding boxes

## Bounding Box Format

Each detection contains 5 values:
- `[x1, y1, x2, y2, confidence]`
- All coordinates are normalized to [0, 1] relative to image size
- If no detection: `[0, 0, 0, 0, 0]`

For 3 cameras with 1 detection each:
- Total bounding box dimension: 3 cameras × 1 detection × 5 values = **15 dimensions**

## Configuration

### Basic Setup

Add these parameters to your SmolVLA configuration:

```python
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

config = SmolVLAConfig(
    # Enable YOLO integration
    use_yolo_bboxes=True,
    yolo_model_path="/path/to/your/yolo11_tomato.pt",
    
    # YOLO detection parameters
    yolo_confidence_threshold=0.75,  # Minimum confidence (0.0-1.0)
    yolo_max_detections_per_camera=1,  # Top N detections per camera
    
    # Camera configuration
    yolo_camera_names=["wrist", "side", "top"],  # Camera names to process
    # Optional: camera weights for future use
    yolo_camera_weights={"wrist": 1.0, "side": 0.5, "top": 0.5},
    
    # State dimension handling
    auto_detect_state_dim=True,  # Recommended: auto-detect from dataset
    # OR manually set:
    # auto_detect_state_dim=False,
    # max_state_dim=47,  # original_state_dim (32) + bbox_dim (15)
)
```

### Command Line Usage

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.use_yolo_bboxes=true \
  --policy.yolo_model_path=/path/to/yolo11_tomato.pt \
  --policy.yolo_confidence_threshold=0.75 \
  --policy.yolo_max_detections_per_camera=1 \
  --policy.yolo_camera_names='["wrist","side","top"]' \
  --policy.auto_detect_state_dim=true \
  --dataset.repo_id=your_dataset \
  --batch_size=64 \
  --steps=200000
```

## State Dimension Calculation

### Auto-Detection (Recommended)

When `auto_detect_state_dim=True`:
1. The system reads the original state dimension from your dataset
2. Calculates bounding box dimension: `num_cameras × max_detections × 5`
3. Sets `max_state_dim = original_state_dim + bbox_dim`
4. Updates the state projection layer automatically

**Example:**
- Original state: 32 dimensions (robot joints, gripper, etc.)
- Bounding boxes: 3 cameras × 1 detection × 5 = 15 dimensions
- **Final state: 32 + 15 = 47 dimensions**

### Manual Configuration

If `auto_detect_state_dim=False`:
- You must manually set `max_state_dim` to accommodate:
  - Original state dimension
  - Plus bounding box dimension
- The system will validate that your setting is sufficient

**Example:**
```python
config = SmolVLAConfig(
    use_yolo_bboxes=True,
    auto_detect_state_dim=False,
    max_state_dim=47,  # Must be >= original_state_dim + bbox_dim
    # ... other params
)
```

## Installation

Make sure you have ultralytics installed:

```bash
pip install ultralytics
```

## Usage Examples

### Example 1: Training with YOLO

```python
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Create configuration
config = SmolVLAConfig(
    use_yolo_bboxes=True,
    yolo_model_path="models/yolo11_tomato.pt",
    yolo_confidence_threshold=0.75,
    yolo_max_detections_per_camera=1,
    yolo_camera_names=["wrist", "side", "top"],
    auto_detect_state_dim=True,
    # ... other config params
)

# Create policy (state dimension will be auto-detected)
policy = SmolVLAPolicy(config)
```

### Example 2: Inference with YOLO

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Load pretrained model
policy = SmolVLAPolicy.from_pretrained("your_model_id")

# Create processors (YOLO will run automatically if enabled in config)
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    "your_model_id"
)

# During inference, YOLO runs on images and bounding boxes are added to state
action = policy.select_action(observation)
```

## State Vector Structure

The final state vector has this structure:

```
[original_state_dims..., bbox_cam1, bbox_cam2, bbox_cam3]
```

Where each `bbox_cam*` is:
```
[x1, y1, x2, y2, confidence]
```

**Example with 3 cameras, 1 detection each:**
```
[robot_joint_1, ..., robot_joint_N, gripper, 
 x1_wrist, y1_wrist, x2_wrist, y2_wrist, conf_wrist,
 x1_side, y1_side, x2_side, y2_side, conf_side,
 x1_top, y1_top, x2_top, y2_top, conf_top]
```

## Troubleshooting

### Error: "YOLO model file not found"

**Solution:** Check that the path to your YOLO model is correct:
```python
config.yolo_model_path = "/absolute/path/to/yolo11_tomato.pt"
```

### Error: "State dimension exceeds max_state_dim"

**Solution:** Enable auto-detection or increase `max_state_dim`:
```python
config.auto_detect_state_dim = True  # Recommended
# OR
config.max_state_dim = 50  # Increase manually
```

### No detections found

**Possible causes:**
1. Confidence threshold too high - try lowering it:
   ```python
   config.yolo_confidence_threshold = 0.5
   ```
2. YOLO model not detecting tomatoes - check your model
3. Camera names don't match - verify `yolo_camera_names`

### Bounding boxes are all zeros

This is normal when no detections are found. The processor pads with zeros:
- `[0, 0, 0, 0, 0]` means no detection for that camera

## Advanced Configuration

### Multiple Detections Per Camera

To use top 3 detections per camera:

```python
config.yolo_max_detections_per_camera = 3
# Bounding box dimension: 3 cameras × 3 detections × 5 = 45 dimensions
```

### Custom Camera Weights

Camera weights are stored for future use (e.g., weighted attention):

```python
config.yolo_camera_weights = {
    "wrist": 1.0,   # Highest weight (closest camera)
    "side": 0.7,
    "top": 0.5
}
```

### Processing Specific Cameras Only

```python
config.yolo_camera_names = ["wrist"]  # Only process wrist camera
# Bounding box dimension: 1 camera × 1 detection × 5 = 5 dimensions
```

## Integration Flow

1. **Image Input**: Camera images arrive in `observation.images`
2. **YOLO Processing**: YOLO runs on each camera image
3. **Bounding Box Extraction**: Top N detections extracted per camera
4. **Normalization**: Coordinates normalized to [0, 1]
5. **State Concatenation**: Bounding boxes concatenated to state
6. **Model Forward**: SmolVLA processes state with bounding box information

## Notes

- YOLO runs **before** batch dimension is added (in processor pipeline)
- Bounding boxes are normalized to [0, 1] for better neural network training
- Missing detections are zero-padded
- The processor handles variable number of cameras gracefully
- State dimension is automatically updated when `auto_detect_state_dim=True`

## Future Enhancements

Potential improvements:
- Batch processing for YOLO (currently processes one image at a time)
- Camera-specific confidence thresholds
- Weighted bounding box aggregation
- Temporal smoothing of detections across frames

## Questions?

If you encounter issues:
1. Check that `ultralytics` is installed: `pip install ultralytics`
2. Verify YOLO model path is correct
3. Ensure camera names match your dataset
4. Check state dimension calculation matches your setup

