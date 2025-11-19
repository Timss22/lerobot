#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""YOLO processor step for extracting bounding boxes from images and concatenating to state."""

from typing import Any

import torch
from ultralytics import YOLO

from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@ProcessorStepRegistry.register(name="yolo_bbox_processor")
class YOLOBoundingBoxProcessorStep(ObservationProcessorStep):
    """
    Processor step that runs YOLO on images to extract bounding boxes and concatenates them to state.
    
    This processor:
    1. Runs YOLO on each camera image
    2. Extracts top N detections per camera (with confidence threshold)
    3. Normalizes bounding box coordinates to [0, 1]
    4. Concatenates bounding boxes to the state vector
    
    Bounding box format per detection: [x1, y1, x2, y2, confidence]
    All coordinates are normalized to [0, 1] relative to image size.
    """

    def __init__(
        self,
        yolo_model_path: str,
        confidence_threshold: float = 0.75,
        max_detections_per_camera: int = 1,
        camera_names: list[str] | None = None,
        camera_weights: dict[str, float] | None = None,
        device: str | None = None,
    ):
        """
        Initialize the YOLO bounding box processor.
        
        Args:
            yolo_model_path: Path to the YOLO model file (.pt)
            confidence_threshold: Minimum confidence score for detections (default: 0.75)
            max_detections_per_camera: Maximum number of detections per camera (default: 1)
            camera_names: List of camera names to process. If None, processes all cameras in observation.images
            camera_weights: Optional dictionary mapping camera names to weights (for future use)
            device: Device to run YOLO on ('cuda', 'cpu', 'mps', or None for auto)
        """
        super().__init__()
        self.yolo_model_path = yolo_model_path
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_camera = max_detections_per_camera
        self.camera_names = camera_names
        self.camera_weights = camera_weights or {}
        self.device = device
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(yolo_model_path)
            if device:
                # YOLO will use the device automatically, but we can set it explicitly
                self.yolo_model.to(device)
        except Exception as e:
            raise ValueError(
                f"Failed to load YOLO model from {yolo_model_path}. "
                f"Make sure ultralytics is installed: pip install ultralytics. Error: {e}"
            )
        
        # Each detection has: [x1, y1, x2, y2, confidence] = 5 values
        self.bbox_dim_per_detection = 5
        self.bbox_dim_per_camera = max_detections_per_camera * self.bbox_dim_per_detection

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization."""
        return {
            "yolo_model_path": self.yolo_model_path,
            "confidence_threshold": self.confidence_threshold,
            "max_detections_per_camera": self.max_detections_per_camera,
            "camera_names": self.camera_names,
            "camera_weights": self.camera_weights,
            "device": self.device,
        }

    def _extract_bboxes_from_image(
        self, image: torch.Tensor, image_height: int, image_width: int
    ) -> torch.Tensor:
        """
        Extract bounding boxes from a single image using YOLO.
        
        Args:
            image: Image tensor of shape (C, H, W) or (H, W, C)
            image_height: Height of the image
            image_width: Width of the image
            
        Returns:
            Tensor of shape (max_detections_per_camera, 5) containing:
            [x1_norm, y1_norm, x2_norm, y2_norm, confidence] for each detection.
            If fewer detections, remaining rows are zero-padded.
        """
        # Convert tensor to numpy for YOLO
        # Handle different tensor formats: (C, H, W) or (H, W, C)
        if image.dim() == 3:
            if image.shape[0] == 3:  # (C, H, W)
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # (H, W, C)
                image_np = image.cpu().numpy()
        else:
            raise ValueError(f"Expected 3D image tensor, got shape {image.shape}")
        
        # Ensure image is in uint8 format [0, 255]
        if image_np.dtype != "uint8":
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype("uint8")
            else:
                image_np = image_np.astype("uint8")
        
        # Run YOLO inference
        results = self.yolo_model(image_np, verbose=False)
        
        # Extract detections
        bboxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Filter by confidence and get top detections
            conf_mask = boxes.conf >= self.confidence_threshold
            if conf_mask.any():
                # Get boxes with sufficient confidence
                valid_boxes = boxes[conf_mask]
                
                # Sort by confidence (descending) and take top N
                confidences = valid_boxes.conf.cpu().numpy()
                sorted_indices = confidences.argsort()[::-1][: self.max_detections_per_camera]
                
                for idx in sorted_indices:
                    box = valid_boxes[idx]
                    # Get bounding box coordinates (x1, y1, x2, y2) in pixel coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    
                    # Normalize to [0, 1]
                    x1_norm = x1 / image_width
                    y1_norm = y1 / image_height
                    x2_norm = x2 / image_width
                    y2_norm = y2 / image_height
                    
                    bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm, confidence])
        
        # Pad to max_detections_per_camera if needed
        while len(bboxes) < self.max_detections_per_camera:
            bboxes.append([0.0, 0.0, 0.0, 0.0, 0.0])  # No detection
        
        return torch.tensor(bboxes, dtype=torch.float32)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Process observation to extract bounding boxes and concatenate to state.
        
        Args:
            observation: Dictionary containing 'observation.images' and 'observation.state'
            
        Returns:
            Modified observation with bounding boxes concatenated to state.
        """
        # Get images from observation
        images_dict = observation.get(OBS_IMAGES, {})
        if not images_dict:
            # No images found, return observation unchanged
            return observation
        
        # Determine which cameras to process
        if self.camera_names is None:
            # Process all cameras
            cameras_to_process = list(images_dict.keys())
        else:
            # Process only specified cameras
            cameras_to_process = [cam for cam in self.camera_names if cam in images_dict]
        
        if not cameras_to_process:
            # No valid cameras found
            return observation
        
        # Extract bounding boxes for each camera
        # Note: This processor runs BEFORE AddBatchDimensionProcessorStep,
        # so images should not have batch dimension yet
        all_bboxes = []
        for camera_name in cameras_to_process:
            image = images_dict[camera_name]
            
            # Handle image format (should be 3D at this point, before batch dimension is added)
            if image.dim() == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    single_image = image
                    h, w = image.shape[1], image.shape[2]
                else:  # (H, W, C)
                    single_image = image
                    h, w = image.shape[0], image.shape[1]
            elif image.dim() == 4:
                # If batch dimension is already present (shouldn't happen, but handle gracefully)
                # Process first image in batch
                if image.shape[1] == 3:  # (B, C, H, W)
                    single_image = image[0]
                    h, w = image.shape[2], image.shape[3]
                else:  # (B, H, W, C)
                    single_image = image[0]
                    h, w = image.shape[1], image.shape[2]
            else:
                raise ValueError(
                    f"Unexpected image shape: {image.shape}. "
                    f"Expected 3D (C, H, W) or (H, W, C), got {image.dim()}D."
                )
            
            # Extract bounding boxes
            bboxes = self._extract_bboxes_from_image(single_image, h, w)
            all_bboxes.append(bboxes)
        
        # Concatenate all bounding boxes: (num_cameras * max_detections_per_camera, 5)
        if all_bboxes:
            concatenated_bboxes = torch.cat(all_bboxes, dim=0)  # Shape: (total_detections, 5)
            # Flatten to 1D: (total_detections * 5,)
            flattened_bboxes = concatenated_bboxes.flatten()
        else:
            # No cameras processed, create zero vector
            total_bbox_dim = len(cameras_to_process) * self.bbox_dim_per_camera
            flattened_bboxes = torch.zeros(total_bbox_dim, dtype=torch.float32)
        
        # Get existing state
        state = observation.get(OBS_STATE)
        if state is None:
            # No existing state, use only bounding boxes
            new_state = flattened_bboxes
        else:
            # Concatenate bounding boxes to existing state
            if isinstance(state, torch.Tensor):
                state_tensor = state
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Ensure both are 1D
            if state_tensor.dim() > 1:
                state_tensor = state_tensor.flatten()
            if flattened_bboxes.dim() > 1:
                flattened_bboxes = flattened_bboxes.flatten()
            
            # Concatenate: [original_state, bbox_cam1, bbox_cam2, ...]
            new_state = torch.cat([state_tensor, flattened_bboxes], dim=0)
        
        # Update observation
        observation = observation.copy()
        observation[OBS_STATE] = new_state
        
        return observation

