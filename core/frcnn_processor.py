# core/frcnn_processor.py

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np
from PIL import Image
import streamlit as st
from typing import List, Dict, Optional

class FRCNNProcessor:
    """Faster R-CNN processor for leaf tip detection"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ADD THESE CONFIG ATTRIBUTES
        self.default_conf_thresh = 0.5
        self.box_size = 10
        self.image_size = 1536
        self.nms_threshold = 0.5
        self.distance_threshold = 15
        self.max_detections = 200
    def load_model(self, model_path: str) -> bool:
        """Load Faster R-CNN model from path"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return False
            
            # Create model architecture
            self.model = self.create_frcnn_model()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_path
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load FRCNN model: {str(e)}")
            return False
    
    def create_frcnn_model(self, num_classes: int = 2, keypoint_size: int = 10):
        """Create Faster R-CNN model architecture"""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Configure anchor generator
        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        model.rpn.anchor_generator = anchor_generator
        
        # Update RPN head
        num_anchors = len(anchor_sizes[0]) * len(aspect_ratios[0])
        model.rpn.head.cls_logits = torch.nn.Conv2d(256, num_anchors, kernel_size=1)
        model.rpn.head.bbox_pred = torch.nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        
        return model
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    # Replace the post_process_detections function in core/frcnn_processor.py

     # Replace the post_process_detections function in core/frcnn_processor.py

    def post_process_detections(self, 
                            predictions: Dict, 
                            conf_thresh: float = None,
                            box_size: int = None,
                            nms_threshold: float = None,
                            distance_threshold: float = None,
                            max_detections: int = None) -> List[Dict]:
        """Enhanced post-processing matching Colab code exactly"""
        
        # Use processor config values as defaults
        conf_thresh = conf_thresh or getattr(self, 'default_conf_thresh', 0.5)
        box_size = box_size or getattr(self, 'box_size', 10)
        nms_threshold = nms_threshold or getattr(self, 'nms_threshold', 0.5)
        distance_threshold = distance_threshold or getattr(self, 'distance_threshold', 15)
        max_detections = max_detections or getattr(self, 'max_detections', 200)
        
        detections = []
        
        if len(predictions['boxes']) == 0:
            return detections
        
        boxes = predictions['boxes']
        scores = predictions['scores'] 
        labels = predictions['labels']
        
        # Step 1: Apply confidence threshold
        keep_conf = scores >= conf_thresh
        boxes = boxes[keep_conf]
        scores = scores[keep_conf]
        labels = labels[keep_conf]
        
        if len(boxes) == 0:
            return detections
        
        # Step 2: Apply standard NMS
        keep_nms = torchvision.ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]
        
        # Step 3: Extract centers for keypoint NMS (THIS WAS MISSING!)
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        
        # Step 4: Apply keypoint NMS (CRITICAL - this removes false positives)
        keep_indices = self.enhanced_keypoint_nms(
            centers_x, centers_y, scores, 
            distance_threshold=distance_threshold, 
            max_detections=max_detections
        )
        
        final_centers_x = centers_x[keep_indices]
        final_centers_y = centers_y[keep_indices] 
        final_scores = scores[keep_indices]
        
        # Step 5: Create standardized boxes (same as Colab)
        half_size = box_size / 2
        
        for i in range(len(final_centers_x)):
            center_x = float(final_centers_x[i])
            center_y = float(final_centers_y[i])
            conf = float(final_scores[i])
            
            # Create standardized bounding box
            new_box = [
                center_x - half_size,  # x1
                center_y - half_size,  # y1
                center_x + half_size,  # x2
                center_y + half_size   # y2
            ]
            
            detections.append({
                'x': center_x,
                'y': center_y,
                'conf': conf,
                'method': 'frcnn',
                'bbox': new_box
            })
        
        return detections

    def enhanced_keypoint_nms(self, centers_x, centers_y, scores, distance_threshold=15, max_detections=200):
        """Keypoint NMS implementation - EXACT copy from Colab"""
        
        # Convert to numpy for processing
        centers_x = centers_x.cpu().numpy()
        centers_y = centers_y.cpu().numpy() 
        scores = scores.cpu().numpy()
        
        # Sort by confidence (highest first)
        order = scores.argsort()[::-1]
        if len(order) > max_detections:
            order = order[:max_detections]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if len(keep) >= max_detections:
                break
            
            # Calculate distances to remaining detections
            distances = np.sqrt(
                (centers_x[i] - centers_x[order[1:]])**2 + 
                (centers_y[i] - centers_y[order[1:]])**2
            )
            
            # Keep only detections beyond distance threshold
            inds = np.where(distances > distance_threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long)

    # Also update the run_inference function to use EXACT Colab defaults:

    def run_inference(self, 
                    image_path: str, 
                    conf_thresh: float = None,  # EXACT match with Colab
                    box_size: int = None) -> List[Dict]:
        """Run FRCNN inference on image with enhanced post-processing"""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded")
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        conf_thresh = conf_thresh or self.default_conf_thresh
        box_size = box_size or self.box_size
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Resize to match training size (1536) - IMPORTANT!
            original_size = image.size
            target_size = self.image_size
            
            # Calculate scale factor
            scale = min(target_size / original_size[0], target_size / original_size[1])
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            
            # Resize image
            image_resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to tensor
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            image_tensor = transform(image_resized).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Post-process detections with enhanced filtering
            detections = self.post_process_detections(
                predictions, conf_thresh, box_size
            )
            
            # Scale coordinates back to original image size
            for detection in detections:
                detection['x'] = detection['x'] / scale
                detection['y'] = detection['y'] / scale
                # Scale bounding box too
                detection['bbox'] = [coord / scale for coord in detection['bbox']]
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"FRCNN inference failed: {str(e)}")    
    def run_inference_on_crop(self, cell_img, conf_thresh=None):
        """
        Run FRCNN inference on PIL image crop (for grid processing)
        Args:
            cell_img: PIL Image (already cropped and resized)
            conf_thresh: Confidence threshold
        Returns:
            List of detections in relative coordinates to the crop
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        conf_thresh = conf_thresh or self.default_conf_thresh
        
        try:
            # Convert PIL image to tensor
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            image_tensor = transform(cell_img).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Post-process detections
            detections = self.post_process_detections(
                predictions, conf_thresh, self.box_size
            )
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"FRCNN inference on crop failed: {str(e)}")
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self.is_loaded():
            return {"loaded": False}
        
        return {
            "loaded": True,
            "path": self.model_path,
            "type": "frcnn",
            "device": str(self.device),
            "model_name": os.path.basename(self.model_path) if self.model_path else "Unknown"
        }

def create_frcnn_processor(model_config: Dict) -> FRCNNProcessor:
    """Create and load FRCNN processor from config with Colab parameters"""
    processor = FRCNNProcessor()
    
    # SET CONFIG PARAMETERS BEFORE LOADING MODEL
    processor.default_conf_thresh = model_config.get("conf_threshold", 0.5)
    processor.box_size = model_config.get("box_size", 10) 
    processor.image_size = model_config.get("image_size", 1536)
    processor.nms_threshold = model_config.get("nms_threshold", 0.5)
    processor.distance_threshold = model_config.get("distance_threshold", 15)
    processor.max_detections = model_config.get("max_detections", 200)
    
    model_path = model_config.get("path")
    
    if model_path and os.path.exists(model_path):
        success = processor.load_model(model_path)
        if success:
            return processor
    
    return processor