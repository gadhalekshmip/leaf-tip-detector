# core/yolo_processor.py

import os
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

class YOLOProcessor:
    """YOLO model processor for leaf tip detection"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_type = None
        
    def load_model(self, model_path: str, model_type: str = "yolo") -> bool:
        """Load YOLO model from path"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return False
                
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.model_type = model_type
            return True
            
        except Exception as e:
            st.error(f"Failed to load YOLO model: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def run_inference(self, 
                     image_path: str, 
                     conf_thresh: float = 0.25, 
                     iou_thresh: float = 0.5,
                     max_det: int = 300) -> List[Dict]:
        """Run YOLO inference on image"""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded")
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Run inference
            results = self.model(
                image_path,
                conf=conf_thresh,
                iou=iou_thresh,
                max_det=max_det,
                agnostic_nms=True,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Handle keypoint format (for grid models)
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    
                    for instance_kpts in keypoints:
                        for kpt_idx, kpt in enumerate(instance_kpts):
                            x, y, conf = kpt
                            if conf >= conf_thresh:
                                detections.append({
                                    'x': float(x),
                                    'y': float(y),
                                    'conf': float(conf),
                                    'method': 'yolo_keypoint'
                                })
                
                # Handle bounding box format (for entire image models)
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        box_coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                        confidences = boxes.conf.cpu().numpy()
                        
                        for i, box in enumerate(box_coords):
                            if confidences[i] >= conf_thresh:
                                # Use center of bounding box as detection point
                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                
                                detections.append({
                                    'x': float(center_x),
                                    'y': float(center_y),
                                    'conf': float(confidences[i]),
                                    'method': 'yolo_bbox',
                                    'bbox': [float(x) for x in box]
                                })
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {str(e)}")
    
    def run_inference_on_crop(self, 
                             image_crop: Image.Image,
                             conf_thresh: float = 0.25) -> List[Dict]:
        """Run inference on PIL Image crop"""
        
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        try:
            # Convert PIL image to RGB if needed
            if image_crop.mode == 'RGBA':
                background = Image.new('RGB', image_crop.size, (255, 255, 255))
                background.paste(image_crop, (0, 0), image_crop.convert('RGBA'))
                image_crop = background
            elif image_crop.mode != 'RGB':
                image_crop = image_crop.convert('RGB')
            
            # Save temporarily for YOLO processing
            temp_path = "temp_crop.png"
            image_crop.save(temp_path)
            
            # Run inference
            detections = self.run_inference(temp_path, conf_thresh)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return detections
            
        except Exception as e:
            # Clean up on error
            if os.path.exists("temp_crop.png"):
                os.remove("temp_crop.png")
            raise RuntimeError(f"Crop inference failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self.is_loaded():
            return {"loaded": False}
        
        return {
            "loaded": True,
            "path": self.model_path,
            "type": self.model_type,
            "model_name": os.path.basename(self.model_path) if self.model_path else "Unknown"
        }

# Factory function for creating YOLO processors
def create_yolo_processor(model_config: Dict) -> YOLOProcessor:
    """Create and load YOLO processor from config"""
    processor = YOLOProcessor()
    
    model_path = model_config.get("path")
    model_type = model_config.get("type", "yolo")
    
    if model_path and os.path.exists(model_path):
        success = processor.load_model(model_path, model_type)
        if success:
            return processor
    
    return processor  # Return even if not loaded, for potential manual loading later