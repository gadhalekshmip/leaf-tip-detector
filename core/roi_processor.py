# core/roi_processor.py

import os
import tempfile
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import streamlit as st

from core.yolo_processor import YOLOProcessor

class ROIProcessor:
    """ROI-based image processing for leaf tip detection"""
    
    def __init__(self, yolo_processor: YOLOProcessor):
        self.yolo_processor = yolo_processor
        
    def run_inference_on_roi(self, 
                            roi_image: Image.Image,
                            conf_thresh: float = 0.25) -> List[Dict]:
        """
        Run inference on ROI image
        Returns detections in ROI coordinate space
        
        Args:
            roi_image: Cropped ROI image
            conf_thresh: Confidence threshold
            
        Returns:
            List of detections in ROI coordinate space
        """
        
        if not self.yolo_processor.is_loaded():
            raise ValueError("YOLO model not loaded")
        
        try:
            # Save ROI image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                roi_image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Run YOLO inference on ROI
            detections = self.yolo_processor.run_inference(temp_path, conf_thresh)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Add ROI method tag
            for detection in detections:
                detection['method'] = 'roi'
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"ROI inference failed: {str(e)}")
    
    def process_roi_on_full_image(self, 
                                 image_path: str,
                                 roi_coords: Tuple[int, int, int, int],
                                 conf_thresh: float = 0.25) -> List[Dict]:
        """
        Complete ROI workflow: Crop → Detect → Stitch back to full image
        
        Args:
            image_path: Path to full image
            roi_coords: (x1, y1, x2, y2) ROI coordinates in full image
            conf_thresh: Confidence threshold
            
        Returns:
            List of detections in original full image coordinate space
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Step 1: Load full image
            full_image = Image.open(image_path)
            
            # Step 2: Extract ROI (crop the box)
            x1, y1, x2, y2 = roi_coords
            roi_image = full_image.crop((x1, y1, x2, y2))
            
            # Step 3: Run detection on cropped ROI
            roi_detections = self.run_inference_on_roi(roi_image, conf_thresh)
            
            # Step 4: Stitch back - convert ROI coordinates to full image coordinates
            full_image_detections = []
            for detection in roi_detections:
                # Convert from ROI space to full image space
                orig_x = x1 + detection['x']  # Add ROI offset
                orig_y = y1 + detection['y']  # Add ROI offset
                
                full_image_detections.append({
                    'x': orig_x,
                    'y': orig_y,
                    'conf': detection['conf'],
                    'method': 'roi',
                    'roi_coords': roi_coords,
                    'roi_relative_x': detection['x'],  # Original ROI coordinates
                    'roi_relative_y': detection['y']   # Original ROI coordinates
                })
            
            return full_image_detections
            
        except Exception as e:
            raise RuntimeError(f"ROI processing failed: {str(e)}")
    
    def validate_roi_coordinates(self, 
                                roi_coords: Tuple[int, int, int, int],
                                image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Validate and correct ROI coordinates
        
        Args:
            roi_coords: (x1, y1, x2, y2) ROI coordinates
            image_size: (width, height) of the image
            
        Returns:
            Validated and corrected ROI coordinates
        """
        
        x1, y1, x2, y2 = roi_coords
        img_width, img_height = image_size
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Ensure x1 < x2 and y1 < y2 (proper rectangle)
        if x1 >= x2:
            x1, x2 = x2, x1
        if y1 >= y2:
            y1, y2 = y2, y1
        
        # Ensure minimum ROI size (at least 50x50 pixels for meaningful detection)
        min_size = 50
        if (x2 - x1) < min_size:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_size // 2)
            x2 = min(img_width, center_x + min_size // 2)
        
        if (y2 - y1) < min_size:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_size // 2)
            y2 = min(img_height, center_y + min_size // 2)
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def get_roi_info(self, 
                    image_path: str,
                    roi_coords: Tuple[int, int, int, int]) -> Dict:
        """
        Get information about ROI region
        
        Args:
            image_path: Path to image
            roi_coords: (x1, y1, x2, y2) ROI coordinates
            
        Returns:
            Dictionary with ROI information
        """
        
        try:
            # Load image to get dimensions
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Validate ROI coordinates
            validated_coords = self.validate_roi_coordinates(roi_coords, (img_width, img_height))
            x1, y1, x2, y2 = validated_coords
            
            # Calculate ROI properties
            roi_width = x2 - x1
            roi_height = y2 - y1
            roi_area = roi_width * roi_height
            image_area = img_width * img_height
            coverage_percentage = (roi_area / image_area) * 100
            
            return {
                'original_coords': roi_coords,
                'validated_coords': validated_coords,
                'roi_size': (roi_width, roi_height),
                'roi_area': roi_area,
                'image_size': (img_width, img_height),
                'image_area': image_area,
                'coverage_percentage': coverage_percentage,
                'aspect_ratio': roi_width / roi_height if roi_height > 0 else 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get ROI info: {str(e)}")
    
    def extract_roi_image(self, 
                         image_path: str,
                         roi_coords: Tuple[int, int, int, int]) -> Image.Image:
        """
        Extract ROI from image as PIL Image
        
        Args:
            image_path: Path to source image
            roi_coords: (x1, y1, x2, y2) ROI coordinates
            
        Returns:
            PIL Image of the ROI
        """
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Validate coordinates
            validated_coords = self.validate_roi_coordinates(roi_coords, image.size)
            
            # Extract ROI
            roi_image = image.crop(validated_coords)
            
            return roi_image
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract ROI: {str(e)}")
    
    def batch_process_rois(self, 
                          image_paths: List[str],
                          roi_coords_list: List[Tuple[int, int, int, int]],
                          conf_thresh: float = 0.25) -> Dict[str, List[Dict]]:
        """
        Process multiple images with their respective ROIs
        
        Args:
            image_paths: List of image file paths
            roi_coords_list: List of ROI coordinates for each image
            conf_thresh: Confidence threshold
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        
        if len(image_paths) != len(roi_coords_list):
            raise ValueError("Number of images must match number of ROI coordinate sets")
        
        results = {}
        
        for image_path, roi_coords in zip(image_paths, roi_coords_list):
            try:
                detections = self.process_roi_on_full_image(
                    image_path, roi_coords, conf_thresh
                )
                results[image_path] = detections
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                results[image_path] = []
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if underlying YOLO processor is loaded"""
        return self.yolo_processor.is_loaded()
    
    def get_model_info(self) -> Dict:
        """Get information about the underlying model"""
        info = self.yolo_processor.get_model_info()
        info['processor_type'] = 'roi'
        return info

def create_roi_processor(yolo_processor: YOLOProcessor) -> ROIProcessor:
    """Create ROI processor with YOLO model"""
    return ROIProcessor(yolo_processor)