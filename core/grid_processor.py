# core/grid_processor.py

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import streamlit as st

from core.yolo_processor import YOLOProcessor

class GridProcessor:
    """Grid-based image processing for leaf tip detection"""
    
    def __init__(self, yolo_processor: YOLOProcessor):
        self.yolo_processor = yolo_processor
        
    def process_image_with_grid(self, 
                               image_path: str,
                               grid_size: int = 3,
                               conf_thresh: float = 0.25) -> Tuple[List[Dict], List[Dict]]:
        """
        Process image with grid approach
        Returns: (raw_detections, final_detections)
        """
        
        if not self.yolo_processor.is_loaded():
            raise ValueError("YOLO model not loaded")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load original image
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                # Convert RGBA to RGB
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, (0, 0), img.convert('RGBA'))
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {str(e)}")
        
        img_width, img_height = img.size
        
        # Calculate grid cell dimensions (no overlap)
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size
        
        # Store all raw detections
        all_raw_detections = []
        
        # Process each grid cell
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate grid cell boundaries
                x_start = int(col * cell_width)
                y_start = int(row * cell_height)
                x_end = int(min(img_width, (col + 1) * cell_width))
                y_end = int(min(img_height, (row + 1) * cell_height))
                
                # Create grid cell image
                cell_img = img.crop((x_start, y_start, x_end, y_end))
                
                # Run detection on cell
                try:
                    cell_detections = self.yolo_processor.run_inference_on_crop(
                        cell_img, conf_thresh
                    )
                    
                    # Convert cell coordinates to original image coordinates
                    for detection in cell_detections:
                        # Convert to original coordinates
                        orig_x = x_start + detection['x']
                        orig_y = y_start + detection['y']
                        
                        # Add to raw detections with cell info
                        all_raw_detections.append({
                            'x': orig_x,
                            'y': orig_y,
                            'conf': detection['conf'],
                            'method': f'grid_{grid_size}x{grid_size}',
                            'cell': (row, col),
                            'cell_coords': (x_start, y_start, x_end, y_end)
                        })
                        
                except Exception as e:
                    st.warning(f"Error processing grid cell ({row},{col}): {str(e)}")
                    continue
        
        # Remove duplicates using distance threshold
        final_detections = self.remove_duplicates(
            all_raw_detections, 
            distance_threshold=8  # Based on your testing results
        )
        
        return all_raw_detections, final_detections
    
    def remove_duplicates(self, 
                         detections: List[Dict], 
                         distance_threshold: float = 8.0) -> List[Dict]:
        """Remove duplicate detections based on distance threshold"""
        
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        unique_detections = []
        
        for detection in sorted_detections:
            x, y = detection['x'], detection['y']
            
            # Check if this detection is too close to any existing unique detection
            is_duplicate = False
            for existing in unique_detections:
                distance = np.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
                if distance <= distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def get_grid_info(self, image_path: str, grid_size: int) -> Dict:
        """Get grid information for visualization"""
        
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            cell_width = img_width / grid_size
            cell_height = img_height / grid_size
            
            # Generate grid lines
            grid_lines = {
                'vertical': [],
                'horizontal': [],
                'cells': []
            }
            
            # Vertical lines
            for i in range(1, grid_size):
                x = i * cell_width
                grid_lines['vertical'].append(x)
            
            # Horizontal lines  
            for i in range(1, grid_size):
                y = i * cell_height
                grid_lines['horizontal'].append(y)
            
            # Cell information
            for row in range(grid_size):
                for col in range(grid_size):
                    x_start = col * cell_width
                    y_start = row * cell_height
                    x_end = (col + 1) * cell_width
                    y_end = (row + 1) * cell_height
                    
                    grid_lines['cells'].append({
                        'row': row,
                        'col': col,
                        'bounds': (x_start, y_start, x_end, y_end),
                        'center': (x_start + cell_width/2, y_start + cell_height/2)
                    })
            
            return {
                'image_size': (img_width, img_height),
                'grid_size': grid_size,
                'cell_size': (cell_width, cell_height),
                'grid_lines': grid_lines
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get grid info: {str(e)}")

def create_grid_processor(yolo_processor: YOLOProcessor) -> GridProcessor:
    """Create grid processor with YOLO model"""
    return GridProcessor(yolo_processor)