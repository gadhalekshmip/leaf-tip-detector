# core/frcnn_grid_processor.py - CPU OPTIMIZED for Streamlit Cloud

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import streamlit as st
import torch
import torchvision
import gc

from core.frcnn_processor import FRCNNProcessor

class FRCNNGridProcessor:
    """FRCNN Grid-based image processing optimized for CPU-only environments (Streamlit Cloud)"""
    
    def __init__(self, frcnn_processor: FRCNNProcessor):
        self.frcnn_processor = frcnn_processor
        
        # DEBUG: Check device info
        print(f"🔍 FRCNN Device: {self.frcnn_processor.device}")
        if hasattr(self.frcnn_processor, 'model') and self.frcnn_processor.model:
            print(f"🔍 Model on device: {next(self.frcnn_processor.model.parameters()).device}")
        
        # REMOVE self.transform creation - we'll create fresh ones each time!
        print("✅ FRCNN Grid using fresh transforms (original 20-sec pattern)")
    def process_image_with_grid(self, 
                               image_path: str,
                               grid_size: int = 3,
                               conf_thresh: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        Process image with FRCNN grid approach - CPU OPTIMIZED
        Returns: (raw_detections, final_detections)
        """
        
        if not self.frcnn_processor.is_loaded():
            raise ValueError("FRCNN model not loaded")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # CPU MEMORY CLEANUP: Force garbage collection before starting
        gc.collect()
        
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
        
        # Step 1: Divide into grid
        grid_cells = self.divide_image_into_grid(img, grid_size)
        
        # Step 2: Run predictions on each cell with aggressive memory management
        grid_predictions = []
        all_raw_detections = []
        
        for i, cell_data in enumerate(grid_cells):
            try:
                cell_detections = self.predict_on_grid_cell(cell_data, conf_thresh)
                grid_predictions.append((cell_data, cell_detections))
                
                # Convert to raw detections format for consistency
                for detection in cell_detections:
                    # Convert to original coordinates
                    orig_x = detection['x'] / cell_data['scale'] + cell_data['x_start']
                    orig_y = detection['y'] / cell_data['scale'] + cell_data['y_start']
                    
                    all_raw_detections.append({
                        'x': orig_x,
                        'y': orig_y,
                        'conf': detection['conf'],
                        'method': f'frcnn_grid_{grid_size}x{grid_size}',
                        'cell': (cell_data['row'], cell_data['col']),
                        'cell_coords': (cell_data['x_start'], cell_data['y_start'], 
                                      cell_data['x_start'] + cell_data['width'], 
                                      cell_data['y_start'] + cell_data['height'])
                    })
                
                # CPU OPTIMIZATION: Force memory cleanup every 3 cells
                if (i + 1) % 3 == 0:
                    gc.collect()
                    
            except Exception as e:
                st.warning(f"Error processing grid cell ({cell_data['row']},{cell_data['col']}): {str(e)}")
                continue
        
        # CPU OPTIMIZATION: Final cleanup before stitching
        gc.collect()
        
        # Step 3: Stitch predictions back and apply final NMS
        final_detections = self.stitch_predictions_back(grid_predictions, img.size, grid_size)
        
        return all_raw_detections, final_detections
    
    def divide_image_into_grid(self, image, grid_size=3):
        """
        Divide image into non-overlapping grid cells
        Returns list of cell data with images and positions
        """
        img_width, img_height = image.size
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size

        grid_cells = []

        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate cell boundaries
                x_start = int(col * cell_width)
                y_start = int(row * cell_height)
                x_end = int((col + 1) * cell_width)
                y_end = int((row + 1) * cell_height)

                # Crop cell from image
                cell_img = image.crop((x_start, y_start, x_end, y_end))

                # Resize if needed (same logic as training)
                target_size = self.frcnn_processor.image_size
                if target_size:
                    orig_width, orig_height = cell_img.size
                    scale = min(target_size / orig_width, target_size / orig_height)
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                    cell_img = cell_img.resize((new_width, new_height), Image.LANCZOS)
                else:
                    scale = 1.0

                grid_cells.append({
                    'image': cell_img,
                    'row': row,
                    'col': col,
                    'x_start': x_start,
                    'y_start': y_start,
                    'width': cell_width,
                    'height': cell_height,
                    'scale': scale
                })

        return grid_cells

    def predict_on_grid_cell(self, cell_data, conf_thresh):
        """Create fresh transform each time like original 20-sec code"""
        import time
        
        start_time = time.time()
        
        # CREATE FRESH TRANSFORM EACH TIME (original pattern)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        cell_tensor = transform(cell_data['image']).unsqueeze(0).to(self.frcnn_processor.device)
        tensor_time = time.time()
        
        self.frcnn_processor.model.eval()
        with torch.no_grad():
            raw_prediction = self.frcnn_processor.model(cell_tensor)[0]
        
        inference_time = time.time()
        
        # Enhanced post-processing (same as original)
        processed_prediction = self.enhanced_post_process_detections(
            [raw_prediction],
            box_size=self.frcnn_processor.box_size,
            conf_threshold=conf_thresh,
            nms_threshold=self.frcnn_processor.nms_threshold
        )[0]
        
        postprocess_time = time.time()
        
        # DEBUG: Print timing
        print(f"Cell ({cell_data['row']},{cell_data['col']}): "
            f"Tensor: {tensor_time-start_time:.2f}s, "
            f"Inference: {inference_time-tensor_time:.2f}s, "
            f"PostProcess: {postprocess_time-inference_time:.2f}s")
        
        # Convert to dict format (same as before)
        detections = []
        if len(processed_prediction['boxes']) > 0:
            boxes = processed_prediction['boxes'].cpu().numpy()
            scores = processed_prediction['scores'].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                detections.append({
                    'x': center_x,
                    'y': center_y,
                    'conf': score,
                    'bbox': box.tolist()
                })
        
        total_time = time.time()
        print(f"Total cell time: {total_time-start_time:.2f}s")
        
        return detections
    def enhanced_post_process_detections(self, detections, box_size=10, conf_threshold=0.3, nms_threshold=0.3):
        """Enhanced post-processing from colab code - EXACT COPY"""
        for i in range(len(detections)):
            if len(detections[i]['boxes']) == 0:
                continue

            boxes = detections[i]['boxes']
            scores = detections[i]['scores']
            labels = detections[i]['labels']

            # Step 1: Apply confidence threshold
            keep_conf = scores >= conf_threshold
            boxes = boxes[keep_conf]
            scores = scores[keep_conf]
            labels = labels[keep_conf]

            if len(boxes) == 0:
                detections[i]['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                detections[i]['scores'] = torch.zeros(0, dtype=torch.float32)
                detections[i]['labels'] = torch.zeros(0, dtype=torch.int64)
                continue

            # Step 2: Apply standard NMS first
            keep_nms = torchvision.ops.nms(boxes, scores, nms_threshold)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
            labels = labels[keep_nms]

            # Step 3: Extract centers and create standardized boxes
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

            # Step 4: Apply keypoint-based NMS
            keep_indices = self.enhanced_keypoint_nms(
                centers_x, centers_y, scores,
                distance_threshold=box_size * 1.5,
                max_detections=200
            )

            centers_x = centers_x[keep_indices]
            centers_y = centers_y[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            # Step 5: Create new standardized boxes around filtered centers
            half_size = box_size / 2
            new_boxes = torch.zeros((len(centers_x), 4), dtype=torch.float32)
            new_boxes[:, 0] = centers_x - half_size  # xmin
            new_boxes[:, 1] = centers_y - half_size  # ymin
            new_boxes[:, 2] = centers_x + half_size  # xmax
            new_boxes[:, 3] = centers_y + half_size  # ymax

            # Update detection with filtered results
            detections[i]['boxes'] = new_boxes
            detections[i]['scores'] = scores
            detections[i]['labels'] = labels

        return detections

    def enhanced_keypoint_nms(self, centers_x, centers_y, scores, distance_threshold=15, max_detections=200):
        """Keypoint NMS from colab code - EXACT COPY"""
        # Move everything to CPU for numpy operations
        centers_x = centers_x.cpu().numpy()
        centers_y = centers_y.cpu().numpy()
        scores = scores.cpu().numpy()

        # Get sorting indices by score (highest first)
        order = scores.argsort()[::-1]

        # Limit to max_detections if we have too many
        if len(order) > max_detections:
            order = order[:max_detections]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if len(keep) >= max_detections:
                break

            # Calculate distances between current point and all others
            distances = np.sqrt(
                (centers_x[i] - centers_x[order[1:]])**2 +
                (centers_y[i] - centers_y[order[1:]])**2
            )

            # Keep only points that are further than threshold
            inds = np.where(distances > distance_threshold)[0]
            order = order[inds + 1]  # +1 because we removed the first element

        return torch.tensor(keep, dtype=torch.long)

    def stitch_predictions_back(self, grid_predictions, original_img_size, grid_size):
        """
        Stitch grid cell predictions back to original image coordinates
        """
        all_predictions = []

        for cell_data, predictions in grid_predictions:
            for pred in predictions:
                # Unscale and shift back to original coordinates
                orig_x = pred['x'] / cell_data['scale'] + cell_data['x_start']
                orig_y = pred['y'] / cell_data['scale'] + cell_data['y_start']
                
                # Also convert bbox if available
                orig_bbox = None
                if 'bbox' in pred:
                    bbox = pred['bbox']
                    orig_bbox = [
                        bbox[0] / cell_data['scale'] + cell_data['x_start'],
                        bbox[1] / cell_data['scale'] + cell_data['y_start'],
                        bbox[2] / cell_data['scale'] + cell_data['x_start'],
                        bbox[3] / cell_data['scale'] + cell_data['y_start']
                    ]

                all_predictions.append({
                    'x': orig_x,
                    'y': orig_y,
                    'conf': pred['conf'],
                    'method': f'frcnn_grid_{grid_size}x{grid_size}',
                    'cell': (cell_data['row'], cell_data['col']),
                    'bbox': orig_bbox
                })

        # Apply final NMS to remove duplicates across grid boundaries
        if len(all_predictions) > 0:
            pred_centers_x = torch.tensor([p['x'] for p in all_predictions])
            pred_centers_y = torch.tensor([p['y'] for p in all_predictions])
            pred_scores = torch.tensor([p['conf'] for p in all_predictions])

            # Enhanced keypoint NMS
            keep_indices = self.enhanced_keypoint_nms(
                pred_centers_x, pred_centers_y, pred_scores,
                distance_threshold=self.frcnn_processor.box_size * 1.5,
                max_detections=200
            )

            final_predictions = [all_predictions[i] for i in keep_indices]
            
            # CPU OPTIMIZATION: Clean up temporary tensors
            del pred_centers_x, pred_centers_y, pred_scores
            gc.collect()
        else:
            final_predictions = []

        return final_predictions


def create_frcnn_grid_processor(model_config: Dict) -> FRCNNGridProcessor:
    """Create FRCNN grid processor from model config - CPU OPTIMIZED"""
    
    # First create the base FRCNN processor
    from core.frcnn_processor import create_frcnn_processor
    
    frcnn_processor = create_frcnn_processor(model_config)
    
    if not frcnn_processor.is_loaded():
        raise RuntimeError(f"Failed to load FRCNN model from {model_config.get('path')}")
    
    # Then wrap it with grid processor
    grid_processor = FRCNNGridProcessor(frcnn_processor)
    
    return grid_processor