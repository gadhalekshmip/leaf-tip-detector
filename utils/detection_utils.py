# utils/detection_utils.py

import numpy as np
from typing import List, Dict, Tuple
import streamlit as st

def remove_duplicates(detections: List[Dict], 
                     distance_threshold: float = 8.0) -> List[Dict]:
    """
    Remove duplicate detections based on Euclidean distance threshold
    
    Args:
        detections: List of detection dictionaries with 'x', 'y', 'conf' keys
        distance_threshold: Maximum distance to consider as duplicate
        
    Returns:
        List of unique detections sorted by confidence
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
    unique_detections = []
    
    for detection in sorted_detections:
        x, y = detection['x'], detection['y']
        
        # Check if this detection is too close to any existing unique detection
        is_duplicate = False
        for existing in unique_detections:
            distance = calculate_distance((x, y), (existing['x'], existing['y']))
            if distance <= distance_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_detections.append(detection)
    
    return unique_detections

def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def add_manual_point(detections: List[Dict], 
                    x: float, y: float, 
                    conf: float = 1.0) -> List[Dict]:
    """
    Add a manual detection point
    
    Args:
        detections: Current list of detections
        x, y: Coordinates of new point
        conf: Confidence score for manual point
        
    Returns:
        Updated detections list
    """
    new_point = {
        'x': float(x),
        'y': float(y),
        'conf': float(conf),
        'method': 'manual',
        'manual': True
    }
    
    return detections + [new_point]

def remove_nearest_point(detections: List[Dict], 
                        x: float, y: float,
                        max_distance: float = 20.0) -> Tuple[List[Dict], bool]:
    """
    Remove the nearest detection point to given coordinates
    
    Args:
        detections: Current list of detections
        x, y: Click coordinates
        max_distance: Maximum distance to consider for removal
        
    Returns:
        (updated_detections, was_removed)
    """
    if not detections:
        return detections, False
    
    # Find nearest point
    min_distance = float('inf')
    nearest_index = -1
    
    for i, detection in enumerate(detections):
        distance = calculate_distance((x, y), (detection['x'], detection['y']))
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    # Remove if within threshold
    if min_distance <= max_distance and nearest_index >= 0:
        updated_detections = detections[:nearest_index] + detections[nearest_index+1:]
        return updated_detections, True
    
    return detections, False

def remove_point_by_index(detections: List[Dict], index: int) -> List[Dict]:
    """Remove detection point by index"""
    if 0 <= index < len(detections):
        return detections[:index] + detections[index+1:]
    return detections

def validate_detections(detections: List[Dict]) -> List[Dict]:
    """
    Validate and clean detection data
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Cleaned and validated detections
    """
    valid_detections = []
    
    for detection in detections:
        # Check required fields
        if not all(key in detection for key in ['x', 'y', 'conf']):
            continue
            
        # Validate coordinates and confidence
        try:
            x = float(detection['x'])
            y = float(detection['y'])
            conf = float(detection['conf'])
            
            # Check if values are reasonable
            if x < 0 or y < 0 or conf < 0 or conf > 1:
                continue
                
            # Create clean detection
            clean_detection = {
                'x': x,
                'y': y,
                'conf': conf,
                'method': detection.get('method', 'unknown'),
                'manual': detection.get('manual', False)
            }
            
            # Preserve additional fields if they exist
            for key in ['cell', 'cell_coords', 'bbox', 'roi']:
                if key in detection:
                    clean_detection[key] = detection[key]
            
            valid_detections.append(clean_detection)
            
        except (ValueError, TypeError):
            continue
    
    return valid_detections

def merge_detections(detection_lists: List[List[Dict]], 
                    remove_duplicates_flag: bool = True,
                    distance_threshold: float = 8.0) -> List[Dict]:
    """
    Merge multiple detection lists
    
    Args:
        detection_lists: List of detection lists to merge
        remove_duplicates_flag: Whether to remove duplicates after merging
        distance_threshold: Distance threshold for duplicate removal
        
    Returns:
        Merged detection list
    """
    merged = []
    for detection_list in detection_lists:
        merged.extend(validate_detections(detection_list))
    
    if remove_duplicates_flag:
        merged = remove_duplicates(merged, distance_threshold)
    
    return merged

def filter_detections_by_confidence(detections: List[Dict], 
                                   min_confidence: float) -> List[Dict]:
    """Filter detections by minimum confidence threshold"""
    return [d for d in detections if d.get('conf', 0) >= min_confidence]

def filter_detections_by_region(detections: List[Dict],
                               region: Tuple[int, int, int, int]) -> List[Dict]:
    """
    Filter detections by region (x1, y1, x2, y2)
    
    Args:
        detections: List of detections
        region: (x1, y1, x2, y2) bounding box
        
    Returns:
        Detections within the region
    """
    x1, y1, x2, y2 = region
    
    filtered = []
    for detection in detections:
        x, y = detection['x'], detection['y']
        if x1 <= x <= x2 and y1 <= y <= y2:
            filtered.append(detection)
    
    return filtered

def get_detection_statistics(detections: List[Dict]) -> Dict:
    """Get statistics about detections"""
    if not detections:
        return {
            'total': 0,
            'manual': 0,
            'automatic': 0,
            'avg_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0
        }
    
    manual_count = sum(1 for d in detections if d.get('manual', False))
    confidences = [d.get('conf', 0) for d in detections]
    
    return {
        'total': len(detections),
        'manual': manual_count,
        'automatic': len(detections) - manual_count,
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0
    }