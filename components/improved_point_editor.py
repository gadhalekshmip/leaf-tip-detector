# components/improved_point_editor.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import List, Dict, Tuple, Optional

def show_interactive_point_editor(
    image: Image.Image, 
    detections: List[Dict], 
    session_prefix: str = "entire_"
) -> List[Dict]:
    """
    Interactive point editor using streamlit_image_coordinates
    
    Args:
        image: PIL Image to display
        detections: Current list of detections
        session_prefix: Session state prefix for isolation
        
    Returns:
        Updated detections list
    """
    
    # Get current editing mode
    mode_key = f"{session_prefix}editing_mode"
    current_mode = st.session_state.get(mode_key)
    
    # Only show interactive canvas if actively in edit mode
    if current_mode == 'add':
        return show_add_point_mode(image, detections, session_prefix)
    elif current_mode == 'remove':
        return show_remove_point_mode(image, detections, session_prefix)
    
    # Don't show any canvas when not editing - let the main display handle it
    return detections

def show_add_point_mode(image: Image.Image, detections: List[Dict], session_prefix: str) -> List[Dict]:
    """Add point mode with clickable image"""
    
    st.success("🖱️ **ADD MODE**: Click anywhere on the image to add a detection point")
    
    # Create image with existing detections
    display_image = draw_detections_on_image_simple(image, detections)
    
    # Get display size setting and match st.image() behavior exactly
    display_size = st.session_state.get('display_size', 'Fit to View')
    
    # Show clickable image with fixed width (Option 1 approach)
    coordinates = streamlit_image_coordinates(
        display_image,
        key=f"{session_prefix}add_coordinates",
        width=800  # Fixed width that works well
    )
    
    # Handle click with proper coordinate scaling
    if coordinates is not None:
        click_x, click_y = coordinates['x'], coordinates['y']
        
        # Scale coordinates back to original image size
        # The display_image might be scaled down to width=800, so we need to scale back up
        original_width, original_height = image.size
        display_width = 800  # The width we're using for display
        
        # Calculate scale factor
        scale_factor = original_width / display_width
        
        # Scale coordinates back to original image coordinates
        actual_x = click_x * scale_factor
        actual_y = click_y * scale_factor
        
        # Add new manual point with scaled coordinates
        new_point = {
            'x': float(actual_x),
            'y': float(actual_y),
            'conf': 1.0,
            'method': 'manual',
            'manual': True
        }
        
        detections.append(new_point)
        
        # Update session state
        st.session_state[f"{session_prefix}detections"] = detections
        st.session_state[f"{session_prefix}editing_mode"] = None
        
        st.success(f"✅ Added manual point at ({int(actual_x)}, {int(actual_y)})")
        st.rerun()
    
    # Cancel button
    if st.button("❌ Cancel Adding", key=f"{session_prefix}cancel_add"):
        st.session_state[f"{session_prefix}editing_mode"] = None
        st.rerun()
    
    return detections

def show_remove_point_mode(image: Image.Image, detections: List[Dict], session_prefix: str) -> List[Dict]:
    """Remove point mode with clickable image"""
    
    if not detections:
        st.warning("No points to remove")
        st.session_state[f"{session_prefix}editing_mode"] = None
        st.rerun()
        return detections
    
    st.warning("🖱️ **REMOVE MODE**: Click near any detection point (red/green circles) to remove it")
    st.info(f"📍 Current points: {len(detections)} (Tolerance: 20 pixels)")
    
    # Create image with existing detections - make them more visible for removal
    display_image = draw_detections_on_image_simple(image, detections, highlight_for_removal=True)
    
    # Get display size setting and match st.image() behavior exactly
    display_size = st.session_state.get('display_size', 'Fit to View')
    
    # Show clickable image with fixed width (Option 1 approach)
    coordinates = streamlit_image_coordinates(
        display_image,
        key=f"{session_prefix}remove_coordinates",
        width=800  # Fixed width that works well
    )
    
    # Handle click with proper coordinate scaling
    if coordinates is not None:
        click_x, click_y = coordinates['x'], coordinates['y']
        
        # Scale coordinates back to original image size
        # The display_image might be scaled down to width=800, so we need to scale back up
        original_width, original_height = image.size
        display_width = 800  # The width we're using for display
        
        # Calculate scale factor
        scale_factor = original_width / display_width
        
        # Scale coordinates back to original image coordinates
        actual_x = click_x * scale_factor
        actual_y = click_y * scale_factor
        
        # Find nearest point within threshold using scaled coordinates
        nearest_index, distance = find_nearest_point(actual_x, actual_y, detections, threshold=20)
        
        if nearest_index is not None:
            # Remove the point
            removed_point = detections.pop(nearest_index)
            
            # Update session state
            st.session_state[f"{session_prefix}detections"] = detections
            st.session_state[f"{session_prefix}editing_mode"] = None
            
            point_type = "Manual" if removed_point.get('manual', False) else "Auto"
            st.success(f"✅ Removed {point_type} point at ({int(removed_point['x'])}, {int(removed_point['y'])})")
            st.rerun()
        else:
            st.warning("⚠️ No point found near click position. Try clicking closer to a detection point.")
    
    # Cancel button
    if st.button("❌ Cancel Removing", key=f"{session_prefix}cancel_remove"):
        st.session_state[f"{session_prefix}editing_mode"] = None
        st.rerun()
    
    return detections

def draw_detections_on_image_simple(
    image: Image.Image, 
    detections: List[Dict],
    highlight_for_removal: bool = False
) -> Image.Image:
    """
    Draw detection points on image
    
    Args:
        image: PIL Image
        detections: List of detection dictionaries
        highlight_for_removal: Make points more visible for removal mode
        
    Returns:
        Image with detections drawn
    """
    
    if not detections:
        return image.copy()
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for detection in detections:
        x, y = int(detection['x']), int(detection['y'])
        is_manual = detection.get('manual', False)
        
        # Determine color and size
        if is_manual:
            color = 'lime' if highlight_for_removal else 'green'
            outline_color = 'white'
        else:
            color = 'red'
            outline_color = 'white'
        
        # Make points larger in removal mode
        point_size = 12 if highlight_for_removal else 8
        outline_width = 3 if highlight_for_removal else 2
        
        # Draw main circle
        draw.ellipse([
            x - point_size, y - point_size,
            x + point_size, y + point_size
        ], fill=color, outline=outline_color, width=outline_width)
        
        # Draw center dot for better visibility
        center_size = 3
        draw.ellipse([
            x - center_size, y - center_size,
            x + center_size, y + center_size
        ], fill='white')
        
        # Add confidence text for automatic detections (small text)
        if not is_manual and 'conf' in detection and not highlight_for_removal:
            conf_text = f"{detection['conf']:.2f}"
            draw.text((x + point_size + 2, y - point_size), conf_text, fill='yellow')
    
    return img_copy

def find_nearest_point(
    click_x: float, 
    click_y: float, 
    detections: List[Dict], 
    threshold: float = 20.0
) -> Tuple[Optional[int], float]:
    """
    Find the nearest detection point to click coordinates
    
    Args:
        click_x, click_y: Click coordinates
        detections: List of detections
        threshold: Maximum distance to consider
        
    Returns:
        (index_of_nearest_point, distance) or (None, inf) if none found
    """
    
    if not detections:
        return None, float('inf')
    
    min_distance = float('inf')
    nearest_index = None
    
    for i, detection in enumerate(detections):
        distance = np.sqrt((click_x - detection['x'])**2 + (click_y - detection['y'])**2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    # Return only if within threshold
    if min_distance <= threshold:
        return nearest_index, min_distance
    else:
        return None, min_distance

# Integration function for entire_image_tab.py
def integrate_point_editor_with_detections(
    image: Image.Image,
    detections: List[Dict],
    session_prefix: str = "entire_"
) -> List[Dict]:
    """
    Main integration function to be called from entire_image_tab.py
    
    Args:
        image: Current image
        detections: Current detections
        session_prefix: Session prefix for state isolation
        
    Returns:
        Updated detections list
    """
    
    return show_interactive_point_editor(image, detections, session_prefix)