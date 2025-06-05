# components/roi_point_editor.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import List, Dict, Tuple, Optional
def show_roi_point_interface(
    image: Image.Image,
    session_prefix: str = "roi_"
):
    """ROI Point Interface - FIXED to work like annotation tab"""
    
    st.markdown("#### 🎯 ROI Controls")
    
    # Get ROI points and coordinates
    roi_points = st.session_state.get(f'{session_prefix}points', [])
    roi_coordinates = st.session_state.get(f'{session_prefix}coordinates')
    point_mode = st.session_state.get(f'{session_prefix}point_mode', False)
    
    # Limit to max 2 points
    if len(roi_points) > 2:
        roi_points = roi_points[:2]
        st.session_state[f'{session_prefix}points'] = roi_points
    
    # Show current status
    if roi_coordinates:
        # Rectangle is complete
        x1, y1, x2, y2 = roi_coordinates
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        st.success(f"✅ **ROI RECTANGLE COMPLETE**: {width}×{height} pixels")
        st.info(f"📍 **Corner 1**: ({int(roi_points[0]['x'])}, {int(roi_points[0]['y'])})")
        st.info(f"📍 **Corner 2**: ({int(roi_points[1]['x'])}, {int(roi_points[1]['y'])})")
        st.info("🚀 **Ready for detection!** Use 'Run ROI Detection' button.")
        
    elif len(roi_points) == 2:
        # Both points placed, waiting for "Done"
        st.warning("⏳ **BOTH CORNERS PLACED** - Ready to create rectangle")
        st.info(f"📍 **Corner 1**: ({int(roi_points[0]['x'])}, {int(roi_points[0]['y'])})")
        st.info(f"📍 **Corner 2**: ({int(roi_points[1]['x'])}, {int(roi_points[1]['y'])})")
        st.success("✅ **Click 'Done Adding ROI' to create rectangle**")
        
    elif len(roi_points) == 1:
        # First point placed, waiting for second
        st.warning(f"⏳ **NEED SECOND CORNER** (1/2 complete)")
        st.info(f"📍 **First corner**: ({int(roi_points[0]['x'])}, {int(roi_points[0]['y'])})")
        if point_mode:
            st.info("🖱️ **Click second corner on the image above**")
        else:
            st.error("🚨 **Click 'Add ROI Point' to continue**")
        
    elif len(roi_points) == 0:
        st.info("⚪ **No corners placed** - Click 'Add ROI Point' to begin")
    
    # Control buttons - FIXED with truly unique keys
    col1, col2, col3 = st.columns(3)
    
    # Get unique suffix for buttons based on state
    upload_counter = st.session_state.get(f'{session_prefix}upload_counter', 0)
    points_count = len(roi_points)
    unique_suffix = f"_{upload_counter}_{points_count}"
    
    with col1:
        if not point_mode and len(roi_points) < 2:
            # Start adding points
            if st.button("📍 Add ROI Point", key=f"{session_prefix}add_roi_point{unique_suffix}"):
                st.session_state[f'{session_prefix}point_mode'] = True
                if len(roi_points) == 0:
                    st.info("🖱️ **Step 1**: Click FIRST corner on the image above")
                st.rerun()
        elif point_mode and len(roi_points) < 2:
            # Currently adding points - show cancel
            if st.button("❌ Cancel Adding", key=f"{session_prefix}cancel_roi{unique_suffix}"):
                st.session_state[f'{session_prefix}point_mode'] = False
                st.rerun()
        elif len(roi_points) == 2 and not roi_coordinates:
            # Both points placed - show Done button (ONLY ONE!)
            if st.button("✅ Done Adding ROI", key=f"{session_prefix}done_roi{unique_suffix}", type="primary"):
                # Create rectangle from 2 points
                x1, y1 = roi_points[0]['x'], roi_points[0]['y']
                x2, y2 = roi_points[1]['x'], roi_points[1]['y']
                
                left = int(min(x1, x2))
                top = int(min(y1, y2))
                right = int(max(x1, x2))
                bottom = int(max(y1, y2))
                
                # FIXED: Create coordinates but KEEP the points for display
                st.session_state[f'{session_prefix}coordinates'] = (left, top, right, bottom)
                st.session_state[f'{session_prefix}point_mode'] = False
                # DON'T clear roi_points - keep them for display
                
                st.success(f"✅ **ROI RECTANGLE CREATED**: ({left}, {top}) to ({right}, {bottom})")
                st.balloons()
                st.rerun()
        else:
            st.success("✅ ROI Complete")
    
    with col2:
        if st.button("🗑️ Clear ROI", key=f"{session_prefix}clear_roi{unique_suffix}"):
            st.session_state[f'{session_prefix}coordinates'] = None
            st.session_state[f'{session_prefix}points'] = []
            st.session_state[f'{session_prefix}point_mode'] = False
            st.session_state[f'{session_prefix}upload_counter'] += 1
            st.success("ROI completely cleared")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Points", key=f"{session_prefix}reset_roi_points{unique_suffix}"):
            st.session_state[f'{session_prefix}points'] = []
            st.session_state[f'{session_prefix}coordinates'] = None
            st.session_state[f'{session_prefix}point_mode'] = False
            st.session_state[f'{session_prefix}upload_counter'] += 1
            st.success("Points reset - start over")
            st.rerun()
    
    # Show active mode status
    if point_mode:
        if len(roi_points) == 0:
            st.info("🖱️ **ADD MODE ACTIVE** - Click first corner on image")
        elif len(roi_points) == 1:
            st.info("🖱️ **ADD MODE ACTIVE** - Click second corner on image")
        else:
            st.info("🖱️ **BOTH POINTS PLACED** - Click 'Done Adding ROI' button")
def show_roi_point_editor(
    image: Image.Image,
    detections: List[Dict],
    session_prefix: str = "roi_"
):
    """
    Interactive ROI point editor - FIXED with dynamic keys
    """
    
    # Only show if in ROI point mode
    if not st.session_state.get(f'{session_prefix}point_mode', False):
        return
    
    roi_points = st.session_state.get(f'{session_prefix}points', [])
    
    if len(roi_points) == 0:
        st.success("🖱️ **ROI MODE**: Click FIRST corner on the image")
    elif len(roi_points) == 1:
        st.success("🖱️ **ROI MODE**: Click SECOND corner on the image (click farther from first point)")
    else:
        st.success("🖱️ **BOTH CORNERS PLACED**: Click 'Done Adding ROI' button below")
    
    # Create display image with ROI points and tip detections
    display_image = create_roi_point_display_image(image, detections, session_prefix)
    
    # FIXED: Use dynamic key that changes with point count to reset component
    roi_point_count = len(roi_points)
    dynamic_key = f"{session_prefix}roi_corner_coordinates_{roi_point_count}"
    
    coordinates = streamlit_image_coordinates(
        display_image,
        key=dynamic_key,  # FIXED: Dynamic key changes with point count
        width=800
    )
    
    # Handle click ONLY if we need more points
    if coordinates is not None and len(roi_points) < 2:
        handle_roi_point_click(coordinates, image, session_prefix)
def create_roi_point_display_image(
    image: Image.Image,
    detections: List[Dict],
    session_prefix: str = "roi_"
) -> Image.Image:
    """Create display image with ROI points and tip detections - FIXED to always show rectangle"""
    
    display_image = image.copy()
    draw = ImageDraw.Draw(display_image)
    
    # Get ROI data
    roi_coordinates = st.session_state.get(f'{session_prefix}coordinates')
    roi_points = st.session_state.get(f'{session_prefix}points', [])
    
    # PRIORITY 1: Draw existing final ROI rectangle if it exists (thick orange)
    if roi_coordinates:
        x1, y1, x2, y2 = roi_coordinates
        for i in range(4):  # Thicker for final rectangle
            draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline='orange', width=1)
        
        # Add corner markers for final rectangle
        corner_size = 15
        draw.rectangle([x1-corner_size, y1-corner_size, x1+corner_size, y1+corner_size], 
                      fill='orange', outline='white', width=2)
        draw.rectangle([x2-corner_size, y2-corner_size, x2+corner_size, y2+corner_size], 
                      fill='orange', outline='white', width=2)
    
    # PRIORITY 2: Draw preview rectangle if 2 points placed but no final rectangle yet
    elif len(roi_points) == 2:
        x1, y1 = int(roi_points[0]['x']), int(roi_points[0]['y'])
        x2, y2 = int(roi_points[1]['x']), int(roi_points[1]['y'])
        
        # Create proper rectangle coordinates
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2), max(y1, y2)
        
        # Draw preview rectangle (blue, thinner)
        for i in range(3):
            draw.rectangle([left+i, top+i, right-i, bottom-i], outline='blue', width=1)
        
        # Draw text indicating it's a preview
        draw.text((left+10, top-25), "PREVIEW - Click 'Done' to confirm", fill='blue', stroke_width=2, stroke_fill='white')
    
    # PRIORITY 3: Draw ROI corner points (blue squares) - ALWAYS visible during point mode
    for i, point in enumerate(roi_points):
        x, y = int(point['x']), int(point['y'])
        
        # Different colors for first and second point
        if i == 0:
            color = 'blue'
            text = "START"
        else:
            color = 'purple'
            text = "END"
        
        # Draw square point marker
        draw.rectangle([x-10, y-10, x+10, y+10], fill=color, outline='white', width=2)
        
        # Draw point number
        draw.text((x+15, y-15), f"{i+1}. {text}", fill=color, stroke_width=2, stroke_fill='white')
    
    # PRIORITY 4: Draw tip detection points (circles - different from ROI squares)
    for detection in detections:
        x, y = int(detection['x']), int(detection['y'])
        is_manual = detection.get('manual', False)
        color = 'green' if is_manual else 'red'
        
        # Small circles for tip detections
        draw.ellipse([x-4, y-4, x+4, y+4], fill=color, outline='white', width=1)
    
    return display_image
def handle_roi_point_click(
    clicked_point,
    image: Image.Image,
    session_prefix: str = "roi_"
):
    """Handle ROI corner point clicks - FIXED with duplicate prevention"""
    
    roi_points = st.session_state.get(f'{session_prefix}points', [])
    
    # Don't allow more than 2 points
    if len(roi_points) >= 2:
        st.warning("ROI already has 2 points. Use 'Clear ROI' first to add new ones.")
        st.session_state[f'{session_prefix}point_mode'] = False
        st.rerun()
        return
    
    click_x, click_y = clicked_point['x'], clicked_point['y']
    
    # Scale coordinates properly
    original_width, original_height = image.size
    display_width = 800
    scale_factor = original_width / display_width
    actual_x = click_x * scale_factor
    actual_y = click_y * scale_factor
    
    # DUPLICATE PREVENTION: Check if this is the same as the last point
    if len(roi_points) == 1:
        last_point = roi_points[0]
        distance = np.sqrt((actual_x - last_point['x'])**2 + (actual_y - last_point['y'])**2)
        
        # If too close to first point (less than 30 pixels), ignore
        if distance < 30:
            st.warning(f"⚠️ Too close to first point! Click farther away (distance: {int(distance)} pixels)")
            return
    
    # Create unique click identifier to prevent same click being processed twice
    click_id = f"{int(actual_x)}_{int(actual_y)}"
    last_click_id = st.session_state.get(f'{session_prefix}last_click_id', '')
    
    # If same click ID, ignore (duplicate click)
    if click_id == last_click_id:
        return
    
    # Store this click ID
    st.session_state[f'{session_prefix}last_click_id'] = click_id
    
    # Add ROI corner point
    roi_points.append({'x': actual_x, 'y': actual_y})
    st.session_state[f'{session_prefix}points'] = roi_points
    
    if len(roi_points) == 1:
        # First point added - STAY IN MODE and show success
        st.success(f"✅ **Point 1/2 placed at ({int(actual_x)}, {int(actual_y)})**")
        st.info("🖱️ **Click second corner to complete ROI (click farther away)**")
        # DON'T EXIT POINT MODE - keep it active for second point
        st.rerun()
        
    elif len(roi_points) == 2:
        # Second point added - STAY IN MODE, show success but don't create rectangle yet
        st.success(f"✅ **Point 2/2 placed at ({int(actual_x)}, {int(actual_y)})**")
        st.success("✅ **Both corners placed! Click 'Done Adding ROI' to create rectangle**")
        # DON'T EXIT POINT MODE - wait for user to click "Done"
        st.rerun()
def get_roi_display_image(
    image: Image.Image,
    detections: List[Dict],
    session_prefix: str = "roi_"
) -> Image.Image:
    """Get static display image with ROI and detections (no interaction) - FIXED"""
    
    display_image = image.copy()
    draw = ImageDraw.Draw(display_image)
    
    # Get ROI data
    roi_coordinates = st.session_state.get(f'{session_prefix}coordinates')
    roi_points = st.session_state.get(f'{session_prefix}points', [])
    
    # Draw final ROI rectangle if exists (thick orange)
    if roi_coordinates:
        try:
            x1, y1, x2, y2 = roi_coordinates
            # Ensure valid rectangle coordinates
            left = min(x1, x2)
            top = min(y1, y2) 
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            # Draw thick orange rectangle for final ROI
            for i in range(4):
                draw.rectangle([left+i, top+i, right-i, bottom-i], outline='orange', width=1)
            
            # Add corner markers
            corner_size = 12
            draw.rectangle([left-corner_size, top-corner_size, left+corner_size, top+corner_size], 
                          fill='orange', outline='white', width=2)
            draw.rectangle([right-corner_size, bottom-corner_size, right+corner_size, bottom+corner_size], 
                          fill='orange', outline='white', width=2)
            
            # Add ROI label
            draw.text((left+10, top-30), "ROI REGION", fill='orange', stroke_width=2, stroke_fill='white')
            
        except Exception as e:
            # If rectangle drawing fails, clear invalid ROI
            st.session_state[f'{session_prefix}coordinates'] = None
    
    # Also show the corner points if they exist (for reference)
    for i, point in enumerate(roi_points[:2]):  # Max 2 points
        x, y = int(point['x']), int(point['y'])
        color = 'blue' if i == 0 else 'purple'
        
        # Small square markers for corner points
        draw.rectangle([x-6, y-6, x+6, y+6], fill=color, outline='white', width=1)
    
    # Draw tip detection points (circles)
    for detection in detections:
        x, y = int(detection['x']), int(detection['y'])
        is_manual = detection.get('manual', False)
        color = 'green' if is_manual else 'red'
        draw.ellipse([x-4, y-4, x+4, y+4], fill=color, outline='white')
    
    return display_image