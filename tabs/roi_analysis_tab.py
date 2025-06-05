# tabs/roi_analysis_tab.py

import streamlit as st
import os
import tempfile
import zipfile
import csv
import io
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple
from streamlit_image_coordinates import streamlit_image_coordinates
# Import core modules
from core.yolo_processor import create_yolo_processor
from core.roi_processor import create_roi_processor
from utils.detection_utils import add_manual_point, remove_point_by_index
from utils.visualization_utils import draw_detections_on_image
from config.model_config import get_model_config


from components.improved_point_editor import integrate_point_editor_with_detections
from components.roi_point_editor import (
    show_roi_point_interface,
    show_roi_point_editor,
    get_roi_display_image
)
def show_roi_analysis_interface():
    """Complete ROI Analysis tab with all features fixed"""
    
    st.markdown("### 🎯 ROI Analysis")
    st.markdown("**Region of Interest Detection:** Draw a box on the image to analyze specific areas")
    
    # Initialize session state
    init_roi_session_state()
    
    # Auto-load ROI model
    load_roi_model()
    
    # Show common menus
    show_common_menus()
    
    # Main layout
    col_main, col_right = st.columns([3, 1])
    
    with col_main:
        show_image_display_area()
    
    with col_right:
        show_right_panel()

def init_roi_session_state():
    """Initialize session state for ROI analysis tab - SIMPLE version"""
    
    # Tab-specific state isolation
    if 'roi_detections' not in st.session_state:
        st.session_state.roi_detections = []
    if 'roi_current_image' not in st.session_state:
        st.session_state.roi_current_image = None
    if 'roi_current_image_path' not in st.session_state:
        st.session_state.roi_current_image_path = None
    if 'roi_batch_results' not in st.session_state:
        st.session_state.roi_batch_results = {}
    if 'roi_image_files' not in st.session_state:
        st.session_state.roi_image_files = []
    if 'roi_current_image_index' not in st.session_state:
        st.session_state.roi_current_image_index = 0
    if 'roi_editing_mode' not in st.session_state:
        st.session_state.roi_editing_mode = None
    
    # ROI-specific state
    if 'roi_coordinates' not in st.session_state:
        st.session_state.roi_coordinates = None
    if 'roi_processor' not in st.session_state:
        st.session_state.roi_processor = None
    if 'roi_drawing_mode' not in st.session_state:
        st.session_state.roi_drawing_mode = False
    if 'roi_drawing_points' not in st.session_state:
        st.session_state.roi_drawing_points = []
    if 'roi_show_manual_input' not in st.session_state:
        st.session_state.roi_show_manual_input = False
    
    # Upload counters
    if 'roi_upload_counter' not in st.session_state:
        st.session_state.roi_upload_counter = 0
    if 'roi_folder_counter' not in st.session_state:
        st.session_state.roi_folder_counter = 0
    if 'roi_points' not in st.session_state:
        st.session_state.roi_points = []
    if 'roi_point_mode' not in st.session_state:
        st.session_state.roi_point_mode = False
    if 'roi_last_click_id' not in st.session_state:
        st.session_state.roi_last_click_id = ''
    # Display size setting
    if 'display_size' not in st.session_state:
        st.session_state.display_size = "Fit to View"
def load_roi_model():
    """Auto-load ROI model on startup"""
    if st.session_state.roi_processor is None:
        try:
            config = get_model_config("roi_model")  # Use dedicated ROI model
            if config and os.path.exists(config["path"]):
                yolo_processor = create_yolo_processor(config)
                if yolo_processor.is_loaded():
                    roi_processor = create_roi_processor(yolo_processor)
                    st.session_state.roi_processor = roi_processor
        except Exception as e:
            # Silent loading - don't show error unless user tries to use it
            pass    
def show_common_menus():
    """FIXED common menus - EXACT correlation with other tabs"""
    
    # File Menu
    with st.expander("📁 File Menu", expanded=False):
        col1, col2, col3, col4 ,col5= st.columns(5)
        
        with col1:
            # FIXED: Use dynamic key that changes after detection
            upload_key = f"roi_upload_image_{st.session_state.get('roi_upload_counter', 0)}"
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key=upload_key
            )
            if uploaded_file:
                handle_image_upload(uploaded_file)
        
        with col2:
            # FIXED: Use dynamic key for folder upload
            folder_key = f"roi_upload_folder_{st.session_state.get('roi_folder_counter', 0)}"
            uploaded_folder = st.file_uploader(
                "Upload Folder", 
                type=['zip'],
                key=folder_key
            )
            if uploaded_folder:
                handle_folder_upload(uploaded_folder)
        
        with col3:
            if st.button("💾 Save Image", key="roi_save_image"):
                save_current_image()
        
        with col4:
            if st.button("📊 Save Coordinate as CSV", key="roi_save_csv"):
                save_coordinates_csv()
        with col5:
            if st.button("📤 Send Current to Annotation", key="roi_send_current", disabled=not has_current_detections()):
                send_current_image_to_annotation()
        
        
    
    # Reset Menu
    with st.expander("🔄 Reset Menu", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Reset Current Detection", key="roi_reset_current_detection"):
                st.session_state.roi_detections = []
                st.session_state.roi_coordinates = None
                st.session_state.roi_drawing_points = []
                st.session_state.roi_editing_mode = None
                st.success("Current detections and ROI cleared")
                st.rerun()
        
        with col2:
            if st.button("🖼️ Reset Current Image", key="roi_reset_current_image"):
                st.session_state.roi_current_image = None
                st.session_state.roi_current_image_path = None
                st.session_state.roi_detections = []
                st.session_state.roi_coordinates = None
                st.session_state.roi_drawing_points = []
                st.session_state.roi_editing_mode = None
                st.session_state.roi_points = []  # Also clear ROI points
                st.session_state.roi_point_mode = False
                st.success("Image and all data cleared")
                st.rerun()
        
        with col3:
            if st.button("📦 Reset Batch Detection", key="roi_reset_batch_detection"):
                st.session_state.roi_batch_results = {}
                st.session_state.roi_detections = []
                st.session_state.roi_coordinates = None
                st.session_state.roi_drawing_points = []
                st.success("Batch results and current detections cleared")
                st.rerun()
        
        with col4:
            if st.button("🗑️ Reset All Batch", key="roi_reset_all_batch"):
                st.session_state.roi_batch_results = {}
                st.session_state.roi_image_files = []
                st.session_state.roi_current_image_index = 0
                st.session_state.roi_current_image = None
                st.session_state.roi_current_image_path = None
                st.session_state.roi_detections = []
                st.session_state.roi_coordinates = None
                st.session_state.roi_drawing_points = []
                st.session_state.roi_editing_mode = None
                st.success("All data cleared")
                st.rerun()

def show_image_display_area():
    """Main image display area - FIXED to prevent duplicate keys"""
    
    if st.session_state.roi_current_image is None:
        st.info("👆 Upload an image to start ROI analysis")
        return
    
    # Check modes
    roi_point_mode = st.session_state.get('roi_point_mode', False)
    tip_editing_mode = st.session_state.get('roi_editing_mode')
    
    if roi_point_mode:
        # ROI point adding mode (uses component)
        from components.roi_point_editor import show_roi_point_editor
        show_roi_point_editor(
            st.session_state.roi_current_image,
            st.session_state.roi_detections,
            "roi_"
        )
    elif tip_editing_mode in ['add', 'remove']:
        # Tip detection point editing mode (uses existing component)
        from components.improved_point_editor import integrate_point_editor_with_detections
        updated_detections = integrate_point_editor_with_detections(
            st.session_state.roi_current_image,
            st.session_state.roi_detections,
            "roi_"
        )
        if updated_detections != st.session_state.roi_detections:
            st.session_state.roi_detections = updated_detections
    else:
        # Static display (uses component)
        show_static_display()
    
    # FIXED: Only show ROI controls ONCE - not in multiple places
    # ROI controls (uses component) - ONLY if not in tip editing mode
    if tip_editing_mode not in ['add', 'remove']:
        from components.roi_point_editor import show_roi_point_interface
        show_roi_point_interface(st.session_state.roi_current_image, "roi_")
    
    # Tip editing controls (existing) - ONLY if not in ROI point mode
    if not roi_point_mode:
        show_edit_points_interface()
    
    # Batch navigation
    if st.session_state.roi_image_files:
        show_batch_navigation()
def has_current_detections():
    """Helper to check if current image has detections"""
    return (st.session_state.roi_current_image is not None and 
            len(st.session_state.roi_detections) > 0)
def show_static_display():
    """Static image display using component"""
    
    from components.roi_point_editor import get_roi_display_image
    
    display_image = get_roi_display_image(
        st.session_state.roi_current_image,
        st.session_state.roi_detections,
        "roi_"
    )
    
    # Apply display size settings
    display_size = st.session_state.get('display_size', 'Fit to View')
    
    if display_size == "Small":
        st.image(display_image, caption="ROI Analysis", width=400)
    elif display_size == "Medium":
        st.image(display_image, caption="ROI Analysis", width=600)
    elif display_size == "Large":
        st.image(display_image, caption="ROI Analysis", width=800)
    else:  # Fit to View
        st.image(display_image, caption="ROI Analysis", use_container_width=True)
def show_edit_points_interface():
    """Edit Points interface - FIXED key collision"""
    
    st.markdown("#### ✏️ Edit Points")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Point", key="roi_add_point_btn"):
            if st.session_state.roi_current_image is not None:
                st.session_state.roi_editing_mode = 'add'
                st.session_state.roi_drawing_mode = False
                st.info("🖱️ **ADD MODE** - Click on the image to add points")
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("➖ Remove Point", key="roi_remove_point_btn"):
            if st.session_state.roi_detections:
                st.session_state.roi_editing_mode = 'remove'
                st.session_state.roi_drawing_mode = False
                st.info("🖱️ **REMOVE MODE** - Click near points to remove them")
                st.rerun()
            else:
                st.warning("No points to remove")
    
    with col3:
        if st.button("🗑️ Clear All Points", key="roi_clear_all_points_btn"):
            st.session_state.roi_detections = []
            st.session_state.roi_editing_mode = None
            st.success("All points cleared")
            st.rerun()
    
    # Show current editing mode status (only when active)
    current_mode = st.session_state.get('roi_editing_mode')
    if current_mode == 'add':
        st.info("🖱️ **ADD MODE ACTIVE** - Click on the image above to add points")
    elif current_mode == 'remove':
        st.info("🖱️ **REMOVE MODE ACTIVE** - Click near points to remove them")
    
    # FIXED: Use unique key for ROI tab
    if current_mode in ['add', 'remove'] and st.session_state.roi_current_image is not None:
        updated_detections = integrate_point_editor_with_detections(
            st.session_state.roi_current_image,
            st.session_state.roi_detections,
            "roi_tips_"  # FIXED: Different prefix to avoid collision with ROI point editor
        )
        
        # Update detections if changed
        if updated_detections != st.session_state.roi_detections:
            st.session_state.roi_detections = updated_detections
    
    # Show statistics (same as other tabs)
    if st.session_state.roi_detections:
        manual_count = sum(1 for d in st.session_state.roi_detections if d.get('manual', False))
        auto_count = len(st.session_state.roi_detections) - manual_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", len(st.session_state.roi_detections))
        with col2:
            st.metric("Auto Points", auto_count)
        with col3:
            st.metric("Manual Points", manual_count)
    else:
        st.info("No detection points yet. Run detection or add points manually.")
def show_batch_navigation():
    """Batch Navigation - ADD send batch button"""
    
    if 'roi_current_image_index' not in st.session_state:
        st.session_state.roi_current_image_index = 0
    
    total_images = len(st.session_state.roi_image_files)
    current_idx = min(st.session_state.roi_current_image_index, total_images - 1)
    
    if current_idx != st.session_state.roi_current_image_index:
        st.session_state.roi_current_image_index = current_idx
    
    st.markdown("---")
    st.markdown("### 📂 Batch Navigation")
    
    # Get original filename
    if hasattr(st.session_state, 'roi_batch_original_names') and current_idx < len(st.session_state.roi_batch_original_names):
        current_name = st.session_state.roi_batch_original_names[current_idx]
    else:
        current_name = os.path.basename(st.session_state.roi_image_files[current_idx])
    
    st.markdown(f"**Image {current_idx + 1} of {total_images}:** {current_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)  # ADD col5
    
    with col1:
        if st.button("◀ Previous", disabled=current_idx <= 0, key="roi_batch_prev"):
            navigate_to_image(current_idx - 1)
    
    with col2:
        if st.button("Next ▶", disabled=current_idx >= total_images - 1, key="roi_batch_next"):
            navigate_to_image(current_idx + 1)
    
    with col3:
        if st.button("🚀 Process All", key="roi_batch_process_all"):
            run_all_images()
    
    with col4:
        if st.button("💾 Download Batch", key="roi_batch_download", disabled=not st.session_state.roi_batch_results):
            show_batch_download()
    
    with col5:  # ADD send batch button
        if st.button("📤 Send Batch to Annotation", key="roi_send_batch", disabled=not st.session_state.roi_batch_results):
            send_batch_to_annotation()
# ROI Analysis Tab - FIXED Batch Download and Navigation Functions
def send_current_image_to_annotation():
    """Send current image + detections + ROI (including manual edits) to annotation"""
    if not st.session_state.roi_current_image or not st.session_state.roi_current_image_path:
        st.warning("No current image to send")
        return
    
    if not st.session_state.roi_detections:
        st.warning("No detections to send. Run detection first or add manual points.")
        return
    
    # Get original filename if available
    if hasattr(st.session_state, 'roi_batch_original_names') and st.session_state.roi_image_files:
        try:
            current_idx = st.session_state.roi_image_files.index(st.session_state.roi_current_image_path)
            original_name = st.session_state.roi_batch_original_names[current_idx]
        except (ValueError, IndexError):
            original_name = 'current_image.png'
    else:
        original_name = 'current_image.png'
    
    # Prepare data for annotation (includes manual edits + ROI)
    annotation_data = {
        'image_path': st.session_state.roi_current_image_path,
        'image': st.session_state.roi_current_image,
        'detections': st.session_state.roi_detections.copy(),  # Includes manual add/remove
        'roi_coordinates': st.session_state.roi_coordinates,   # Include ROI coordinates
        'original_name': original_name,
        'source': 'roi_analysis_tab',
        'method': 'roi'
    }
    
    # Store in session state for annotation page
    st.session_state.annotation_received_image = annotation_data
    roi_info = f" (ROI: {st.session_state.roi_coordinates})" if st.session_state.roi_coordinates else ""
    st.success(f"✅ Current image sent to annotation tool! ({len(st.session_state.roi_detections)} points including manual edits{roi_info})")

def send_batch_to_annotation():
    """Send entire batch to annotation (including manual edits + ROI from current image)"""
    
    # CRITICAL: Save current image's manual edits and ROI first
    if st.session_state.roi_current_image_path:
        current_path = st.session_state.roi_current_image_path
        
        # Get original name
        if hasattr(st.session_state, 'roi_batch_original_names'):
            try:
                index = st.session_state.roi_image_files.index(current_path)
                original_name = st.session_state.roi_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # Save current detections AND ROI (including manual edits) to batch results
        st.session_state.roi_batch_results[current_path] = {
            'detections': st.session_state.roi_detections.copy(),  # Include manual edits
            'roi_coordinates': st.session_state.roi_coordinates,   # Include ROI
            'image_name': original_name,
            'method': 'roi'
        }
    
    if not st.session_state.roi_batch_results:
        st.warning("No batch results to send")
        return
    
    # Prepare batch data (includes all manual edits + ROIs)
    batch_data = {
        'image_files': st.session_state.roi_image_files.copy(),
        'original_names': getattr(st.session_state, 'roi_batch_original_names', []),
        'batch_results': st.session_state.roi_batch_results.copy(),  # Includes manual edits + ROI
        'source': 'roi_analysis_tab',
        'method': 'roi'
    }
    
    st.session_state.annotation_received_batch = batch_data
    
    # Count total points including manual edits
    total_points = sum(len(r['detections']) for r in st.session_state.roi_batch_results.values())
    roi_count = sum(1 for r in st.session_state.roi_batch_results.values() if r.get('roi_coordinates'))
    st.success(f"✅ Batch sent to annotation tool! ({len(st.session_state.roi_batch_results)} images, {total_points} total points, {roi_count} ROIs including manual edits)")
def navigate_to_image(new_index: int):
    """Navigate to specific batch image - FIXED to save manual edits and ROI"""
    
    total_images = len(st.session_state.roi_image_files)
    
    if not (0 <= new_index < total_images):
        return
    
    # CRITICAL FIX: Save current ROI and detections ALWAYS (including manual edits)
    if st.session_state.roi_current_image_path:
        current_path = st.session_state.roi_current_image_path
        
        if hasattr(st.session_state, 'roi_batch_original_names'):
            try:
                current_original_idx = st.session_state.roi_image_files.index(current_path)
                original_name = st.session_state.roi_batch_original_names[current_original_idx]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # ALWAYS save current detections and ROI (including manual edits and empty lists)
        st.session_state.roi_batch_results[current_path] = {
            'detections': st.session_state.roi_detections.copy(),  # Include manual edits
            'roi_coordinates': st.session_state.roi_coordinates,  # Save ROI coordinates
            'image_name': original_name,
            'method': 'roi'
        }
    
    # Load new image
    new_image_path = st.session_state.roi_image_files[new_index]
    
    try:
        image = Image.open(new_image_path)
        st.session_state.roi_current_image = image
        st.session_state.roi_current_image_path = new_image_path
        st.session_state.roi_current_image_index = new_index
        
        # Restore ROI and detections if they exist
        if new_image_path in st.session_state.roi_batch_results:
            result = st.session_state.roi_batch_results[new_image_path]
            st.session_state.roi_detections = result['detections'].copy()
            st.session_state.roi_coordinates = result.get('roi_coordinates')
        else:
            st.session_state.roi_detections = []
            st.session_state.roi_coordinates = None
        
        # Reset drawing states
        st.session_state.roi_drawing_mode = False
        st.session_state.roi_drawing_points = []
        st.session_state.roi_editing_mode = None
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def show_batch_download():
    """FIXED - Save current image before creating download"""
    
    # CRITICAL FIX: Save current image's manual edits and ROI before creating download
    if st.session_state.roi_current_image_path:
        current_path = st.session_state.roi_current_image_path
        
        # Get original filename
        if hasattr(st.session_state, 'roi_batch_original_names'):
            try:
                index = st.session_state.roi_image_files.index(current_path)
                original_name = st.session_state.roi_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # SAVE CURRENT DETECTIONS AND ROI (including manual edits)
        st.session_state.roi_batch_results[current_path] = {
            'detections': st.session_state.roi_detections.copy(),  # This includes manual edits!
            'roi_coordinates': st.session_state.roi_coordinates,  # Save ROI coordinates
            'image_name': original_name,
            'method': 'roi'
        }
    
    # Check if we have any results to download
    if not st.session_state.roi_batch_results:
        st.warning("No batch results to download")
        return
    
    try:
        import zipfile
        import io
        import csv
        import os
        from PIL import ImageDraw
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Create summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Tips', 'Auto_Tips', 'Manual_Tips', 'Method', 'ROI_Coordinates', 'ROI_Width', 'ROI_Height'])
            
            processed_names = set()  # Track processed files to avoid duplicates
            
            for image_path, results in st.session_state.roi_batch_results.items():
                detections = results['detections']
                roi_coords = results.get('roi_coordinates', (0, 0, 0, 0))
                
                # Get original name from mapping
                if hasattr(st.session_state, 'roi_batch_original_names'):
                    try:
                        index = st.session_state.roi_image_files.index(image_path)
                        original_name = st.session_state.roi_batch_original_names[index]
                    except (ValueError, IndexError):
                        original_name = results.get('image_name', os.path.basename(image_path))
                else:
                    original_name = results.get('image_name', os.path.basename(image_path))
                
                # Skip if already processed (avoid duplicates)
                if original_name in processed_names:
                    continue
                processed_names.add(original_name)
                
                base_name = os.path.splitext(original_name)[0]
                
                # Count manual vs auto tips
                total_tips = len(detections)
                manual_tips = sum(1 for d in detections if d.get('manual', False))
                auto_tips = total_tips - manual_tips
                
                # Calculate ROI dimensions
                if roi_coords and len(roi_coords) == 4:
                    x1, y1, x2, y2 = roi_coords
                    roi_width = abs(x2 - x1)
                    roi_height = abs(y2 - y1)
                    roi_coords_str = f"({x1},{y1},{x2},{y2})"
                else:
                    roi_width = roi_height = 0
                    roi_coords_str = "No ROI"
                
                # Add to summary
                summary_writer.writerow([
                    original_name,
                    total_tips,
                    auto_tips,
                    manual_tips,
                    results.get('method', 'roi'),
                    roi_coords_str,
                    roi_width,
                    roi_height
                ])
                
                # Create annotated image if detections or ROI exist
                if detections or roi_coords:
                    try:
                        original_image = Image.open(image_path)
                        annotated_image = original_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        
                        # Draw ROI rectangle
                        if roi_coords and len(roi_coords) == 4:
                            x1, y1, x2, y2 = roi_coords
                            for i in range(3):
                                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline='orange', width=1)
                        
                        # Draw detections
                        for detection in detections:
                            x, y = int(detection['x']), int(detection['y'])
                            is_manual = detection.get('manual', False)
                            color = 'lime' if is_manual else 'red'
                            draw.ellipse([x-6, y-6, x+6, y+6], fill=color, outline='white', width=2)
                        
                        # Save annotated image to ZIP
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='PNG')
                        zip_file.writestr(f"annotated_images/{base_name}_roi_detected.png", img_buffer.getvalue())
                        
                    except Exception as e:
                        st.warning(f"Could not create annotated image for {original_name}: {str(e)}")
                
                # Individual CSV with coordinates
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'confidence', 'method', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2', 'type'])
                
                for detection in detections:
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords if roi_coords and len(roi_coords) == 4 else (0, 0, 0, 0)
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('conf', 1.0),
                        detection.get('method', 'roi'),
                        roi_x1, roi_y1, roi_x2, roi_y2,
                        'manual' if detection.get('manual', False) else 'automatic'
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_roi_coordinates.csv", coord_output.getvalue())
            
            # Add summary
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Complete ROI Batch Results",
            data=zip_content,
            file_name="roi_analysis_batch_results.zip",
            mime="application/zip",
            key="download_complete_roi_batch"
        )
        
        st.success(f"Batch download includes {len(processed_names)} unique images: annotated images + coordinate CSVs + summary")
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")

def run_all_images():
    """Detect tips on all batch images using their ROIs - FIXED to save manual edits"""
    
    if not st.session_state.roi_image_files:
        st.warning("No batch images loaded")
        return
    
    if not st.session_state.roi_processor:
        st.error("ROI model not loaded")
        return
    
    # Check if we should use same ROI for all images
    use_same_roi = st.session_state.get('batch_same_roi', False)
    current_roi = st.session_state.roi_coordinates
    
    if use_same_roi and not current_roi:
        st.error("Please set ROI first for batch processing")
        return
    
    # CLEAR batch results before processing (don't preserve manual edits during batch processing)
    st.session_state.roi_batch_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, image_file in enumerate(st.session_state.roi_image_files):
            progress = (i + 1) / len(st.session_state.roi_image_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {os.path.basename(image_file)} ({i+1}/{len(st.session_state.roi_image_files)})")
            
            try:
                # Determine ROI to use
                if use_same_roi:
                    roi_coords = current_roi
                else:
                    # Check if this image has saved ROI coordinates
                    if str(image_file) in st.session_state.roi_batch_results:
                        roi_coords = st.session_state.roi_batch_results[str(image_file)].get('roi_coordinates')
                        if not roi_coords:
                            st.warning(f"No ROI defined for {os.path.basename(image_file)}, skipping...")
                            continue
                    else:
                        st.warning(f"No ROI defined for {os.path.basename(image_file)}, skipping...")
                        continue
                
                # Run ROI detection
                detections = st.session_state.roi_processor.process_roi_on_full_image(
                    str(image_file),
                    roi_coords,
                    conf_thresh=0.25
                )
                
                # Get original name
                if hasattr(st.session_state, 'roi_batch_original_names'):
                    try:
                        index = st.session_state.roi_image_files.index(str(image_file))
                        original_name = st.session_state.roi_batch_original_names[index]
                    except (ValueError, IndexError):
                        original_name = os.path.basename(image_file)
                else:
                    original_name = os.path.basename(image_file)
                
                # Update batch results
                st.session_state.roi_batch_results[str(image_file)] = {
                    'detections': detections,
                    'roi_coordinates': roi_coords,
                    'image_name': original_name,
                    'method': 'roi'
                }
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(image_file)}: {str(e)}")
        
        total_detections = sum(len(r['detections']) for r in st.session_state.roi_batch_results.values())
        processed_count = len([r for r in st.session_state.roi_batch_results.values() if r['detections']])
        st.success(f"Batch ROI processing complete! {processed_count} images processed, {total_detections} total detections")
        
        # FIXED: Reset upload counters after batch processing
        st.session_state.roi_upload_counter += 1
        st.session_state.roi_folder_counter += 1
        
        # Refresh current image if it was processed
        if st.session_state.roi_current_image_path in st.session_state.roi_batch_results:
            result = st.session_state.roi_batch_results[st.session_state.roi_current_image_path]
            st.session_state.roi_detections = result['detections'].copy()
            st.session_state.roi_coordinates = result.get('roi_coordinates')
            st.rerun()
        
    finally:
        progress_bar.empty()
        status_text.empty()

def run_roi_detection():
    """Run detection within selected ROI - FIXED to save manual edits"""
    
    if not st.session_state.roi_coordinates:
        st.error("No ROI selected")
        return
    
    if not st.session_state.roi_processor:
        st.error("ROI model not loaded")
        return
    
    if not st.session_state.roi_current_image_path:
        st.error("No image loaded")
        return
    
    try:
        with st.spinner("Running ROI detection..."):
            # FIXED: Store existing manual points before detection
            existing_manual_points = [d for d in st.session_state.roi_detections if d.get('manual', False)]
            
            # Run detection
            detections = st.session_state.roi_processor.process_roi_on_full_image(
                st.session_state.roi_current_image_path,
                st.session_state.roi_coordinates,
                conf_thresh=0.25
            )
            
            # FIXED: Combine new detections with existing manual points
            st.session_state.roi_detections = existing_manual_points + detections
            
            # Reset upload counter after detection is complete
            st.session_state.roi_upload_counter += 1
            
            st.success(f"ROI detection completed! Found {len(detections)} new leaf tips in selected region.")
            if existing_manual_points:
                st.info(f"Preserved {len(existing_manual_points)} existing manual points")
            st.rerun()
            
    except Exception as e:
        st.error(f"ROI detection failed: {str(e)}")

def save_coordinates_csv():
    """Save detected tip coordinates to CSV - FIXED to include manual edits"""
    if not st.session_state.roi_detections:
        st.warning("No detections to save")
        return
    
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['x', 'y', 'confidence', 'method', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2', 'type'])
        
        # Use actual ROI coordinates if they exist
        roi_coords = st.session_state.roi_coordinates
        if roi_coords and len(roi_coords) == 4:
            x1, y1, x2, y2 = roi_coords
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        # FIXED: Include all detections (both manual and automatic)
        for detection in st.session_state.roi_detections:
            writer.writerow([
                detection['x'],
                detection['y'],
                detection.get('conf', 1.0),
                detection.get('method', 'roi'),
                x1, y1, x2, y2,
                'manual' if detection.get('manual', False) else 'automatic'
            ])
        
        csv_content = output.getvalue()
        
        st.download_button(
            label="⬇️ Download ROI Coordinates CSV",
            data=csv_content,
            file_name="roi_analysis_coordinates.csv",
            mime="text/csv",
            key="roi_download_csv"
        )
        
    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")

def show_right_panel():
    """Right panel with ROI detection controls"""
    
    st.markdown("#### 🎛️ ROI Controls")
    
    # ROI Model Status
    show_model_status()
    
    # Run ROI Detection
    show_run_detection()
    
    # Tip Count
    show_tip_count()
    
    # Batch ROI Settings (for batch processing)
    show_batch_roi_settings()

def show_model_status():
    """Show ROI model loading status"""
    
    st.markdown("**Model Status:**")
    if st.session_state.roi_processor:
        st.success("✅ ROI Model Ready")
    else:
        st.warning("⚪ ROI Model Not Loaded")
        if st.button("📥 Load ROI Model", key="load_roi_model_btn"):
            load_roi_model_manually()

def load_roi_model_manually():
    """Manually load ROI model"""
    try:
        config = get_model_config("roi_model")  # Use dedicated ROI model
        if config:
            with st.spinner("Loading ROI model..."):
                yolo_processor = create_yolo_processor(config)
                
                if yolo_processor.is_loaded():
                    roi_processor = create_roi_processor(yolo_processor)
                    st.session_state.roi_processor = roi_processor
                    st.success("✅ ROI model loaded successfully")
                    st.rerun()
                else:
                    st.error("❌ Failed to load ROI model")
        else:
            st.error("ROI model configuration not found")
    except Exception as e:
        st.error(f"Error loading ROI model: {str(e)}")

def show_run_detection():
    """Run ROI Detection button"""
    
    # Check requirements
    roi_selected = st.session_state.roi_coordinates is not None
    model_ready = st.session_state.roi_processor is not None
    image_loaded = st.session_state.roi_current_image is not None
    
    detect_disabled = not (roi_selected and model_ready and image_loaded)
    
    if st.button(
        "🚀 Run ROI Detection",
        key="roi_run_detection",
        disabled=detect_disabled,
        use_container_width=True,
        type="primary"
    ):
        run_roi_detection()
    
    # Show status
    if not image_loaded:
        st.warning("⚠️ No image loaded")
    elif not roi_selected:
        st.warning("⚠️ No ROI selected")
    elif not model_ready:
        st.warning("⚠️ No model loaded")
    else:
        st.info("✅ Ready for detection")

def show_tip_count():
    """Dynamic display of detected tips"""
    
    count = len(st.session_state.roi_detections)
    manual_count = sum(1 for d in st.session_state.roi_detections if d.get('manual', False))
    auto_count = count - manual_count
    
    st.markdown("**Tip Count:**")
    st.metric("Total Tips", count)
    
    if count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Auto", auto_count)
        with col2:
            st.metric("Manual", manual_count)

def show_batch_roi_settings():
    """Batch ROI Settings for processing multiple images"""
    
    st.markdown("---")
    st.markdown("**Batch ROI Settings:**")
    
    # Option 1: Use same ROI for all images
    use_same_roi = st.checkbox(
        "Use same ROI for all batch images",
        value=False,
        key="batch_same_roi",
        help="Apply current ROI coordinates to all images in batch"
    )
    
    if use_same_roi and st.session_state.roi_coordinates:
        x1, y1, x2, y2 = st.session_state.roi_coordinates
        st.info(f"ROI: ({x1}, {y1}) to ({x2}, {y2})")
        st.caption("This ROI will be applied to all batch images")
    elif use_same_roi:
        st.warning("Please set ROI first")

def handle_folder_upload(uploaded_folder):
    """Handle folder (ZIP) upload - EXACT correlation with other tabs"""
    
    # Show immediate feedback
    st.info("📦 Processing ZIP file...")
    
    try:
        # Clear any existing data completely
        st.session_state.roi_current_image = None
        st.session_state.roi_current_image_path = None
        st.session_state.roi_detections = []
        st.session_state.roi_coordinates = None
        st.session_state.roi_drawing_points = []
        st.session_state.roi_drawing_mode = False
        st.session_state.roi_editing_mode = None
        st.session_state.roi_image_files = []
        st.session_state.roi_batch_original_names = []
        st.session_state.roi_batch_results = {}
        
        # Process the ZIP file
        with st.spinner("Extracting and loading images..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "upload.zip")
                
                with open(zip_path, "wb") as f:
                    f.write(uploaded_folder.read())
                
                extract_dir = os.path.join(temp_dir, "extracted")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find image files (avoid duplicates)
                image_files = []
                seen_names = set()
                
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    for img_file in Path(extract_dir).rglob(f"*{ext}"):
                        if img_file.name not in seen_names:
                            image_files.append(img_file)
                            seen_names.add(img_file.name)
                    for img_file in Path(extract_dir).rglob(f"*{ext.upper()}"):
                        if img_file.name not in seen_names:
                            image_files.append(img_file)
                            seen_names.add(img_file.name)
                
                if image_files:
                    # Copy to persistent location (no duplicates)
                    persistent_files = []
                    original_names = []
                    
                    for img_file in image_files:
                        # Create unique temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=img_file.suffix) as tmp:
                            with open(img_file, 'rb') as src:
                                tmp.write(src.read())
                            persistent_files.append(tmp.name)
                            original_names.append(img_file.name)
                    
                    # Store batch data (ensure no duplicates)
                    st.session_state.roi_image_files = persistent_files
                    st.session_state.roi_batch_original_names = original_names
                    st.session_state.roi_current_image_index = 0
                    st.session_state.roi_batch_results = {}
                    
                    # Increment folder counter to reset uploader
                    st.session_state.roi_folder_counter += 1
                    
                    # Load first image
                    navigate_to_image(0)
                    
                    st.success(f"✅ Successfully loaded {len(image_files)} unique images!")
                    st.info("👇 Use the navigation controls below the image to browse through your batch.")
                    st.rerun()
                    
                else:
                    st.warning("⚠️ No image files found in the ZIP archive.")
                    
    except Exception as e:
        st.error(f"❌ Error processing ZIP file: {str(e)}")
        st.info("Please make sure your ZIP file contains valid image files (.png, .jpg, .jpeg, .bmp, .tiff)")
def handle_image_upload(uploaded_file):
    """Handle single image upload - FIXED to increment counter immediately"""
    try:
        # CRITICAL FIX: Increment upload counter FIRST to reset file uploader
        st.session_state.roi_upload_counter += 1
        
        # Clear any existing batch data when single image uploaded
        st.session_state.roi_image_files = []
        st.session_state.roi_batch_results = {}
        st.session_state.roi_current_image_index = 0
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load image
        image = Image.open(temp_path)
        st.session_state.roi_current_image = image
        st.session_state.roi_current_image_path = temp_path
        
        # Clear previous data
        st.session_state.roi_detections = []
        st.session_state.roi_coordinates = None
        st.session_state.roi_drawing_points = []
        st.session_state.roi_drawing_mode = False
        st.session_state.roi_editing_mode = None
        st.session_state.roi_points = []  # Clear ROI points
        st.session_state.roi_point_mode = False
        st.session_state.roi_last_click_id = ''  # Reset click tracking
        
        st.success(f"Image loaded: {uploaded_file.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def save_current_image():
    """Save current image with ROI and detections"""
    if not st.session_state.roi_current_image:
        st.warning("No image loaded")
        return
    
    try:
        # Create annotated image
        annotated_image = st.session_state.roi_current_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Draw ROI rectangle if exists
        if st.session_state.roi_coordinates:
            x1, y1, x2, y2 = st.session_state.roi_coordinates
            for i in range(3):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline='orange', width=1)
        
        # Draw detections
        for detection in st.session_state.roi_detections:
            x, y = int(detection['x']), int(detection['y'])
            draw.ellipse([x-4, y-4, x+4, y+4], fill='red', outline='white')
        
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download ROI Analysis Image",
            data=img_bytes,
            file_name="roi_analysis_result.png",
            mime="image/png",
            key="roi_download_image"
        )
        
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")

