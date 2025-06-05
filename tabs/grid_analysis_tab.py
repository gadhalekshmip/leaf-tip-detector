# tabs/grid_analysis_tab_enhanced.py

import streamlit as st
import os
import tempfile
import zipfile
import csv
import io
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from core.grid_processor import create_grid_processor
# Import core modules
from core.yolo_processor import create_yolo_processor
from core.grid_processor import create_grid_processor
from utils.detection_utils import add_manual_point, remove_point_by_index
from utils.visualization_utils import (
    draw_detections_on_image, 
    draw_grid_lines,
    create_interactive_image_plot
)
from components.grid_debug_visualizer import show_grid_debug_visualization

from debug.stitching_visualizer import create_stitching_visualization
from config.model_config import get_model_config
from components.improved_point_editor import integrate_point_editor_with_detections
def show_grid_analysis_interface():
    """Enhanced Grid Analysis with ALL missing features including Debug Menu"""
    
    st.markdown("### 📊 Grid Analysis")
    st.markdown("Grid-based detection with 3x3 or 5x5 patterns and debug visualization")
    
    # Initialize session state
    init_grid_session_state()
    
    # Show common menus
    show_common_menus()
    
    # Show Debug Menu (only visible in Grid Analysis)
    show_debug_menu()
    
    # Main layout
    col_main, col_right = st.columns([3, 1])
    
    with col_main:
        show_image_display_area()
    
    with col_right:
        show_right_panel()




def show_right_panel():
    """Right panel with grid detection controls"""
    
    st.markdown("#### 🎛️ Grid Controls")
    
    # Detection Model Options: Grid 3x3 / Grid 5x5
    show_grid_model_selection()
    
    # Model loading buttons
    show_model_loading_buttons()
    
    # Run Detection button
    show_run_detection()
    
    # Tip Count
    show_tip_count()
    
    

def show_model_loading_buttons():
    """Model Loading Buttons: Load 3x3 Grid, Load 5x5 Grid"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Load FRCNN Grid 3x3", key="load_frcnn_grid_3x3"):
            load_grid_model('frcnn_grid_3x3')
    
    with col2:
        if st.button("📥 Load YOLO Grid 3x3", key="load_yolo_grid_3x3"):
            load_grid_model('yolo_grid_3x3')
    
    with col3:
        if st.button("📥 Load YOLO Grid 5x5", key="load_yolo_grid_5x5"):
            load_grid_model('yolo_grid_5x5')
    
    # Show model status
    show_model_status()
def load_grid_model(model_type):
    """Load grid model for specified type"""
    
    try:
        if model_type == 'frcnn_grid_3x3':
            config = get_model_config("frcnn_grid_3x3")
            session_key = 'grid_frcnn_3x3_processor'
            
            with st.spinner("Loading FRCNN Grid 3x3 model..."):
                from core.frcnn_grid_processor import create_frcnn_grid_processor
                grid_processor = create_frcnn_grid_processor(config)
                st.session_state[session_key] = grid_processor
                st.success("✅ FRCNN Grid 3x3 model loaded successfully")
                
        elif model_type == 'yolo_grid_3x3':
            config = get_model_config("grid_3x3")
            session_key = 'grid_3x3_processor'
            
            with st.spinner("Loading YOLO Grid 3x3 model..."):
                yolo_processor = create_yolo_processor(config)
                if yolo_processor.is_loaded():
                    grid_processor = create_grid_processor(yolo_processor)
                    st.session_state[session_key] = grid_processor
                    st.success("✅ YOLO Grid 3x3 model loaded successfully")
                else:
                    st.error("❌ Failed to load YOLO Grid 3x3 model")
                    
        elif model_type == 'yolo_grid_5x5':
            config = get_model_config("grid_5x5")
            session_key = 'grid_5x5_processor'
            
            with st.spinner("Loading YOLO Grid 5x5 model..."):
                yolo_processor = create_yolo_processor(config)
                if yolo_processor.is_loaded():
                    grid_processor = create_grid_processor(yolo_processor)
                    st.session_state[session_key] = grid_processor
                    st.success("✅ YOLO Grid 5x5 model loaded successfully")
                else:
                    st.error("❌ Failed to load YOLO Grid 5x5 model")
        
        else:
            st.error(f"Unknown model type: {model_type}")
            
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")

def show_model_status():
    """Show model loading status"""
    
    st.markdown("**Model Status:**")
    
    # 3x3 Grid status
    if st.session_state.get('grid_frcnn_3x3_processor'):
        st.success("✅ FRCNN Grid 3x3 Ready (Best)")
    else:
        st.warning("⚪ FRCNN Grid 3x3 Not Loaded")
    if st.session_state.grid_3x3_processor:
        st.success("✅ YOLO Grid 3x3 Grid Ready")
    else:
        st.warning("⚪YOLO Grid 3x3 Grid Not Loaded")
    
    # 5x5 Grid status
    if st.session_state.grid_5x5_processor:
        st.success("✅ YOLO Grid 5x5 Grid Ready")
    else:
        st.warning("⚪ YOLO Grid5x5 Grid Not Loaded")

def show_run_detection():
    """Run Detection button - FIXED to use grid_model_type"""
    
    # Get selected model type
    selected_model = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    
    # Check if appropriate model is loaded
    if selected_model == 'frcnn_grid_3x3':
        model_ready = st.session_state.get('grid_frcnn_3x3_processor') is not None
    elif selected_model == 'yolo_grid_3x3':
        model_ready = st.session_state.get('grid_3x3_processor') is not None
    elif selected_model == 'yolo_grid_5x5':
        model_ready = st.session_state.get('grid_5x5_processor') is not None
    else:
        model_ready = False
    
    detect_disabled = (st.session_state.grid_current_image is None or not model_ready)
    
    if st.button(
        "🚀 Run Detection",
        key="grid_run_detection",
        disabled=detect_disabled,
        use_container_width=True,
        type="primary"
    ):
        run_detection()

def show_tip_count():
    """Dynamic display of detected tips"""
    
    count = len(st.session_state.grid_detections)
    manual_count = sum(1 for d in st.session_state.grid_detections if d.get('manual', False))
    auto_count = count - manual_count
    
    st.markdown("**Tip Count:**")
    st.metric("Total Tips", count)
    
    if count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Auto", auto_count)
        with col2:
            st.metric("Manual", manual_count)
    
    # Show raw detection count if available
    if st.session_state.grid_raw_detections:
        raw_count = len(st.session_state.grid_raw_detections)
        duplicates_removed = raw_count - count
        st.caption(f"Raw detections: {raw_count} (Duplicates removed: {duplicates_removed})")


def handle_folder_upload(uploaded_folder):
    """Handle folder (ZIP) upload with proper status feedback"""
    
    # Show immediate feedback
    st.info("📦 Processing ZIP file...")
    
    try:
        # Clear any existing data completely
        st.session_state.grid_current_image = None
        st.session_state.grid_current_image_path = None
        st.session_state.grid_detections = []
        st.session_state.grid_batch_images = []  # Clear existing batch
        st.session_state.grid_batch_original_names = []  # Clear existing names
        st.session_state.grid_batch_results = {}
        
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
                    st.session_state.grid_image_files = persistent_files  # NOT entire_batch_images
                    st.session_state.grid_batch_original_names = original_names 
                    st.session_state.grid_current_batch_index = 0
                    st.session_state.grid_batch_results = {}
                    
                    # Increment folder counter to reset uploader
                    st.session_state.grid_folder_counter += 1  
                    
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


def show_edit_points_interface():
    """Improved point editing interface using streamlit_image_coordinates"""
    
    st.markdown("#### ✏️ Edit Points")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Point", key="grid_add_point_btn"):
            if st.session_state.grid_current_image is not None:
                st.session_state.grid_editing_mode = 'add'
                st.rerun()
            else:
                st.warning("Please upload an image first")
    
    with col2:
        if st.button("➖ Remove Point", key="grid_remove_point_btn"):
            if st.session_state.grid_detections:
                st.session_state.grid_editing_mode = 'remove'
                st.rerun()
            else:
                st.warning("No points to remove")
    
    with col3:
        if st.button("🗑️ Clear All Points", key="grid_clear_all_btn"):
            st.session_state.grid_detections = []
            st.session_state.grid_editing_mode = None
            st.success("All points cleared")
            st.rerun()
    
    # Show current editing mode status (only when active)
    current_mode = st.session_state.get('grid_editing_mode')
    if current_mode == 'add':
        st.info("🖱️ **ADD MODE ACTIVE** - Click on the image below to add points")
    elif current_mode == 'remove':
        st.info("🖱️ **REMOVE MODE ACTIVE** - Click near points to remove them")
    
    # Only use the interactive point editor when in editing mode AND image exists
    if current_mode in ['add', 'remove'] and st.session_state.grid_current_image is not None:
        updated_detections = integrate_point_editor_with_detections(
            st.session_state.grid_current_image,
            st.session_state.grid_detections,
            "grid_"  # Use grid_ prefix
        )
        
        # Update detections if changed
        if updated_detections != st.session_state.grid_detections:
            st.session_state.grid_detections = updated_detections
    
    # Show statistics
    if st.session_state.grid_detections:
        manual_count = sum(1 for d in st.session_state.grid_detections if d.get('manual', False))
        auto_count = len(st.session_state.grid_detections) - manual_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", len(st.session_state.grid_detections))
        with col2:
            st.metric("Auto Points", auto_count)
        with col3:
            st.metric("Manual Points", manual_count)
    else:
        st.info("No detection points yet. Run detection or add points manually.")
def show_batch_download():
    """Complete batch download for grid analysis tab - INCLUDES MANUAL EDITS"""
    
    # CRITICAL FIX: Save current image's manual edits before creating download
    if st.session_state.grid_current_image_path:
        current_path = st.session_state.grid_current_image_path
        
        # Get original filename
        if hasattr(st.session_state, 'grid_batch_original_names'):
            try:
                index = st.session_state.grid_image_files.index(current_path)
                original_name = st.session_state.grid_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        
        # SAVE CURRENT DETECTIONS (including manual add/remove edits)
        st.session_state.grid_batch_results[current_path] = {
            'detections': st.session_state.grid_detections.copy(),  # This includes manual edits!
            'raw_detections': st.session_state.grid_raw_detections.copy(),
            'image_name': original_name,
            'method': current_model_type,
            'model_type': current_model_type
        }
    
    # Check if we have any results to download
    if not st.session_state.grid_batch_results:
        st.warning("No batch results to download")
        return
    
    try:
        import zipfile
        import io
        import csv
        import os
        from utils.visualization_utils import draw_detections_on_image
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Create summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Tips', 'Auto_Tips', 'Manual_Tips', 'Raw_Tips', 'Method', 'Model_Type'])
            
            processed_names = set()  # Track processed files to avoid duplicates
            
            for image_path, results in st.session_state.grid_batch_results.items():
                detections = results['detections']
                raw_detections = results.get('raw_detections', [])
                
                # Get original name from mapping
                if hasattr(st.session_state, 'grid_batch_original_names'):
                    try:
                        index = st.session_state.grid_image_files.index(image_path)
                        original_name = st.session_state.grid_batch_original_names[index]
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
                raw_tips = len(raw_detections)
                
                # Use the ACTUAL method from results
                actual_method = results.get('method', 'frcnn_grid_3x3')
                actual_model_type = results.get('model_type', actual_method)
                
                # Add to summary
                summary_writer.writerow([
                    original_name,
                    total_tips,
                    auto_tips,
                    manual_tips,
                    raw_tips,
                    actual_method,
                    actual_model_type
                ])
                
                # Create annotated image if detections exist
                if detections:
                    try:
                        original_image = Image.open(image_path)
                        annotated_image = draw_detections_on_image(original_image, detections)
                        
                        # Save annotated image to ZIP with method name
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='PNG')
                        zip_file.writestr(f"annotated_images/{base_name}_detected_{actual_method}.png", img_buffer.getvalue())
                        
                    except Exception as e:
                        st.warning(f"Could not create annotated image for {original_name}: {str(e)}")
                
                # Individual CSV with coordinates
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'confidence', 'method', 'model_type', 'cell_row', 'cell_col', 'type'])
                
                for detection in detections:
                    cell_info = detection.get('cell', (None, None))
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('conf', 1.0),
                        actual_method,
                        actual_model_type,
                        cell_info[0] if cell_info[0] is not None else '',
                        cell_info[1] if cell_info[1] is not None else '',
                        'manual' if detection.get('manual', False) else 'automatic'
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_coordinates_{actual_method}.csv", coord_output.getvalue())
            
            # Add summary
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        # Get current model type for filename
        current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        
        st.download_button(
            label="⬇️ Download Complete Batch Results",
            data=zip_content,
            file_name=f"{current_model_type}_batch_results.zip",
            mime="application/zip",
            key="download_complete_batch_grid"
        )
        
        st.success(f"Batch download includes {len(processed_names)} unique images: annotated images + coordinate CSVs + summary")
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")

def navigate_to_image(new_index: int):
    """Navigate to specific batch image - FIXED to save manual edits"""
    
    total_images = len(st.session_state.grid_image_files)
    
    if not (0 <= new_index < total_images):
        return
    
    # CRITICAL FIX: Save current detections ALWAYS (including manual edits)
    if st.session_state.grid_current_image_path:
        current_path = st.session_state.grid_current_image_path
        
        if hasattr(st.session_state, 'grid_batch_original_names'):
            try:
                current_original_idx = st.session_state.grid_image_files.index(current_path)
                original_name = st.session_state.grid_batch_original_names[current_original_idx]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
            
        current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        
        # ALWAYS save current detections (including empty lists and manual edits)
        st.session_state.grid_batch_results[current_path] = {
            'detections': st.session_state.grid_detections.copy(),  # Include manual edits
            'raw_detections': st.session_state.grid_raw_detections.copy(),
            'image_name': original_name,
            'method': current_model_type,
            'model_type': current_model_type
        }
    
    # Load new image
    new_image_path = st.session_state.grid_image_files[new_index]
    
    try:
        image = Image.open(new_image_path)
        st.session_state.grid_current_image = image
        st.session_state.grid_current_image_path = new_image_path
        st.session_state.grid_current_image_index = new_index
        
        # Restore detections if they exist
        if new_image_path in st.session_state.grid_batch_results:
            result = st.session_state.grid_batch_results[new_image_path]
            st.session_state.grid_detections = result['detections'].copy()
            st.session_state.grid_raw_detections = result.get('raw_detections', []).copy()
            
            # DON'T restore model type from saved results (this was causing switching)
            # Keep current model type selection
            
        else:
            st.session_state.grid_detections = []
            st.session_state.grid_raw_detections = []
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# For grid_analysis_tab.py - Replace the run_all_images() function

def run_all_images():
    """OPTIMIZED - Process all images without Streamlit overhead"""
    
    if not st.session_state.grid_image_files:
        st.warning("No batch images loaded")
        return
    
    selected_model = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    
    # Get processor and parameters
    if selected_model == 'frcnn_grid_3x3':
        processor = st.session_state.get('grid_frcnn_3x3_processor')
        grid_size = 3
        conf_thresh = 0.3
    elif selected_model == 'yolo_grid_3x3':
        processor = st.session_state.get('grid_3x3_processor')
        grid_size = 3
        conf_thresh = 0.25
    elif selected_model == 'yolo_grid_5x5':
        processor = st.session_state.get('grid_5x5_processor')
        grid_size = 5
        conf_thresh = 0.20
    else:
        st.error(f"Unknown model type: {selected_model}")
        return
    
    if not processor:
        st.error(f"{selected_model} model not loaded")
        return
    
    st.info(f"Processing {len(st.session_state.grid_image_files)} images with {selected_model}...")
    
    
    # CREATE LOCAL BATCH RESULTS (not session state during processing)
    local_batch_results = {}
    
    # REMOVE ALL UI UPDATES DURING PROCESSING
    with st.spinner("Processing batch..."):
        try:
            for i, image_file in enumerate(st.session_state.grid_image_files):
                # NO STREAMLIT UPDATES INSIDE LOOP - this was causing 30+ sec overhead per image
                
                try:
                    raw_detections, final_detections = processor.process_image_with_grid(
                        str(image_file),
                        grid_size=grid_size,
                        conf_thresh=conf_thresh
                    )
                    
                    # Get original name
                    if hasattr(st.session_state, 'grid_batch_original_names'):
                        try:
                            index = st.session_state.grid_image_files.index(str(image_file))
                            original_name = st.session_state.grid_batch_original_names[index]
                        except (ValueError, IndexError):
                            original_name = os.path.basename(image_file)
                    else:
                        original_name = os.path.basename(image_file)
                    
                    # STORE IN LOCAL VARIABLE (not session state - avoids serialization overhead)
                    local_batch_results[str(image_file)] = {
                        'detections': final_detections,
                        'raw_detections': raw_detections,
                        'image_name': original_name,
                        'method': selected_model,
                        'model_type': selected_model
                    }
                    
                except Exception as e:
                    # Don't show warnings during batch processing (slows things down)
                    continue
        
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
            return
    
    # SINGLE SESSION STATE UPDATE AT THE END (not during loop)
    st.session_state.grid_batch_results = local_batch_results
    
    total_detections = sum(len(r['detections']) for r in local_batch_results.values())
    st.success(f"Batch processing complete with {selected_model}! {len(local_batch_results)} images, {total_detections} total detections")
    
    # Refresh current image if it was processed
    if st.session_state.grid_current_image_path in local_batch_results:
        result = local_batch_results[st.session_state.grid_current_image_path]
        st.session_state.grid_detections = result['detections'].copy()
        st.session_state.grid_raw_detections = result.get('raw_detections', []).copy()
        
        st.session_state.grid_upload_counter += 1
        st.session_state.grid_folder_counter += 1
        
        st.rerun()

# ALSO UPDATE: run_detection() to add memory cleanup for single image FRCNN
def run_detection():
    """Run grid detection on current image - FIXED with memory cleanup"""
    
    if not st.session_state.grid_current_image_path:
        st.error("No image loaded")
        return
    
    selected_model = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    
    # Get processor and parameters based on model type
    if selected_model == 'frcnn_grid_3x3':
        processor = st.session_state.get('grid_frcnn_3x3_processor')
        grid_size = 3
        conf_thresh = 0.3
        method_name = 'frcnn_grid_3x3'
    elif selected_model == 'yolo_grid_3x3':
        processor = st.session_state.get('grid_3x3_processor')
        grid_size = 3
        conf_thresh = 0.25
        method_name = 'yolo_grid_3x3'
    elif selected_model == 'yolo_grid_5x5':
        processor = st.session_state.get('grid_5x5_processor')
        grid_size = 5
        conf_thresh = 0.20
        method_name = 'yolo_grid_5x5'
    else:
        st.error(f"Unknown model type: {selected_model}")
        return
    
    if not processor:
        st.error(f"{selected_model} model not loaded")
        return
    
    try:
        # MEMORY FIX: Clear cache before FRCNN processing
        if selected_model == 'frcnn_grid_3x3':
            import torch
            import gc
            gc.collect()
            for key in list(st.session_state.keys()):
                if 'tensor' in str(type(st.session_state[key])).lower():
                    del st.session_state[key]
        with st.spinner(f"Running {method_name} detection..."):
            
            raw_detections, final_detections = processor.process_image_with_grid(
                st.session_state.grid_current_image_path,
                grid_size=grid_size,
                conf_thresh=conf_thresh
            )
            
            st.session_state.grid_raw_detections = raw_detections
            st.session_state.grid_detections = final_detections
            
            # Increment upload counter to reset file uploader
            st.session_state.grid_upload_counter += 1
            
            st.success(f"Detection completed! Found {len(final_detections)} leaf tips using {method_name}.")
            st.rerun()
            
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")


def show_common_menus():
    """FIXED common menus with correct grid keys"""
    
    # File Menu
    with st.expander("📁 File Menu", expanded=False):
        col1, col2, col3, col4,col5 = st.columns(5)
        
        with col1:
            upload_key = f"grid_upload_image_{st.session_state.get('grid_upload_counter', 0)}"
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key=upload_key
            )
            if uploaded_file:
                handle_image_upload(uploaded_file)
        
        with col2:
            folder_key = f"grid_upload_folder_{st.session_state.get('grid_folder_counter', 0)}"
            uploaded_folder = st.file_uploader(
                "Upload Folder",
                type=['zip'],
                key=folder_key
            )
            if uploaded_folder:
                handle_folder_upload(uploaded_folder)
        
        with col3:
            if st.button("💾 Save Image", key="grid_save_image"):
                save_current_image()
        
        with col4:
            if st.button("📊 Save Coordinate as CSV", key="grid_save_csv"):
                save_coordinates_csv()
        with col5:
            if st.button("📤 Send Current to Annotation", key="grid_send_current", disabled=not has_current_detections()):
                send_current_image_to_annotation()
        
        
    
    # Reset Menu
    with st.expander("🔄 Reset Menu", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Reset Current Detection", key="grid_reset_current_detection"):
                st.session_state.grid_detections = []
                st.session_state.grid_raw_detections = []
                st.session_state.grid_editing_mode = None
                st.success("Current detections cleared")
                st.rerun()
        
        with col2:
            if st.button("🖼️ Reset Current Image", key="grid_reset_current_image"):
                st.session_state.grid_current_image = None
                st.session_state.grid_current_image_path = None
                st.session_state.grid_detections = []
                st.session_state.grid_raw_detections = []
                st.session_state.grid_editing_mode = None
                st.session_state.grid_show_visualization = False
                st.success("Image cleared")
                st.rerun()
        
        with col3:
            if st.button("📦 Reset Batch Detection", key="grid_reset_batch_detection"):
                st.session_state.grid_batch_results = {}
                st.session_state.grid_detections = []
                st.session_state.grid_raw_detections = []
                st.session_state.grid_editing_mode = None
                st.success("Batch results and current detections cleared")
                st.rerun()
        
        with col4:
            if st.button("🗑️ Reset All Batch", key="grid_reset_all_batch"):
                st.session_state.grid_batch_results = {}
                st.session_state.grid_image_files = []
                st.session_state.grid_current_image_index = 0
                st.session_state.grid_current_image = None
                st.session_state.grid_current_image_path = None
                st.session_state.grid_detections = []
                st.session_state.grid_raw_detections = []
                st.session_state.grid_editing_mode = None
                st.session_state.grid_show_visualization = False
                st.success("All data cleared")
                st.rerun()


def handle_image_upload(uploaded_file):
    """Handle single image upload - EXACT copy of quick detection logic"""
    try:
        # Clear any existing batch data when single image uploaded
        st.session_state.grid_image_files = []  # Changed from grid_batch_images
        st.session_state.grid_batch_results = {}
        st.session_state.grid_current_image_index = 0  # Changed from grid_current_batch_index
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load image
        image = Image.open(temp_path)
        st.session_state.grid_current_image = image
        st.session_state.grid_current_image_path = temp_path
        
        # Clear previous detections
        st.session_state.grid_detections = []
        st.session_state.grid_raw_detections = []  # Additional for grid
        
        st.success(f"Image loaded: {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def has_current_detections():
    """Helper to check if current image has detections"""
    return (st.session_state.grid_current_image is not None and 
            len(st.session_state.grid_detections) > 0)

# 5. FIXED show_stitching_debug_visualization() - Fit to container properly
def show_stitching_debug_visualization():
    """FIXED Show debug visualization - fit to container"""
    
    if not st.session_state.grid_current_image or not st.session_state.grid_raw_detections:
        st.warning("No image or detection data available for visualization")
        return
    
    # Show the debug visualization at the bottom
    st.markdown("---")
    st.markdown("### 🧩 Stitching Process Debug")
    
    # FIXED: Add close button to hide debug
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("❌ Close Debug", key="close_debug_viz"):
            st.session_state.grid_show_visualization = False
            st.rerun()
    
    try:
        # FIXED: Use the debug visualization with proper container sizing
        create_stitching_visualization(
            st.session_state.grid_current_image,
            st.session_state.grid_raw_detections,
            st.session_state.grid_detections,
            st.session_state.grid_selected_size
        )
    except Exception as e:
        st.error(f"Error creating debug visualization: {str(e)}")
        st.info("This feature requires the debug visualization module to be properly configured.")

def save_current_image():
    """Save current image with detections"""
    if not st.session_state.grid_current_image or not st.session_state.grid_detections:
        st.warning("No image or detections to save")
        return
    
    try:
        annotated_image = draw_detections_on_image(
            st.session_state.grid_current_image,
            st.session_state.grid_detections
        )
        
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Show download immediately
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=img_bytes,
            file_name="leaf_tips_detected.png",
            mime="image/png",
            key="download_image_btn_now"
        )
        
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
def send_current_image_to_annotation():
    """Send current image + detections (including manual edits) to annotation"""
    if not st.session_state.grid_current_image or not st.session_state.grid_current_image_path:
        st.warning("No current image to send")
        return
    
    if not st.session_state.grid_detections:
        st.warning("No detections to send. Run detection first or add manual points.")
        return
    
    # Get original filename if available
    if hasattr(st.session_state, 'grid_batch_original_names') and st.session_state.grid_image_files:
        try:
            current_idx = st.session_state.grid_image_files.index(st.session_state.grid_current_image_path)
            original_name = st.session_state.grid_batch_original_names[current_idx]
        except (ValueError, IndexError):
            original_name = 'current_image.png'
    else:
        original_name = 'current_image.png'
    
    # Get current model type
    current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    
    # Prepare data for annotation (includes manual edits)
    annotation_data = {
        'image_path': st.session_state.grid_current_image_path,
        'image': st.session_state.grid_current_image,
        'detections': st.session_state.grid_detections.copy(),  # Includes manual add/remove
        'original_name': original_name,
        'source': 'grid_analysis_tab',
        'method': current_model_type
    }
    
    # Store in session state for annotation page
    st.session_state.annotation_received_image = annotation_data
    st.success(f"✅ Current image sent to annotation tool! ({len(st.session_state.grid_detections)} points including manual edits)")

def send_batch_to_annotation():
    """Send entire batch to annotation (including manual edits from current image)"""
    
    # CRITICAL: Save current image's manual edits first
    if st.session_state.grid_current_image_path and st.session_state.grid_detections:
        current_path = st.session_state.grid_current_image_path
        
        # Get original name
        if hasattr(st.session_state, 'grid_batch_original_names'):
            try:
                index = st.session_state.grid_image_files.index(current_path)
                original_name = st.session_state.grid_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        
        # Save current detections (including manual edits) to batch results
        st.session_state.grid_batch_results[current_path] = {
            'detections': st.session_state.grid_detections.copy(),
            'raw_detections': st.session_state.grid_raw_detections.copy(),
            'image_name': original_name,
            'method': current_model_type,
            'model_type': current_model_type
        }
    
    if not st.session_state.grid_batch_results:
        st.warning("No batch results to send")
        return
    
    # Prepare batch data (includes all manual edits)
    batch_data = {
        'image_files': st.session_state.grid_image_files.copy(),
        'original_names': getattr(st.session_state, 'grid_batch_original_names', []),
        'batch_results': st.session_state.grid_batch_results.copy(),  # Includes manual edits
        'source': 'grid_analysis_tab',
        'method': st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    }
    
    st.session_state.annotation_received_batch = batch_data
    
    # Count total points including manual edits
    total_points = sum(len(r['detections']) for r in st.session_state.grid_batch_results.values())
    st.success(f"✅ Batch sent to annotation tool! ({len(st.session_state.grid_batch_results)} images, {total_points} total points including manual edits)")
def show_batch_navigation():
    """Show batch navigation controls - ADD send batch button"""
    
    if 'grid_current_image_index' not in st.session_state:
        st.session_state.grid_current_image_index = 0
    
    total_images = len(st.session_state.grid_image_files)
    current_idx = min(st.session_state.grid_current_image_index, total_images - 1)
    
    if current_idx != st.session_state.grid_current_image_index:
        st.session_state.grid_current_image_index = current_idx
    
    st.markdown("---")
    st.markdown("### 📂 Batch Navigation")
    
    # Get original filename
    if hasattr(st.session_state, 'grid_batch_original_names') and current_idx < len(st.session_state.grid_batch_original_names):
        current_name = st.session_state.grid_batch_original_names[current_idx]
    else:
        current_name = os.path.basename(st.session_state.grid_image_files[current_idx])
    
    st.markdown(f"**Image {current_idx + 1} of {total_images}:** {current_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)  # ADD col5
    
    with col1:
        if st.button("◀ Previous", disabled=current_idx <= 0, key="grid_batch_prev"):
            navigate_to_image(current_idx - 1)
    
    with col2:
        if st.button("Next ▶", disabled=current_idx >= total_images - 1, key="grid_batch_next"):
            navigate_to_image(current_idx + 1)
    
    with col3:
        if st.button("🚀 Process All", key="grid_batch_process_all"):
            run_all_images()
    
    with col4:
        if st.button("💾 Download Batch", key="grid_batch_download", disabled=not st.session_state.grid_batch_results):
            show_batch_download()
    
    with col5:  # ADD send batch button
        if st.button("📤 Send Batch to Annotation", key="grid_send_batch", disabled=not st.session_state.grid_batch_results):
            send_batch_to_annotation()
def show_image_display_area():
    """Main image display area - FIXED to use grid_model_type"""
    
    if st.session_state.grid_current_image is None:
        st.info("👆 Upload an image to start grid analysis")
        return
    
    # Check if we're in editing mode
    editing_mode = st.session_state.get('grid_editing_mode')
    
    if editing_mode not in ['add', 'remove']:
        # Get grid size from model type
        model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        if 'grid_5x5' in model_type:
            grid_size = 5
        else:
            grid_size = 3
            
        if st.session_state.grid_detections:
            # Show detected image with detections
            result_image = draw_detections_on_image(
                st.session_state.grid_current_image,
                st.session_state.grid_detections
            )
            
            # Add grid overlay if enabled (unique to grid analysis)
            if st.session_state.grid_show_visualization:
                result_image = draw_grid_lines(
                    result_image,
                    grid_size,  # Use extracted grid size
                    show_cell_ids=True
                )
            
            # Apply display size settings from sidebar
            display_size = st.session_state.get('display_size', 'Fit to View')
            
            if display_size == "Small":
                st.image(result_image, caption="Detection Results", width=400)
            elif display_size == "Medium":
                st.image(result_image, caption="Detection Results", width=600)
            elif display_size == "Large":
                st.image(result_image, caption="Detection Results", width=800)
            else:  # Fit to View
                st.image(result_image, caption="Detection Results", use_container_width=True)
        else:
            # Show original image when no detections
            display_image = st.session_state.grid_current_image.copy()
            
            # Add grid overlay if enabled (unique to grid analysis)
            if st.session_state.grid_show_visualization:
                display_image = draw_grid_lines(
                    display_image,
                    grid_size,  # Use extracted grid size
                    show_cell_ids=True
                )
            
            # Apply display size settings
            display_size = st.session_state.get('display_size', 'Fit to View')
            
            if display_size == "Small":
                st.image(display_image, caption="Original Image", width=400)
            elif display_size == "Medium":
                st.image(display_image, caption="Original Image", width=600)
            elif display_size == "Large":
                st.image(display_image, caption="Original Image", width=800)
            else:  # Fit to View
                st.image(display_image, caption="Original Image", use_container_width=True)
    
    # Edit Points interface
    show_edit_points_interface()
    
    # Batch Navigation
    if st.session_state.grid_image_files:
        show_batch_navigation()
    
    # Debug visualization at the bottom (unique to grid analysis)
    if st.session_state.get('grid_show_debug', False) and st.session_state.grid_raw_detections:
        st.markdown("---")
        
        # Get grid size for debug visualization
        model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        if 'grid_5x5' in model_type:
            debug_grid_size = 5
        else:
            debug_grid_size = 3
            
        closed = show_grid_debug_visualization(
            st.session_state.grid_current_image,
            st.session_state.grid_raw_detections,
            st.session_state.grid_detections,
            debug_grid_size,  # Use extracted grid size
            "grid_debug"
        )
        
        if closed == "close":  # FIXED: Check for "close" string
            st.session_state.grid_show_debug = False
            st.rerun()
# 3. FIXED show_grid_model_selection() - Clean, no debug, proper index setting
def show_grid_model_selection():
    """Detection Model Options: Add FRCNN Grid 3x3 option"""
    st.markdown("**Detection Model Options:**")
    
    # Get current selection
    current_selection = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    
    options = ['frcnn_grid_3x3', 'yolo_grid_3x3', 'yolo_grid_5x5']
    labels = ['FRCNN Grid 3x3 (Best)', 'YOLO Grid 3x3', 'YOLO Grid 5x5']
    
    selected = st.radio(
        "Model Type",
        options=options,
        format_func=lambda x: labels[options.index(x)],
        index=options.index(current_selection),
        key="grid_model_selection",
        horizontal=True
    )
    
    st.session_state.grid_model_type = selected

def init_grid_session_state():
    """Initialize session state - FIXED to use only grid_model_type"""
    
    # Tab-specific state isolation
    if 'grid_detections' not in st.session_state:
        st.session_state.grid_detections = []
    if 'grid_raw_detections' not in st.session_state:
        st.session_state.grid_raw_detections = []
    if 'grid_current_image' not in st.session_state:
        st.session_state.grid_current_image = None
    if 'grid_current_image_path' not in st.session_state:
        st.session_state.grid_current_image_path = None
    if 'grid_batch_results' not in st.session_state:
        st.session_state.grid_batch_results = {}
    if 'grid_image_files' not in st.session_state:
        st.session_state.grid_image_files = []
    if 'grid_current_image_index' not in st.session_state:
        st.session_state.grid_current_image_index = 0
    if 'grid_editing_mode' not in st.session_state:
        st.session_state.grid_editing_mode = None
    
    # Model processors
    if 'grid_3x3_processor' not in st.session_state:
        st.session_state.grid_3x3_processor = None
    if 'grid_5x5_processor' not in st.session_state:
        st.session_state.grid_5x5_processor = None
    if 'grid_frcnn_3x3_processor' not in st.session_state:
        st.session_state.grid_frcnn_3x3_processor = None
    
    # UI state - ONLY use grid_model_type
    if 'grid_model_type' not in st.session_state:
        st.session_state.grid_model_type = 'frcnn_grid_3x3'  # Default to best model
    # REMOVE grid_selected_size completely
    
    if 'grid_show_visualization' not in st.session_state:
        st.session_state.grid_show_visualization = False
    if 'grid_show_debug' not in st.session_state:
        st.session_state.grid_show_debug = False
    
    # Upload counters
    if 'grid_upload_counter' not in st.session_state:
        st.session_state.grid_upload_counter = 0
    if 'grid_folder_counter' not in st.session_state:
        st.session_state.grid_folder_counter = 0
    
    # Set default display size
    if 'display_size' not in st.session_state:
        st.session_state.display_size = "Fit to View"
def save_coordinates_csv():
    """Save detection coordinates as CSV - FIXED to use model type"""
    if not st.session_state.grid_detections:
        st.warning("No detections to save")
        return
    
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['x', 'y', 'confidence', 'method', 'model_type', 'cell_row', 'cell_col', 'type'])
        
        # Use CURRENT selected model type
        current_model_type = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
        
        # Write detection data
        for detection in st.session_state.grid_detections:
            cell_info = detection.get('cell', (None, None))
            writer.writerow([
                detection['x'],
                detection['y'],
                detection.get('conf', 1.0),
                current_model_type,  # Use model type
                current_model_type,  # Store model type
                cell_info[0] if cell_info[0] is not None else '',
                cell_info[1] if cell_info[1] is not None else '',
                'manual' if detection.get('manual', False) else 'automatic'
            ])
        
        csv_content = output.getvalue()
        
        st.download_button(
            label="⬇️ Download Grid Coordinates CSV",
            data=csv_content,
            file_name=f"{current_model_type}_coordinates.csv",
            mime="text/csv",
            key="grid_download_csv_btn"
        )
        
    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")
# 7. FIXED show_debug_menu() - Clean version without debug output
def show_debug_menu():
    """Debug Menu - Clean version"""
    
    st.markdown("### 🐛 Debug Menu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Show Grid Lines", key="grid_visualize_btn", help="Display grid lines on the image"):
            if st.session_state.grid_current_image:
                st.session_state.grid_show_visualization = not st.session_state.grid_show_visualization
                st.rerun()
            else:
                st.warning("Please load an image first")
    
    with col2:
        if st.button("🧩 Debug Process", key="grid_debug_process_btn", help="Show step-by-step detection process"):
            if st.session_state.grid_current_image and st.session_state.grid_raw_detections:
                st.session_state.grid_show_debug = not st.session_state.get('grid_show_debug', False)
                st.rerun()
            else:
                st.warning("Please load an image and run detection first")
    
    # Show current state
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.grid_show_visualization:
            st.success("✅ Grid lines ON")
        else:
            st.info("⚪ Grid lines OFF")
    
    with status_col2:
        if st.session_state.get('grid_show_debug', False):
            st.success("✅ Debug mode ON")
        else:
            st.info("⚪ Debug mode OFF")
def run_detection_debug():
    """DEBUG VERSION - Run grid detection with debug info"""
    
    st.write("🐛 DEBUG INFO:")
    st.write(f"Current image: {st.session_state.grid_current_image is not None}")
    st.write(f"Current image path: {st.session_state.grid_current_image_path}")
    
    selected_model = st.session_state.get('grid_model_type', 'frcnn_grid_3x3')
    st.write(f"Selected model: {selected_model}")
    
    if not st.session_state.grid_current_image_path:
        st.error("No image loaded")
        return
    
    # Get processor and parameters based on model type
    if selected_model == 'frcnn_grid_3x3':
        processor = st.session_state.get('grid_frcnn_3x3_processor')
        grid_size = 3
        conf_thresh = 0.3
    elif selected_model == 'yolo_grid_3x3':
        processor = st.session_state.get('grid_3x3_processor')
        grid_size = 3
        conf_thresh = 0.25
    elif selected_model == 'yolo_grid_5x5':
        processor = st.session_state.get('grid_5x5_processor')
        grid_size = 5
        conf_thresh = 0.20
    else:
        st.error(f"Unknown model type: {selected_model}")
        return
    
    st.write(f"🐛 Processor loaded: {processor is not None}")
    st.write(f"🐛 Grid size: {grid_size}")
    st.write(f"🐛 Confidence threshold: {conf_thresh}")
    
    if not processor:
        st.error(f"{selected_model} model not loaded")
        return
    
    try:
        st.write("🐛 About to start detection...")
        with st.spinner(f"Running {selected_model} detection..."):
            
            raw_detections, final_detections = processor.process_image_with_grid(
                st.session_state.grid_current_image_path,
                grid_size=grid_size,
                conf_thresh=conf_thresh
            )
            
            st.write(f"🐛 Detection results: {len(raw_detections)} raw, {len(final_detections)} final")
            
            st.session_state.grid_raw_detections = raw_detections
            st.session_state.grid_detections = final_detections
            
            # Increment upload counter
            st.session_state.grid_upload_counter += 1
            st.write(f"🐛 Upload counter incremented to: {st.session_state.grid_upload_counter}")
            
            st.success(f"Detection completed! Found {len(final_detections)} leaf tips using {selected_model}.")
            st.rerun()
            
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        st.write(f"🐛 Exception details: {e}")