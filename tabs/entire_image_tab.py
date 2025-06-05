# tabs/entire_image_tab.py

import streamlit as st
import os
import tempfile
import zipfile
import csv
import io
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional

# Import core modules
from core.yolo_processor import create_yolo_processor
from core.frcnn_processor import create_frcnn_processor
from utils.detection_utils import add_manual_point, remove_point_by_index
from utils.visualization_utils import draw_detections_on_image, create_interactive_image_plot
from config.model_config import get_model_config

from components.improved_point_editor import integrate_point_editor_with_detections

def show_entire_image_interface():
    """Complete Entire Image tab with ALL missing features"""
    
    st.markdown("### 🖼️ Entire Image Analysis")
    st.markdown("Process entire images with YOLO or Faster R-CNN models")
    
    # Initialize session state
    init_entire_image_session_state()
    
    # Show common menus
    show_common_menus()
    
    # Main layout
    col_main, col_right = st.columns([3, 1])
    
    with col_main:
        show_image_display_area()
    
    with col_right:
        show_right_panel()


# Updated show_edit_points_interface() function for tabs/entire_image_tab.py
def show_edit_points_interface():
    """Improved point editing interface using streamlit_image_coordinates"""
    
    st.markdown("#### ✏️ Edit Points")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Point", key="entire_add_point_btn"):
            st.session_state.entire_editing_mode = 'add'
            st.rerun()
    
    with col2:
        if st.button("➖ Remove Point", key="entire_remove_point_btn"):
            if st.session_state.entire_detections:
                st.session_state.entire_editing_mode = 'remove'
                st.rerun()
            else:
                st.warning("No points to remove")
    
    with col3:
        if st.button("🗑️ Clear All Points", key="entire_clear_all_btn"):
            st.session_state.entire_detections = []
            st.session_state.entire_editing_mode = None
            st.success("All points cleared")
            st.rerun()
    
    # Show current editing mode status (only when active)
    current_mode = st.session_state.get('entire_editing_mode')
    if current_mode == 'add':
        st.info("🖱️ **ADD MODE ACTIVE** - Click on the image below to add points")
    elif current_mode == 'remove':
        st.info("🖱️ **REMOVE MODE ACTIVE** - Click near points to remove them")
    
    # Only use the interactive point editor when in editing mode
    if current_mode in ['add', 'remove'] and st.session_state.entire_current_image is not None:
        updated_detections = integrate_point_editor_with_detections(
            st.session_state.entire_current_image,
            st.session_state.entire_detections,
            "entire_"
        )
        
        # Update detections if changed
        if updated_detections != st.session_state.entire_detections:
            st.session_state.entire_detections = updated_detections
    
    # Show statistics
    if st.session_state.entire_detections:
        manual_count = sum(1 for d in st.session_state.entire_detections if d.get('manual', False))
        auto_count = len(st.session_state.entire_detections) - manual_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", len(st.session_state.entire_detections))
        with col2:
            st.metric("Auto Points", auto_count)
        with col3:
            st.metric("Manual Points", manual_count)
    else:
        st.info("No detection points yet. Run detection or add points manually.")
def show_common_menus():
    """FIXED common menus - same as quick_detection.py"""
    
    # File Menu
    with st.expander("📁 File Menu", expanded=False):
        col1, col2, col3, col4,col5 = st.columns(5)
        
        with col1:
            # FIXED: Use dynamic key that changes after detection (like quick_detection)
            upload_key = f"entire_upload_image_{st.session_state.get('entire_upload_counter', 0)}"
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key=upload_key  # FIXED: Use dynamic key
            )
            if uploaded_file:
                handle_image_upload(uploaded_file)
        
        with col2:
            # FIXED: Use dynamic key that changes after processing
            folder_key = f"entire_upload_folder_{st.session_state.get('entire_folder_counter', 0)}"
            uploaded_folder = st.file_uploader(
                "Upload Folder",
                type=['zip'],
                key=folder_key  # FIXED: Use dynamic key
            )
            if uploaded_folder:
                handle_folder_upload(uploaded_folder)
        
        
        with col3:
            if st.button("💾 Save Image", key="entire_save_image"):
                save_current_image()
        
        with col4:
            if st.button("📊 Save Coordinate as CSV", key="entire_save_csv"):
                save_coordinates_csv()
        with col5:
            if st.button("📤 Send Current to Annotation", key="entire_send_current", disabled=not has_current_detections()):
                send_current_image_to_annotation()
        
        
    
    # Reset Menu
    with st.expander("🔄 Reset Menu", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Reset Current Detection", key="entire_reset_current_detection"):
                st.session_state.entire_detections = []
                st.success("Current detections cleared")
                st.rerun()
        
        with col2:
            if st.button("🖼️ Reset Current Image", key="entire_reset_current_image"):
                # FIXED: Actually clear the image
                st.session_state.entire_current_image = None
                st.session_state.entire_current_image_path = None
                st.session_state.entire_detections = []
                st.session_state.entire_editing_mode = None
                st.success("Image cleared")
                st.rerun()
        
        with col3:
            if st.button("📦 Reset Batch Detection", key="entire_reset_batch_detection"):
                st.session_state.entire_batch_results = {}
                # ALSO reset current image detections
                st.session_state.entire_detections = []
                st.success("Batch results and current detections cleared")
                st.rerun()
        
        with col4:
            if st.button("🗑️ Reset All Batch", key="reset_all_batch"):
                st.session_state.entire_batch_results = {}
                st.session_state.entire_image_files = []
                st.session_state.entire_current_image_index = 0
                st.session_state.entire_current_image = None
                st.session_state.entire_current_image_path = None
                st.session_state.entire_detections = []
                st.session_state.entire_editing_mode = None
                st.success("All data cleared")
                st.rerun()


def show_image_display_area():
    """Main image display area with integrated point editing"""
    
    if st.session_state.entire_current_image is None:
        st.info("👆 Upload an image to start analysis")
        return
    
    # Check if we're in editing mode
    editing_mode = st.session_state.get('entire_editing_mode')
    
    if editing_mode not in ['add', 'remove']:
        # Only display static image when NOT in editing mode
        if st.session_state.entire_detections:
            # Import your existing visualization function
            from utils.visualization_utils import draw_detections_on_image
            
            result_image = draw_detections_on_image(
                st.session_state.entire_current_image,
                st.session_state.entire_detections
            )
            
            # Apply display size settings
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
            display_size = st.session_state.get('display_size', 'Fit to View')
            
            if display_size == "Small":
                st.image(st.session_state.entire_current_image, caption="Original Image", width=400)
            elif display_size == "Medium":
                st.image(st.session_state.entire_current_image, caption="Original Image", width=600)
            elif display_size == "Large":
                st.image(st.session_state.entire_current_image, caption="Original Image", width=800)
            else:  # Fit to View
                st.image(st.session_state.entire_current_image, caption="Original Image", use_container_width=True)
    
    # Edit Points interface (this will handle interactive display when in edit mode)
    show_edit_points_interface()
    
    # Batch Navigation (only visible after folder upload)
    if st.session_state.get('entire_image_files'):
        show_batch_navigation()

def has_current_detections():
    """Helper to check if current image has detections"""
    return (st.session_state.entire_current_image is not None and 
            len(st.session_state.entire_detections) > 0)
def show_batch_download():
    """Complete batch download for entire image tab - INCLUDES MANUAL EDITS"""
    
    # CRITICAL FIX: Save current image's manual edits before creating download
    if st.session_state.entire_current_image_path:
        current_path = st.session_state.entire_current_image_path
        
        # Get original filename
        if hasattr(st.session_state, 'entire_batch_original_names'):
            try:
                index = st.session_state.entire_image_files.index(current_path)
                original_name = st.session_state.entire_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # SAVE CURRENT DETECTIONS (including manual add/remove edits)
        st.session_state.entire_batch_results[current_path] = {
            'detections': st.session_state.entire_detections.copy(),  # This includes manual edits!
            'image_name': original_name,
            'method': st.session_state.entire_selected_model
        }
    
    # Check if we have any results to download
    if not st.session_state.entire_batch_results:
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
            summary_writer.writerow(['Image', 'Total_Tips', 'Auto_Tips', 'Manual_Tips', 'Method'])
            
            processed_names = set()  # Track processed files to avoid duplicates
            
            for image_path, results in st.session_state.entire_batch_results.items():
                detections = results['detections']
                
                # Get original name from mapping
                if hasattr(st.session_state, 'entire_batch_original_names'):
                    try:
                        index = st.session_state.entire_image_files.index(image_path)
                        original_name = st.session_state.entire_batch_original_names[index]
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
                
                # Add to summary
                summary_writer.writerow([
                    original_name,
                    total_tips,
                    auto_tips,
                    manual_tips,
                    results.get('method', st.session_state.entire_selected_model)
                ])
                
                # Create annotated image if detections exist
                if detections:
                    try:
                        original_image = Image.open(image_path)
                        annotated_image = draw_detections_on_image(original_image, detections)
                        
                        # Save annotated image to ZIP with original name
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='PNG')
                        zip_file.writestr(f"annotated_images/{base_name}_detected.png", img_buffer.getvalue())
                        
                    except Exception as e:
                        st.warning(f"Could not create annotated image for {original_name}: {str(e)}")
                
                # Individual CSV with coordinates
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'confidence', 'method', 'type'])
                
                for detection in detections:
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('conf', 1.0),
                        detection.get('method', results.get('method', st.session_state.entire_selected_model)),
                        'manual' if detection.get('manual', False) else 'automatic'
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_coordinates.csv", coord_output.getvalue())
            
            # Add summary
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        # Get current model for filename
        current_model = st.session_state.entire_selected_model
        
        st.download_button(
            label="⬇️ Download Complete Batch Results",
            data=zip_content,
            file_name=f"entire_image_{current_model}_batch_results.zip",
            mime="application/zip",
            key="download_complete_batch_entire"
        )
        
        st.success(f"Batch download includes {len(processed_names)} unique images: annotated images + coordinate CSVs + summary")
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")


def show_right_panel():
    """Right panel with detection controls"""
    
    st.markdown("#### 🎛️ Detection Controls")
    
    # Detection Model selection
    show_model_selection()
    
    # Model loading buttons
    show_model_loading()
    
    # Run Detection button
    show_run_detection()
    
    # Tip Count
    show_tip_count()
    
 
def show_model_selection():
    """Detection Model: Radio buttons → Faster R-CNN / YOLO"""
    
    st.markdown("**Detection Model:**")
    
    model_choice = st.radio(
        "Select Model",
        options=['yolo', 'frcnn'],
        format_func=lambda x: 'YOLO' if x == 'yolo' else 'Faster R-CNN',
        key="entire_model_choice",
        horizontal=True
    )
    
    # DON'T CLEAR DETECTIONS HERE - this runs on every rerun!
    st.session_state.entire_selected_model = model_choice

def show_model_loading():
    """Load Model buttons"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load YOLO Model", key="load_yolo"):
            load_yolo_model()
    
    with col2:
        if st.button("📥 Load FRCNN Model", key="load_frcnn"):
            load_frcnn_model()
    
    # Show model status
    show_model_status()

def load_yolo_model():
    """Load YOLO model"""
    try:
        config = get_model_config("yolo_entire")
        if config:
            with st.spinner("Loading YOLO model..."):
                yolo_processor = create_yolo_processor(config)
                
                if yolo_processor.is_loaded():
                    st.session_state.entire_yolo_processor = yolo_processor
                    st.success("✅ YOLO model loaded successfully")
                else:
                    st.error("❌ Failed to load YOLO model")
        else:
            st.error("YOLO model configuration not found")
            
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")

def load_frcnn_model():
    """Load Faster R-CNN model"""
    try:
        config = get_model_config("frcnn")
        if config:
            with st.spinner("Loading Faster R-CNN model..."):
                frcnn_processor = create_frcnn_processor(config)
                
                if frcnn_processor.is_loaded():
                    st.session_state.entire_frcnn_processor = frcnn_processor
                    st.success("✅ Faster R-CNN model loaded successfully")
                else:
                    st.error("❌ Failed to load Faster R-CNN model")
        else:
            st.error("Faster R-CNN model configuration not found")
            
    except Exception as e:
        st.error(f"Error loading Faster R-CNN model: {str(e)}")

def show_model_status():
    """Show model loading status"""
    
    st.markdown("**Model Status:**")
    
    # YOLO status
    if st.session_state.entire_yolo_processor and st.session_state.entire_yolo_processor.is_loaded():
        st.success("✅ YOLO Ready")
    else:
        st.warning("⚪ YOLO Not Loaded")
    
    # FRCNN status
    if st.session_state.entire_frcnn_processor and st.session_state.entire_frcnn_processor.is_loaded():
        st.success("✅ FRCNN Ready")
    else:
        st.warning("⚪ FRCNN Not Loaded")

def show_run_detection():
    """Run Detection button"""
    
    # Check if appropriate model is loaded
    selected_model = st.session_state.entire_selected_model
    model_ready = False
    
    if selected_model == 'yolo':
        model_ready = (st.session_state.entire_yolo_processor and 
                      st.session_state.entire_yolo_processor.is_loaded())
    else:  # frcnn
        model_ready = (st.session_state.entire_frcnn_processor and 
                      st.session_state.entire_frcnn_processor.is_loaded())
    
    detect_disabled = (st.session_state.entire_current_image is None or not model_ready)
    
    if st.button(
        "🚀 Run Detection",
        key="entire_run_detection",
        disabled=detect_disabled,
        use_container_width=True,
        type="primary"
    ):
        run_detection()

def show_tip_count():
    """Dynamic display of detected tips with enhanced info"""
    
    count = len(st.session_state.entire_detections)
    manual_count = sum(1 for d in st.session_state.entire_detections if d.get('manual', False))
    auto_count = count - manual_count
    
    st.markdown("**Tip Count:**")
    st.metric("Total Tips", count)
    
    if count > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Auto", auto_count)
        with col2:
            st.metric("Manual", manual_count)
        
        # Show confidence values if enabled
        if st.session_state.get('show_confidence', True):
            confidences = [d.get('conf', 1.0) for d in st.session_state.entire_detections if not d.get('manual', False)]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                st.caption(f"📊 Avg Confidence: {avg_conf:.2f}")
        
        # Show detection statistics if enabled
        if st.session_state.get('show_statistics', True):
            methods = {}
            for d in st.session_state.entire_detections:
                method = d.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
            
            st.caption("📈 Methods:")
            for method, count in methods.items():
                st.caption(f"  • {method}: {count}")


def run_detection():
    """Run detection on current image"""
    
    if not st.session_state.entire_current_image_path:
        st.error("No image loaded")
        return
    
    selected_model = st.session_state.entire_selected_model
    
    try:
        with st.spinner(f"Running {selected_model.upper()} detection..."):
            
            if selected_model == 'yolo':
                if not st.session_state.entire_yolo_processor:
                    st.error("YOLO model not loaded")
                    return
                detections = st.session_state.entire_yolo_processor.run_inference(
                    st.session_state.entire_current_image_path,
                    conf_thresh=0.25
                )
            else:  # frcnn
                if not st.session_state.entire_frcnn_processor:
                    st.error("FRCNN model not loaded")
                    return
                detections = st.session_state.entire_frcnn_processor.run_inference(
                    st.session_state.entire_current_image_path
                )
            
            st.session_state.entire_detections = detections
            
            # ADD THIS: Reset uploader after detection
            st.session_state.entire_upload_counter += 1
            
            st.success(f"Detection completed! Found {len(detections)} leaf tips.")
            st.rerun()
            
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
def send_current_image_to_annotation():
    """Send current image + detections (including manual edits) to annotation"""
    if not st.session_state.entire_current_image or not st.session_state.entire_current_image_path:
        st.warning("No current image to send")
        return
    
    if not st.session_state.entire_detections:
        st.warning("No detections to send. Run detection first or add manual points.")
        return
    
    # Get original filename if available
    if hasattr(st.session_state, 'entire_batch_original_names') and st.session_state.entire_image_files:
        try:
            current_idx = st.session_state.entire_image_files.index(st.session_state.entire_current_image_path)
            original_name = st.session_state.entire_batch_original_names[current_idx]
        except (ValueError, IndexError):
            original_name = 'current_image.png'
    else:
        original_name = 'current_image.png'
    
    # Prepare data for annotation (includes manual edits)
    annotation_data = {
        'image_path': st.session_state.entire_current_image_path,
        'image': st.session_state.entire_current_image,
        'detections': st.session_state.entire_detections.copy(),  # Includes manual add/remove
        'original_name': original_name,
        'source': 'entire_image_tab',
        'method': st.session_state.entire_selected_model
    }
    
    # Store in session state for annotation page
    st.session_state.annotation_received_image = annotation_data
    st.success(f"✅ Current image sent to annotation tool! ({len(st.session_state.entire_detections)} points including manual edits)")

def send_batch_to_annotation():
    """Send entire batch to annotation (including manual edits from current image)"""
    
    # CRITICAL: Save current image's manual edits first
    if st.session_state.entire_current_image_path and st.session_state.entire_detections:
        current_path = st.session_state.entire_current_image_path
        
        # Get original name
        if hasattr(st.session_state, 'entire_batch_original_names'):
            try:
                index = st.session_state.entire_image_files.index(current_path)
                original_name = st.session_state.entire_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # Save current detections (including manual edits) to batch results
        st.session_state.entire_batch_results[current_path] = {
            'detections': st.session_state.entire_detections.copy(),
            'image_name': original_name,
            'method': st.session_state.entire_selected_model
        }
    
    if not st.session_state.entire_batch_results:
        st.warning("No batch results to send")
        return
    
    # Prepare batch data (includes all manual edits)
    batch_data = {
        'image_files': st.session_state.entire_image_files.copy(),
        'original_names': getattr(st.session_state, 'entire_batch_original_names', []),
        'batch_results': st.session_state.entire_batch_results.copy(),  # Includes manual edits
        'source': 'entire_image_tab',
        'method': st.session_state.entire_selected_model
    }
    
    st.session_state.annotation_received_batch = batch_data
    
    # Count total points including manual edits
    total_points = sum(len(r['detections']) for r in st.session_state.entire_batch_results.values())
    st.success(f"✅ Batch sent to annotation tool! ({len(st.session_state.entire_batch_results)} images, {total_points} total points including manual edits)")
# FIXES for entire_image_tab.py - Replace these functions

def init_entire_image_session_state():
    """FIXED session state initialization with correct variable names"""
    
    # Tab-specific state (isolation)
    if 'entire_detections' not in st.session_state:
        st.session_state.entire_detections = []
    if 'entire_current_image' not in st.session_state:
        st.session_state.entire_current_image = None
    if 'entire_current_image_path' not in st.session_state:
        st.session_state.entire_current_image_path = None
    if 'entire_batch_results' not in st.session_state:
        st.session_state.entire_batch_results = {}
    
    # FIXED: Use consistent variable name (match grid_analysis_tab.py)
    if 'entire_image_files' not in st.session_state:
        st.session_state.entire_image_files = []
    if 'entire_current_image_index' not in st.session_state:  # FIXED: Use index not batch_index
        st.session_state.entire_current_image_index = 0
        
    if 'entire_editing_mode' not in st.session_state:
        st.session_state.entire_editing_mode = None
    
    # Model processors
    if 'entire_yolo_processor' not in st.session_state:
        st.session_state.entire_yolo_processor = None
    if 'entire_frcnn_processor' not in st.session_state:
        st.session_state.entire_frcnn_processor = None
    
    # UI state
    if 'entire_selected_model' not in st.session_state:
        st.session_state.entire_selected_model = 'yolo'
    
    # Upload counters for resetting file uploaders
    if 'entire_upload_counter' not in st.session_state:
        st.session_state.entire_upload_counter = 0
    if 'entire_folder_counter' not in st.session_state:
        st.session_state.entire_folder_counter = 0

def handle_image_upload(uploaded_file):
    """FIXED Handle single image upload - clear batch data properly"""
    try:
        # FIXED: Clear any existing batch data when single image uploaded
        st.session_state.entire_image_files = []  # FIXED: Use entire_image_files not entire_batch_images
        st.session_state.entire_batch_results = {}
        st.session_state.entire_current_image_index = 0  # FIXED: Use index not batch_index
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load image
        image = Image.open(temp_path)
        st.session_state.entire_current_image = image
        st.session_state.entire_current_image_path = temp_path
        
        # Clear previous detections
        st.session_state.entire_detections = []
        
        st.success(f"Image loaded: {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def show_batch_navigation():
    """FIXED Batch Navigation - use correct variable names"""
    
    # FIXED: Use entire_current_image_index consistently
    if 'entire_current_image_index' not in st.session_state:
        st.session_state.entire_current_image_index = 0
    
    total_images = len(st.session_state.entire_image_files)
    current_idx = min(st.session_state.entire_current_image_index, total_images - 1)
    
    # FIXED: Update session state if needed
    if current_idx != st.session_state.entire_current_image_index:
        st.session_state.entire_current_image_index = current_idx
    
    st.markdown("---")
    st.markdown("### 📂 Batch Navigation")
    
    # Get original filename
    if hasattr(st.session_state, 'entire_batch_original_names') and current_idx < len(st.session_state.entire_batch_original_names):
        current_name = st.session_state.entire_batch_original_names[current_idx]
    else:
        current_name = os.path.basename(st.session_state.entire_image_files[current_idx])
    
    st.markdown(f"**Image {current_idx + 1} of {total_images}:** {current_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("◀ Previous", disabled=current_idx <= 0, key="entire_batch_prev"):
            navigate_to_image(current_idx - 1)
    
    with col2:
        if st.button("Next ▶", disabled=current_idx >= total_images - 1, key="entire_batch_next"):
            navigate_to_image(current_idx + 1)
    
    with col3:
        if st.button("🚀 Process All", key="entire_batch_process_all"):
            run_all_images()
    
    with col4:
        if st.button("💾 Download Batch", key="entire_batch_download", disabled=not st.session_state.entire_batch_results):
            show_batch_download()
    
    with col5:
        if st.button("📤 Send Batch to Annotation", key="entire_send_batch", disabled=not st.session_state.entire_batch_results):
            send_batch_to_annotation()

def navigate_to_image(new_index: int):
    """FIXED Navigate to specific batch image - use correct variable names"""
    
    total_images = len(st.session_state.entire_image_files)
    
    if not (0 <= new_index < total_images):
        return
    
    # CRITICAL FIX: Save current detections to batch results ALWAYS (includes manual edits)
    if st.session_state.entire_current_image_path:
        current_path = st.session_state.entire_current_image_path
        
        if hasattr(st.session_state, 'entire_batch_original_names'):
            try:
                current_original_idx = st.session_state.entire_image_files.index(current_path)
                original_name = st.session_state.entire_batch_original_names[current_original_idx]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
            
        # ALWAYS save current detections (including manual edits and empty lists)
        st.session_state.entire_batch_results[current_path] = {
            'detections': st.session_state.entire_detections.copy(),  # Include manual edits
            'image_name': original_name,
            'method': st.session_state.entire_selected_model
        }
    
    # Load new image
    new_image_path = st.session_state.entire_image_files[new_index]
    
    try:
        image = Image.open(new_image_path)
        st.session_state.entire_current_image = image
        st.session_state.entire_current_image_path = new_image_path
        st.session_state.entire_current_image_index = new_index  # FIXED: Update index
        
        # Restore detections if they exist
        if new_image_path in st.session_state.entire_batch_results:
            st.session_state.entire_detections = st.session_state.entire_batch_results[new_image_path]['detections'].copy()
        else:
            st.session_state.entire_detections = []
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

def handle_folder_upload(uploaded_folder):
    """FIXED Handle folder (ZIP) upload - use correct variable names"""
    
    # Show immediate feedback
    st.info("📦 Processing ZIP file...")
    
    try:
        # Clear any existing data completely
        st.session_state.entire_current_image = None
        st.session_state.entire_current_image_path = None
        st.session_state.entire_detections = []
        st.session_state.entire_image_files = []  # FIXED: Use correct variable name
        st.session_state.entire_batch_original_names = []
        st.session_state.entire_batch_results = {}
        
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
                    
                    # FIXED: Store batch data with correct variable names
                    st.session_state.entire_image_files = persistent_files  # FIXED: Use entire_image_files
                    st.session_state.entire_batch_original_names = original_names 
                    st.session_state.entire_current_image_index = 0  # FIXED: Use index not batch_index
                    st.session_state.entire_batch_results = {}
                    
                    # Increment folder counter to reset uploader
                    st.session_state.entire_folder_counter += 1  
                    
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


def run_all_images():
    """Detect tips on all batch images"""
    
    if not st.session_state.entire_image_files:
        st.warning("No batch images loaded")
        return
    
    selected_model = st.session_state.entire_selected_model
    
    # ONLY clear batch results, NOT current detections
    st.session_state.entire_batch_results = {}
    
    # Check if model is loaded
    if selected_model == 'yolo':
        processor = st.session_state.entire_yolo_processor
        conf_thresh = 0.25
    else:  # frcnn
        processor = st.session_state.entire_frcnn_processor
        conf_thresh = None
    
    if not processor or not processor.is_loaded():
        st.error(f"{selected_model.upper()} model not loaded")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, image_file in enumerate(st.session_state.entire_image_files):
            progress = (i + 1) / len(st.session_state.entire_image_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {os.path.basename(image_file)} ({i+1}/{len(st.session_state.entire_image_files)})")
            
            try:
                # Run detection based on selected model
                if selected_model == 'yolo':
                    detections = processor.run_inference(str(image_file), conf_thresh=conf_thresh)
                else:  # frcnn
                    detections = processor.run_inference(str(image_file))
                
                st.session_state.entire_batch_results[str(image_file)] = {
                    'detections': detections,
                    'image_name': os.path.basename(image_file),
                    'method': selected_model  # This will be correct now
                }
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(image_file)}: {str(e)}")
        
        total_detections = sum(len(r['detections']) for r in st.session_state.entire_batch_results.values())
        st.success(f"Batch processing complete! {len(st.session_state.entire_batch_results)} images, {total_detections} total detections")
        
        # Restore current image detections if available
        if st.session_state.entire_current_image_path in st.session_state.entire_batch_results:
            st.session_state.entire_detections = st.session_state.entire_batch_results[st.session_state.entire_current_image_path]['detections'].copy()
            st.rerun()
        
    finally:
        progress_bar.empty()
        status_text.empty()


def has_detections() -> bool:
    """Check if current image has detections"""
    return (st.session_state.entire_current_image is not None and 
            len(st.session_state.entire_detections) > 0)
def save_current_image():
    """Save current image with detections"""
    if not st.session_state.entire_current_image or not st.session_state.entire_detections:
        st.warning("No image or detections to save")
        return
    
    try:
        annotated_image = draw_detections_on_image(
            st.session_state.entire_current_image,
            st.session_state.entire_detections
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

def save_coordinates_csv():
    """Save detection coordinates as CSV - FIXED to use current detections"""
    if not st.session_state.entire_detections:  # Changed from has_detections()
        st.warning("No detections to save")
        return
    
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['x', 'y', 'confidence', 'method', 'type'])
        
        # Use CURRENT detections (includes manual edits)
        for detection in st.session_state.entire_detections:
            writer.writerow([
                detection['x'],
                detection['y'],
                detection.get('conf', 1.0),
                detection.get('method', st.session_state.entire_selected_model),
                'manual' if detection.get('manual', False) else 'automatic'
            ])
        
        csv_content = output.getvalue()
        
        st.download_button(
            label="⬇️ Download Coordinates CSV",
            data=csv_content,
            file_name="leaf_tip_coordinates.csv",
            mime="text/csv",
            key="download_csv_btn"
        )
        
    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")

    """Save detected images and CSVs for entire folder"""
    if not st.session_state.entire_batch_results:
        st.warning("No batch results to save")
        return
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Tips', 'Auto_Tips', 'Manual_Tips', 'Method', 'Coordinates_File'])
            
            for image_path, results in st.session_state.entire_batch_results.items():
                detections = results['detections']
                image_name = results['image_name']
                base_name = os.path.splitext(image_name)[0]
                
                total = len(detections)
                manual = sum(1 for d in detections if d.get('manual', False))
                auto = total - manual
                
                summary_writer.writerow([
                    image_name,
                    total,
                    auto,
                    manual,
                    results.get('method', 'unknown'),
                    f"coordinates/{base_name}_coordinates.csv"
                ])
                
                # Individual coordinates CSV
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'confidence', 'method', 'type'])
                
                for detection in detections:
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('conf', 1.0),
                        detection.get('method', results.get('method', 'unknown')),
                        'manual' if detection.get('manual', False) else 'automatic'
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_coordinates.csv", coord_output.getvalue())
            
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Folder Results ZIP",
            data=zip_content,
            file_name=f"entire_image_{st.session_state.entire_selected_model}_batch_results.zip",
            mime="application/zip",
            key="entire_download_batch"
        )
        
    except Exception as e:
        st.error(f"Error creating folder results: {str(e)}")