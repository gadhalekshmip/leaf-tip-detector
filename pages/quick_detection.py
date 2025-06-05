# pages/quick_detection.py

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
from core.grid_processor import create_grid_processor
from utils.visualization_utils import draw_detections_on_image
from config.model_config import get_best_model


# Update the main interface in pages/quick_detection.py

# Replace show_quick_detection_interface function
def show_quick_detection_interface():
    """Simple & Fast Quick Detection interface - EXACTLY as per documentation"""
    
    st.markdown("## 🌱 Quick Detection")
    st.markdown("Simple & Fast detection using our best performing model (5x5 Grid)")
    
    # Initialize session state
    init_quick_session_state()
    
    # Auto-load best model
    load_best_model()
    
    # Layout: File Menu at top
    show_file_menu()
    
    # Show upload status clearly
    show_upload_status()
    
    # Main layout: Central Screen + Right Panel
    col_main, col_right = st.columns([3, 1])
    
    with col_main:
        show_central_screen()
    
    with col_right:
        show_right_panel()
    
    # Bottom section (empty now since tip count moved)
    show_bottom_section()

# Also update central screen to show better feedback
def show_central_screen():
    """Central Screen - displays current image and detection results"""
    
    if st.session_state.quick_current_image is None:
        if st.session_state.quick_batch_images:
            st.info("📦 Batch loaded! Navigate through images using controls below.")
        else:
            st.info("👆 Upload an image or ZIP folder to start detection")
        return
    
    # Always show detected image if detections exist, otherwise show original
    if st.session_state.quick_detections:
        # Create and display detected image
        result_image = draw_detections_on_image(
            st.session_state.quick_current_image,
            st.session_state.quick_detections
        )
        st.image(result_image, caption=f"🌿 Detected: {len(st.session_state.quick_detections)} leaf tips", use_container_width=True)
        st.success(f"✅ Detection complete! Found {len(st.session_state.quick_detections)} leaf tips.")
    else:
        # Show original image
        st.image(st.session_state.quick_current_image, caption="Original Image - Click 'Detect' to find leaf tips", use_container_width=True)
        st.info("👉 Click the 'Detect' button to analyze this image")
    
    # Show batch navigation if batch images are loaded
    if st.session_state.get('quick_batch_images'):
        show_batch_navigation()
def load_best_model():
    """Auto-load best model (5x5 Grid YOLO)"""
    if st.session_state.quick_processor is None:
        try:
            best_model_key, best_config = get_best_model()
            yolo_processor = create_yolo_processor(best_config)
            
            if yolo_processor.is_loaded():
                grid_processor = create_grid_processor(yolo_processor)
                st.session_state.quick_processor = grid_processor
        except Exception as e:
            st.error(f"Failed to load best model: {str(e)}")


def process_all_batch_images():
    """Process all images in batch"""
    
    if not st.session_state.quick_processor:
        st.error("Model not loaded")
        return
    
    if not st.session_state.quick_batch_images:
        st.error("No batch images loaded")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, image_path in enumerate(st.session_state.quick_batch_images):
            progress = (i + 1) / len(st.session_state.quick_batch_images)
            progress_bar.progress(progress)
            status_text.text(f"Processing {os.path.basename(image_path)} ({i+1}/{len(st.session_state.quick_batch_images)})")
            
            try:
                # Process with 5x5 grid
                raw_detections, final_detections = st.session_state.quick_processor.process_image_with_grid(
                    image_path,
                    grid_size=5,
                    conf_thresh=0.20
                )
                
                # Store results
                st.session_state.quick_batch_results[image_path] = {
                    'detections': final_detections,
                    'image_name': os.path.basename(image_path),
                    'method': 'grid_5x5'
                }
                
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        
        total_detections = sum(len(r['detections']) for r in st.session_state.quick_batch_results.values())
        st.success(f"Batch processing complete! {len(st.session_state.quick_batch_results)} images, {total_detections} total detections")
        
        # Load first processed image
        if st.session_state.quick_batch_images:
            navigate_batch_image(0)
        
    finally:
        progress_bar.empty()
        status_text.empty()


# Replace handle_folder_upload function to avoid duplication

# Replace show_batch_download to use original names
def show_batch_download():
    """Show batch download options with original filenames"""
    
    if not st.session_state.quick_batch_results:
        st.warning("No batch results to download")
        return
    
    try:
        # Create ZIP with annotated images and CSVs using original names
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Tips', 'Method'])
            
            processed_names = set()  # Track processed files to avoid duplicates
            
            for image_path, results in st.session_state.quick_batch_results.items():
                detections = results['detections']
                
                # Get original name from mapping
                if hasattr(st.session_state, 'quick_batch_original_names'):
                    try:
                        index = st.session_state.quick_batch_images.index(image_path)
                        original_name = st.session_state.quick_batch_original_names[index]
                    except (ValueError, IndexError):
                        original_name = results.get('image_name', os.path.basename(image_path))
                else:
                    original_name = results.get('image_name', os.path.basename(image_path))
                
                # Skip if already processed (avoid duplicates)
                if original_name in processed_names:
                    continue
                processed_names.add(original_name)
                
                base_name = os.path.splitext(original_name)[0]
                
                # Add to summary
                summary_writer.writerow([
                    original_name,
                    len(detections),
                    results.get('method', 'grid_5x5')
                ])
                
                # Create annotated image
                try:
                    original_image = Image.open(image_path)
                    annotated_image = draw_detections_on_image(original_image, detections)
                    
                    # Save annotated image to ZIP with original name
                    img_buffer = io.BytesIO()
                    annotated_image.save(img_buffer, format='PNG')
                    zip_file.writestr(f"annotated_images/{base_name}_detected.png", img_buffer.getvalue())
                    
                except Exception as e:
                    st.warning(f"Could not create annotated image for {original_name}: {str(e)}")
                
                # Individual CSV with original name
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'method'])  # Removed confidence column
                
                for detection in detections:
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('method', 'grid_5x5')
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_coordinates.csv", coord_output.getvalue())
            
            # Add summary
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Complete Batch Results",
            data=zip_content,
            file_name="leaf_tip_batch_results.zip",
            mime="application/zip",
            key="download_complete_batch"
        )
        
        st.success(f"Batch download includes {len(processed_names)} unique images: annotated images + coordinate CSVs + summary")
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")


def show_right_panel():
    """Right Side Panel as per documentation"""
    
    st.markdown("#### 🎛️ Controls")
    
    # Tip Count - Move here from bottom
    count = len(st.session_state.quick_detections)
    st.markdown("**Tip Count:**")
    st.metric("🌿 Total Tips", count)
    
    st.markdown("---")
    
    
    
    # Detect Button
    detect_disabled = st.session_state.quick_current_image is None
    
    if st.button(
        "🚀 Detect",
        key="quick_detect_btn",
        disabled=detect_disabled,
        use_container_width=True,
        type="primary"
    ):
        run_detection()
    
    # Model Status
    st.markdown("---")
    st.markdown("**Model Status:**")
    if st.session_state.quick_processor:
        st.success("✅ 5x5 Grid Ready")
    else:
        st.error("❌ Model Not Loaded")

def show_bottom_section():
    """Bottom Section - now empty since tip count moved to right panel"""
    pass

def handle_image_upload(uploaded_file):
    """Handle single image upload"""
    try:
        # Clear any existing batch data when single image uploaded
        st.session_state.quick_batch_images = []
        st.session_state.quick_batch_results = {}
        st.session_state.quick_current_batch_index = 0
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load image
        image = Image.open(temp_path)
        st.session_state.quick_current_image = image
        st.session_state.quick_current_image_path = temp_path
        
        # Clear previous detections
        st.session_state.quick_detections = []
        
        st.success(f"Image loaded: {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
# Fix for pages/quick_detection.py

# Replace show_file_menu function to handle the upload display issue
def show_file_menu():
    """File Menu as per documentation"""
    
    st.markdown("### 📁 File Menu")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Upload Image - use a key that changes after detection
        upload_key = f"quick_upload_image_{st.session_state.get('upload_counter', 0)}"
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key=upload_key
        )
        if uploaded_file:
            handle_image_upload(uploaded_file)
    
    with col2:
        # Upload Folder - use a key that changes after processing
        folder_key = f"quick_upload_folder_{st.session_state.get('folder_counter', 0)}"
        uploaded_folder = st.file_uploader(
            "Upload Folder",
            type=['zip'],
            key=folder_key
        )
        if uploaded_folder:
            handle_folder_upload(uploaded_folder)
    
    with col3:
        # Save Image
        if st.button("💾 Save Image", disabled=not has_detections()):
            save_current_image()
    
    with col4:
        # Save Points
        if st.button("📊 Save Points", disabled=not has_detections()):
            save_points_csv()
    
    with col5:
        if st.button("📤 Send Current to Annotation", disabled=not has_current_image_with_detections()):
            send_current_image_to_annotation()

# Initialize counters in session state
def init_quick_session_state():
    """Initialize session state for quick mode"""
    if 'quick_detections' not in st.session_state:
        st.session_state.quick_detections = []
    if 'quick_current_image' not in st.session_state:
        st.session_state.quick_current_image = None
    if 'quick_current_image_path' not in st.session_state:
        st.session_state.quick_current_image_path = None
    if 'quick_processor' not in st.session_state:
        st.session_state.quick_processor = None
    if 'quick_batch_results' not in st.session_state:
        st.session_state.quick_batch_results = {}
    if 'quick_zoom_level' not in st.session_state:
        st.session_state.quick_zoom_level = 1.0
    # Add batch-specific state
    if 'quick_batch_images' not in st.session_state:
        st.session_state.quick_batch_images = []
    if 'quick_current_batch_index' not in st.session_state:
        st.session_state.quick_current_batch_index = 0
    # Add upload counters to reset file uploaders
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0
    if 'folder_counter' not in st.session_state:
        st.session_state.folder_counter = 0
def has_current_image_with_detections():
    """Helper to check if current image has detections"""
    return (st.session_state.quick_current_image is not None and 
            len(st.session_state.quick_detections) > 0)
# Replace run_detection to clear uploader after detection
def run_detection():
    """Run detection on current image"""
    if not st.session_state.quick_processor:
        st.error("Model not loaded")
        return
    
    if not st.session_state.quick_current_image_path:
        st.error("No image loaded")
        return
    
    try:
        with st.spinner("Running detection..."):
            # Process with 5x5 grid (best model)
            raw_detections, final_detections = st.session_state.quick_processor.process_image_with_grid(
                st.session_state.quick_current_image_path,
                grid_size=5,
                conf_thresh=0.20
            )
            
            st.session_state.quick_detections = final_detections
            
            # Increment upload counter to reset file uploader
            st.session_state.upload_counter += 1
            
            st.success(f"Detection completed! Found {len(final_detections)} leaf tips.")
            st.rerun()
            
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
# Fix for pages/quick_detection.py

# Replace handle_folder_upload function to prevent duplication
def handle_folder_upload(uploaded_folder):
    """Handle folder (ZIP) upload with proper status feedback"""
    
    # Show immediate feedback
    st.info("📦 Processing ZIP file...")
    
    try:
        # Clear any existing data completely
        st.session_state.quick_current_image = None
        st.session_state.quick_current_image_path = None
        st.session_state.quick_detections = []
        st.session_state.quick_batch_images = []  # Clear existing batch
        st.session_state.quick_batch_original_names = []  # Clear existing names
        st.session_state.quick_batch_results = {}
        
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
                    st.session_state.quick_batch_images = persistent_files
                    st.session_state.quick_batch_original_names = original_names
                    st.session_state.quick_current_batch_index = 0
                    st.session_state.quick_batch_results = {}
                    
                    # Increment folder counter to reset uploader
                    st.session_state.folder_counter += 1
                    
                    # Load first image
                    navigate_batch_image(0)
                    
                    st.success(f"✅ Successfully loaded {len(image_files)} unique images!")
                    st.info("👇 Use the navigation controls below the image to browse through your batch.")
                    st.rerun()
                    
                else:
                    st.warning("⚠️ No image files found in the ZIP archive.")
                    
    except Exception as e:
        st.error(f"❌ Error processing ZIP file: {str(e)}")
        st.info("Please make sure your ZIP file contains valid image files (.png, .jpg, .jpeg, .bmp, .tiff)")

def show_batch_navigation():
    """Show batch navigation controls with send batch button"""
    
    if 'quick_current_batch_index' not in st.session_state:
        st.session_state.quick_current_batch_index = 0
    
    total_images = len(st.session_state.quick_batch_images)
    current_idx = min(st.session_state.quick_current_batch_index, total_images - 1)
    
    if current_idx != st.session_state.quick_current_batch_index:
        st.session_state.quick_current_batch_index = current_idx
    
    st.markdown("---")
    st.markdown("### 📂 Batch Navigation")
    
    # Get original filename
    if hasattr(st.session_state, 'quick_batch_original_names') and current_idx < len(st.session_state.quick_batch_original_names):
        current_name = st.session_state.quick_batch_original_names[current_idx]
    else:
        current_name = os.path.basename(st.session_state.quick_batch_images[current_idx])
    
    st.markdown(f"**Image {current_idx + 1} of {total_images}:** {current_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)  # ADD col5
    
    with col1:
        if st.button("◀ Previous", disabled=current_idx <= 0, key="batch_prev"):
            navigate_batch_image(current_idx - 1)
    
    with col2:
        if st.button("Next ▶", disabled=current_idx >= total_images - 1, key="batch_next"):
            navigate_batch_image(current_idx + 1)
    
    with col3:
        if st.button("🚀 Process All", key="batch_process_all"):
            process_all_batch_images()
    
    with col4:
        if st.button("💾 Download Batch", key="batch_download", disabled=not st.session_state.quick_batch_results):
            show_batch_download()
    
    with col5:  # ADD new column for send to annotation
        if st.button("📤 Send Batch to Annotation", key="send_batch_to_annotation", disabled=not st.session_state.quick_batch_results):
            send_batch_to_annotation()
# Fix navigate_batch_image to handle bounds properly
def navigate_batch_image(new_index: int):
    """Navigate to specific batch image with bounds checking"""
    
    total_images = len(st.session_state.quick_batch_images)
    
    # Ensure index is within bounds
    if not (0 <= new_index < total_images):
        return
    
    # Save current detections to batch results
    if st.session_state.quick_current_image_path and st.session_state.quick_detections:
        current_path = st.session_state.quick_current_image_path
        # Get original name for this image
        if hasattr(st.session_state, 'quick_batch_original_names'):
            try:
                current_original_idx = st.session_state.quick_batch_images.index(current_path)
                original_name = st.session_state.quick_batch_original_names[current_original_idx]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
            
        st.session_state.quick_batch_results[current_path] = {
            'detections': st.session_state.quick_detections.copy(),
            'image_name': original_name,
            'method': 'grid_5x5'
        }
    
    # Load new image
    new_image_path = st.session_state.quick_batch_images[new_index]
    
    try:
        # Load image
        image = Image.open(new_image_path)
        st.session_state.quick_current_image = image
        st.session_state.quick_current_image_path = new_image_path
        st.session_state.quick_current_batch_index = new_index
        
        # Restore detections if they exist
        if new_image_path in st.session_state.quick_batch_results:
            st.session_state.quick_detections = st.session_state.quick_batch_results[new_image_path]['detections'].copy()
        else:
            st.session_state.quick_detections = []
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# Add a clear status display function
def show_upload_status():
    """Show current upload status"""
    
    if st.session_state.quick_current_image:
        st.success("📷 Single image loaded")
    elif st.session_state.quick_batch_images:
        st.success(f"📦 Batch loaded: {len(st.session_state.quick_batch_images)} images")
    else:
        st.info("📁 No images loaded - use the file menu above")
import os
def process_batch_images(image_files):
    """Process batch of images"""
    if not st.session_state.quick_processor:
        st.error("Model not loaded")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_results = {}
    
    try:
        for i, image_file in enumerate(image_files):
            progress = (i + 1) / len(image_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {image_file.name} ({i+1}/{len(image_files)})")
            
            try:
                raw_detections, final_detections = st.session_state.quick_processor.process_image_with_grid(
                    str(image_file),
                    grid_size=5,
                    conf_thresh=0.20
                )
                
                batch_results[str(image_file)] = {
                    'detections': final_detections,
                    'image_name': image_file.name,
                    'method': 'grid_5x5'
                }
                
            except Exception as e:
                st.warning(f"Error processing {image_file.name}: {str(e)}")
        
        st.session_state.quick_batch_results = batch_results
        
        # Show results
        total_detections = sum(len(r['detections']) for r in batch_results.values())
        st.success(f"Batch processing complete! {len(batch_results)} images, {total_detections} total detections")
        
    finally:
        progress_bar.empty()
        status_text.empty()
def send_current_image_to_annotation():
    """Send current image + detections to annotation"""
    if not st.session_state.quick_current_image or not st.session_state.quick_current_image_path:
        st.warning("No current image to send")
        return
    
    # Prepare data for annotation
    annotation_data = {
        'image_path': st.session_state.quick_current_image_path,
        'image': st.session_state.quick_current_image,
        'detections': st.session_state.quick_detections.copy(),
        'original_name': getattr(st.session_state, 'quick_current_image_name', 'current_image.png'),
        'source': 'quick_detection',
        'method': 'grid_5x5'
    }
    
    # Store in session state for annotation page
    st.session_state.annotation_received_image = annotation_data
    st.success("✅ Current image sent to annotation tool!")

def send_batch_to_annotation():
    """Send entire batch to annotation"""
    if not st.session_state.quick_batch_results:
        st.warning("No batch results to send")
        return
    
    # Prepare batch data
    batch_data = {
        'image_files': st.session_state.quick_batch_images.copy(),
        'original_names': getattr(st.session_state, 'quick_batch_original_names', []),
        'batch_results': st.session_state.quick_batch_results.copy(),
        'source': 'quick_detection',
        'method': 'grid_5x5'
    }
    
    st.session_state.annotation_received_batch = batch_data
    st.success(f"✅ Batch with {len(st.session_state.quick_batch_results)} images sent to annotation tool!")
def save_current_image():
    """Save current image with detections"""
    if not st.session_state.quick_current_image or not st.session_state.quick_detections:
        st.warning("No image or detections to save")
        return
    
    try:
        annotated_image = draw_detections_on_image(
            st.session_state.quick_current_image,
            st.session_state.quick_detections
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
def save_points_csv():
    """Save detection coordinates as CSV"""
    if not has_detections():
        return
    
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['x', 'y', 'confidence', 'method'])
        
        for detection in st.session_state.quick_detections:
            writer.writerow([
                detection['x'],
                detection['y'],
                detection.get('conf', 1.0),
                detection.get('method', 'grid_5x5')
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

def save_folder_results():
    """Save folder processing results as ZIP"""
    if not st.session_state.quick_batch_results:
        return
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Tips', 'Method'])
            
            for image_path, results in st.session_state.quick_batch_results.items():
                detections = results['detections']
                
                summary_writer.writerow([
                    results['image_name'],
                    len(detections),
                    results.get('method', 'grid_5x5')
                ])
                
                # Individual CSV
                base_name = os.path.splitext(results['image_name'])[0]
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'confidence', 'method'])
                
                for detection in detections:
                    coord_writer.writerow([
                        detection['x'],
                        detection['y'],
                        detection.get('conf', 1.0),
                        detection.get('method', 'grid_5x5')
                    ])
                
                zip_file.writestr(f"coordinates/{base_name}_coordinates.csv", coord_output.getvalue())
            
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Batch Results ZIP",
            data=zip_content,
            file_name="leaf_tip_batch_results.zip",
            mime="application/zip",
            key="download_batch_zip_btn"
        )
        
    except Exception as e:
        st.error(f"Error creating batch ZIP: {str(e)}")

def has_detections() -> bool:
    """Check if current image has detections"""
    return (st.session_state.quick_current_image is not None and 
            len(st.session_state.quick_detections) > 0)