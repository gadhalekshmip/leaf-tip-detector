import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import xml.etree.ElementTree as ET
import numpy as np
import time
import tempfile
import zipfile
import csv
import os
from pathlib import Path

def show_annotation_interface():
    """Main annotation interface"""
    st.title("✏️ Annotation Tool")
    st.markdown("Manual annotation of leaf tip points")
    
    # Initialize session state
    init_annotation_session_state()
    show_received_data_section()
    # Show menus
    show_file_menu()
    show_reset_menu()
    
    # Check for uploaded image
    if st.session_state.annotation_current_image is not None:
        # Display image info
        st.info(f"📷 Image loaded ({st.session_state.annotation_current_image.width}x{st.session_state.annotation_current_image.height})")
        
        # Main layout
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            st.markdown("### 🎛️ Controls")
            
            # Point count
            st.metric("📍 Total Points", len(st.session_state.annotation_points))
            
            # Mode selection
            st.markdown("**Annotation Mode:**")
            
            # Add mode button
            add_button_type = "primary" if st.session_state.annotation_add_mode else "secondary"
            if st.button("➕ Start Adding Points", use_container_width=True, type=add_button_type):
                if not st.session_state.annotation_add_mode:
                    st.session_state.annotation_add_mode = True
                    st.session_state.annotation_remove_mode = False
                    st.rerun()
            
            # Remove mode button  
            remove_button_type = "primary" if st.session_state.annotation_remove_mode else "secondary"
            if st.button("➖ Start Removing Points", use_container_width=True, type=remove_button_type):
                if not st.session_state.annotation_remove_mode:
                    st.session_state.annotation_remove_mode = True
                    st.session_state.annotation_add_mode = False
                    st.rerun()
            
            # Cancel buttons - only show when in a mode
            if st.session_state.annotation_add_mode:
                if st.button("❌ Cancel Adding", use_container_width=True, type="secondary"):
                    st.session_state.annotation_add_mode = False
                    st.rerun()
                st.success("🖱️ **ADD MODE ACTIVE**")
                st.info("Click anywhere on the image to add points. Click 'Cancel Adding' when done.")
                
            elif st.session_state.annotation_remove_mode:
                if st.button("❌ Cancel Removing", use_container_width=True, type="secondary"):
                    st.session_state.annotation_remove_mode = False
                    st.rerun()
                st.warning("🖱️ **REMOVE MODE ACTIVE**") 
                st.info("Click near any point to remove it. Click 'Cancel Removing' when done.")
            else:
                st.info("ℹ️ Select a mode above to start annotating")
            
            st.markdown("---")
            
            # Show points list (simplified)
            if st.session_state.annotation_points:
                st.markdown("### 📋 Point List")
                for i, point in enumerate(st.session_state.annotation_points):
                    st.text(f"Point {i+1}: ({point['x']:.1f}, {point['y']:.1f})")
        
        with col_main:
            st.markdown("### 🖼️ Image Annotation")
            
            # Create annotated image
            annotated_image = draw_points_on_image(st.session_state.annotation_current_image, st.session_state.annotation_points)
            
            # Display image with persistent click handling
            if st.session_state.annotation_add_mode or st.session_state.annotation_remove_mode:
                show_persistent_annotation_interface(annotated_image, st.session_state.annotation_current_image)
            else:
                # Static display
                st.image(annotated_image, use_container_width=True, caption="Annotated Image")
        
        # Batch Navigation (if batch loaded)
        if st.session_state.annotation_image_files:
            show_batch_navigation()
    
    else:
        st.info("👆 Please upload an image to start annotation")

def show_persistent_annotation_interface(annotated_image, original_image):
    """Persistent annotation interface with annotation_ prefix"""
    from streamlit_image_coordinates import streamlit_image_coordinates
    
    # Single coordinates component that stays active
    coordinates = streamlit_image_coordinates(
        annotated_image,
        key="persistent_annotation",  # Fixed key - doesn't change
        width=min(800, original_image.width)
    )
    
    if coordinates is not None:
        # Get current time for duplicate prevention
        current_time = time.time()
        click_x, click_y = coordinates['x'], coordinates['y']
        
        # Create a unique identifier for this click location
        current_coords_id = f"{click_x}_{click_y}"
        
        # Prevent duplicate clicks - check if same location clicked within 1 second
        time_diff = current_time - st.session_state.annotation_last_click_time
        coords_diff = st.session_state.annotation_last_coordinates != current_coords_id
        
        if time_diff > 1.0 or coords_diff:  # Allow click if >1 second OR different location
            # Update last click tracking
            st.session_state.annotation_last_click_time = current_time
            st.session_state.annotation_last_coordinates = current_coords_id
            
            # Scale coordinates to original image size
            scale_factor = original_image.width / min(800, original_image.width)
            actual_x = click_x * scale_factor
            actual_y = click_y * scale_factor
            
            if st.session_state.annotation_add_mode:
                # Add point
                new_point = {
                    'x': float(actual_x),
                    'y': float(actual_y),
                    'id': len(st.session_state.annotation_points) + 1,
                    'manual': False
                }
                st.session_state.annotation_points.append(new_point)
                st.success(f"✅ Added point {len(st.session_state.annotation_points)} at ({actual_x:.1f}, {actual_y:.1f})")
                st.rerun()
                
            elif st.session_state.annotation_remove_mode:
                # Remove nearest point
                nearest_index = find_nearest_point(actual_x, actual_y, st.session_state.annotation_points)
                if nearest_index is not None:
                    removed_point = st.session_state.annotation_points.pop(nearest_index)
                    st.success(f"❌ Removed point at ({removed_point['x']:.1f}, {removed_point['y']:.1f})")
                    st.rerun()
                else:
                    st.warning("⚠️ No point found near click location (try clicking closer to a point)")

def draw_points_on_image(image, points):
    """Draw points on image with different colors - NO CHANGES NEEDED"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for i, point in enumerate(points):
        x, y = int(point['x']), int(point['y'])
        # Different colors based on point type
        if point.get('auto_detected', False):
            color = 'blue'  # Auto-detected from other pages
        elif point.get('manual', False):
            color = 'lime'  # Manually added in annotation
        else:
            color = 'red'   # Default
        # Draw point as circle
        radius = 8
        color = 'lime' if point.get('manual', False) else 'red'
        
        # Draw circle with white border for visibility
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=color, outline='white', width=2)
        
        # Draw point number
        try:
            # Try to use a font
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw number with background for visibility
        text = str(i+1)
        text_x, text_y = x+radius+2, y-radius
        draw.text((text_x, text_y), text, fill='white', font=font, stroke_width=1, stroke_fill='black')
    
    return img_copy

def find_nearest_point(click_x, click_y, points, threshold=20):
    """Find the nearest point to click coordinates - NO CHANGES NEEDED"""
    if not points:
        return None
    
    min_distance = float('inf')
    nearest_index = None
    
    for i, point in enumerate(points):
        distance = np.sqrt((click_x - point['x'])**2 + (click_y - point['y'])**2)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            nearest_index = i
    
    return nearest_index
def draw_points_on_image_clean(image, points):
    """Draw points WITHOUT numbers - FOR DOWNLOAD"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for i, point in enumerate(points):
        x, y = int(point['x']), int(point['y'])
        
        # Draw point as circle
        radius = 8
        color = 'lime' if point.get('manual', False) else 'red'
        
        # Draw circle with white border for visibility
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=color, outline='white', width=2)
        
        # NO NUMBER DRAWING - clean image
    
    return img_copy
def create_csv_export(points):
    """Create CSV export data - REMOVED MANUAL COLUMN"""
    if not points:
        return "x,y,id\n"  # CHANGED: removed 'manual' from header
    
    # Create simplified dataframe with only x, y, id
    simplified_points = []
    for i, point in enumerate(points):
        simplified_points.append({
            'x': point['x'],
            'y': point['y'], 
            'id': i + 1  # Sequential ID
        })
    
    df = pd.DataFrame(simplified_points)
    return df.to_csv(index=False)

def create_xml_export(points):
    """Create XML export in the format you specified - NO CHANGES NEEDED"""
    root = ET.Element("annotations")
    
    for point in points:
        ET.SubElement(root, "box", 
                     x=str(point['x']), 
                     y=str(point['y']),
                     width="10",  # Fixed width as in your example
                     height="10")  # Fixed height as in your example
    
    # Convert to string with proper formatting
    rough_string = ET.tostring(root, encoding='unicode')
    return rough_string

# ALL THE NEW FUNCTIONS YOU NEED TO ADD:

def init_annotation_session_state():
    """Initialize session state for annotation page"""
    if 'annotation_points' not in st.session_state:
        st.session_state.annotation_points = []
    if 'annotation_current_image' not in st.session_state:
        st.session_state.annotation_current_image = None
    if 'annotation_current_image_path' not in st.session_state:
        st.session_state.annotation_current_image_path = None
    if 'annotation_add_mode' not in st.session_state:
        st.session_state.annotation_add_mode = False
    if 'annotation_remove_mode' not in st.session_state:
        st.session_state.annotation_remove_mode = False
    if 'annotation_last_click_time' not in st.session_state:
        st.session_state.annotation_last_click_time = 0
    if 'annotation_last_coordinates' not in st.session_state:
        st.session_state.annotation_last_coordinates = None
    # Upload counters for resetting file uploaders
    if 'annotation_upload_counter' not in st.session_state:
        st.session_state.annotation_upload_counter = 0
    if 'annotation_folder_counter' not in st.session_state:
        st.session_state.annotation_folder_counter = 0
    # Batch functionality
    if 'annotation_image_files' not in st.session_state:
        st.session_state.annotation_image_files = []
    if 'annotation_current_image_index' not in st.session_state:
        st.session_state.annotation_current_image_index = 0
    if 'annotation_batch_results' not in st.session_state:
        st.session_state.annotation_batch_results = {}
    if 'annotation_received_image' not in st.session_state:
        st.session_state.annotation_received_image = None
    if 'annotation_received_batch' not in st.session_state:
        st.session_state.annotation_received_batch = None

def show_file_menu():
    """Complete File Menu for annotation page"""
    with st.expander("📁 File Menu", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            upload_key = f"annotation_upload_image_{st.session_state.get('annotation_upload_counter', 0)}"
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key=upload_key
            )
            if uploaded_file:
                handle_image_upload(uploaded_file)
        
        with col2:
            folder_key = f"annotation_upload_folder_{st.session_state.get('annotation_folder_counter', 0)}"
            uploaded_folder = st.file_uploader(
                "Upload Folder",
                type=['zip'],
                key=folder_key
            )
            if uploaded_folder:
                handle_folder_upload(uploaded_folder)
        
        with col3:
            if st.button("💾 Save Annotated Image", key="annotation_save_image"):
                save_annotated_image()
        
        with col4:
            if st.button("📊 Save CSV", key="annotation_save_csv"):
                save_coordinates_csv()
        
        with col5:
            if st.button("📋 Save XML", key="annotation_save_xml"):
                save_coordinates_xml()

def show_reset_menu():
    """Reset Menu for annotation page"""
    with st.expander("🔄 Reset Menu", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Reset Annotations", key="annotation_reset_points"):
                st.session_state.annotation_points = []
                st.session_state.annotation_add_mode = False
                st.session_state.annotation_remove_mode = False
                st.success("All annotations cleared")
                st.rerun()
        
        with col2:
            if st.button("🖼️ Reset Image", key="annotation_reset_image"):
                st.session_state.annotation_current_image = None
                st.session_state.annotation_current_image_path = None
                st.session_state.annotation_points = []
                st.session_state.annotation_add_mode = False
                st.session_state.annotation_remove_mode = False
                st.success("Image cleared")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset All", key="annotation_reset_all"):
                # Reset everything
                st.session_state.annotation_points = []
                st.session_state.annotation_current_image = None
                st.session_state.annotation_current_image_path = None
                st.session_state.annotation_add_mode = False
                st.session_state.annotation_remove_mode = False
                st.session_state.annotation_image_files = []
                st.session_state.annotation_batch_results = {}
                st.session_state.annotation_upload_counter += 1
                st.session_state.annotation_folder_counter += 1
                st.success("Everything reset")
                st.rerun()
def show_received_data_section():
    """Show data received from other pages"""
    
    received_image = st.session_state.annotation_received_image
    received_batch = st.session_state.annotation_received_batch
    
    if received_image or received_batch:
        st.markdown("### 📨 Received from Detection Pages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if received_image:
                st.info(f"**Single Image**: {received_image['original_name']}")
                st.caption(f"From: {received_image['source']} | Method: {received_image['method']}")
                st.caption(f"Detections: {len(received_image['detections'])}")
                
                if st.button("📥 Load Received Image", key="load_received_image"):
                    load_received_image()
        
        with col2:
            if received_batch:
                batch_count = len(received_batch['batch_results'])
                st.info(f"**Batch**: {batch_count} images")
                st.caption(f"From: {received_batch['source']} | Method: {received_batch['method']}")
                
                if st.button("📥 Load Received Batch", key="load_received_batch"):
                    load_received_batch()
        
        # Clear buttons
        if st.button("🗑️ Clear Received Data", key="clear_received"):
            st.session_state.annotation_received_image = None
            st.session_state.annotation_received_batch = None
            st.success("Received data cleared")
            st.rerun()
        
        st.markdown("---")
def load_received_image():
    """Load single image from detection page"""
    data = st.session_state.annotation_received_image
    
    # Clear existing annotation data
    st.session_state.annotation_current_image = data['image']
    st.session_state.annotation_current_image_path = data['image_path']
    
    # Load detections with different color for auto-detected points
    received_detections = []
    for i, detection in enumerate(data['detections']):
        point = {
            'x': detection['x'],
            'y': detection['y'],
            'id': i + 1,
            'manual': False,
            'auto_detected': True,  # ADD flag for different color
            'source': data['source'],
            'method': data['method']
        }
        received_detections.append(point)
    
    st.session_state.annotation_points = received_detections
    
    # Clear received data
    st.session_state.annotation_received_image = None
    
    st.success(f"✅ Loaded image with {len(received_detections)} auto-detected points")
    st.rerun()

def load_received_batch():
    """Load batch from detection page"""
    data = st.session_state.annotation_received_batch
    
    # Set up batch data
    st.session_state.annotation_image_files = data['image_files'].copy()
    st.session_state.annotation_batch_original_names = data['original_names'].copy()
    st.session_state.annotation_current_image_index = 0
    
    # Convert batch results to annotation format
    annotation_batch_results = {}
    for image_path, result in data['batch_results'].items():
        converted_points = []
        for i, detection in enumerate(result['detections']):
            point = {
                'x': detection['x'],
                'y': detection['y'],
                'id': i + 1,
                'manual': False,
                'auto_detected': True,
                'source': data['source'],
                'method': data['method']
            }
            converted_points.append(point)
        
        annotation_batch_results[image_path] = {
            'points': converted_points,
            'image_name': result['image_name']
        }
    
    st.session_state.annotation_batch_results = annotation_batch_results
    
    # Load first image
    navigate_to_image(0)
    
    # Clear received data
    st.session_state.annotation_received_batch = None
    
    st.success(f"✅ Loaded batch with {len(data['batch_results'])} images")
    st.rerun()
def handle_image_upload(uploaded_file):
    """Handle single image upload"""
    try:
        # Clear any existing batch data
        st.session_state.annotation_image_files = []
        st.session_state.annotation_batch_results = {}
        st.session_state.annotation_current_image_index = 0
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Load image
        image = Image.open(temp_path)
        st.session_state.annotation_current_image = image
        st.session_state.annotation_current_image_path = temp_path
        
        # Clear previous annotations
        st.session_state.annotation_points = []
        st.session_state.annotation_add_mode = False
        st.session_state.annotation_remove_mode = False
        
        st.session_state.annotation_upload_counter += 1
        
        st.success(f"Image loaded: {uploaded_file.name}")
        st.rerun() 
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")


def handle_folder_upload(uploaded_folder):
    """Handle batch folder upload - same pattern as other tabs"""
    st.info("📦 Processing ZIP file...")
    
    try:
        # Clear existing data
        st.session_state.annotation_current_image = None
        st.session_state.annotation_current_image_path = None
        st.session_state.annotation_points = []
        st.session_state.annotation_image_files = []
        st.session_state.annotation_batch_original_names = []
        st.session_state.annotation_batch_results = {}
        st.session_state.annotation_add_mode = False
        st.session_state.annotation_remove_mode = False
        
        with st.spinner("Extracting and loading images..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "upload.zip")
                
                with open(zip_path, "wb") as f:
                    f.write(uploaded_folder.read())
                
                extract_dir = os.path.join(temp_dir, "extracted")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find image files
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
                    # Copy to persistent location
                    persistent_files = []
                    original_names = []
                    
                    for img_file in image_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=img_file.suffix) as tmp:
                            with open(img_file, 'rb') as src:
                                tmp.write(src.read())
                            persistent_files.append(tmp.name)
                            original_names.append(img_file.name)
                    
                    # Store batch data
                    st.session_state.annotation_image_files = persistent_files
                    st.session_state.annotation_batch_original_names = original_names
                    st.session_state.annotation_current_image_index = 0
                    st.session_state.annotation_batch_results = {}
                    
                    # Increment folder counter
                    st.session_state.annotation_folder_counter += 1
                    
                    # Load first image
                    navigate_to_image(0)
                    
                    st.success(f"✅ Successfully loaded {len(image_files)} unique images!")
                    st.rerun()
                else:
                    st.warning("⚠️ No image files found in the ZIP archive.")
                    
    except Exception as e:
        st.error(f"❌ Error processing ZIP file: {str(e)}")

def navigate_to_image(new_index: int):
    """Navigate to specific batch image"""
    total_images = len(st.session_state.annotation_image_files)
    
    if not (0 <= new_index < total_images):
        return
    
    # Save current annotations to batch results
    if st.session_state.annotation_current_image_path:
        current_path = st.session_state.annotation_current_image_path
        
        if hasattr(st.session_state, 'annotation_batch_original_names'):
            try:
                current_idx = st.session_state.annotation_image_files.index(current_path)
                original_name = st.session_state.annotation_batch_original_names[current_idx]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        # Save current annotations
        st.session_state.annotation_batch_results[current_path] = {
            'points': st.session_state.annotation_points.copy(),
            'image_name': original_name
        }
    
    # Load new image
    new_image_path = st.session_state.annotation_image_files[new_index]
    
    try:
        image = Image.open(new_image_path)
        st.session_state.annotation_current_image = image
        st.session_state.annotation_current_image_path = new_image_path
        st.session_state.annotation_current_image_index = new_index
        
        # Restore annotations if they exist
        if new_image_path in st.session_state.annotation_batch_results:
            st.session_state.annotation_points = st.session_state.annotation_batch_results[new_image_path]['points'].copy()
        else:
            st.session_state.annotation_points = []
        
        # Reset modes
        st.session_state.annotation_add_mode = False
        st.session_state.annotation_remove_mode = False
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
def show_batch_navigation():
    """Batch navigation controls"""
    if 'annotation_current_image_index' not in st.session_state:
        st.session_state.annotation_current_image_index = 0
    
    total_images = len(st.session_state.annotation_image_files)
    current_idx = min(st.session_state.annotation_current_image_index, total_images - 1)
    
    if current_idx != st.session_state.annotation_current_image_index:
        st.session_state.annotation_current_image_index = current_idx
    
    st.markdown("---")
    st.markdown("### 📂 Batch Navigation")
    
    # Get original filename
    if hasattr(st.session_state, 'annotation_batch_original_names') and current_idx < len(st.session_state.annotation_batch_original_names):
        current_name = st.session_state.annotation_batch_original_names[current_idx]
    else:
        current_name = os.path.basename(st.session_state.annotation_image_files[current_idx])
    
    st.markdown(f"**Image {current_idx + 1} of {total_images}:** {current_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("◀ Previous", disabled=current_idx <= 0, key="annotation_batch_prev"):
            navigate_to_image(current_idx - 1)
    
    with col2:
        if st.button("Next ▶", disabled=current_idx >= total_images - 1, key="annotation_batch_next"):
            navigate_to_image(current_idx + 1)
    
    with col3:
        if st.button("💾 Download Batch", key="annotation_batch_download", disabled=not st.session_state.annotation_batch_results):
            show_batch_download()

def save_coordinates_xml():
    """Save coordinates as XML"""
    if not st.session_state.annotation_points:
        st.warning("No annotations to save")
        return
    
    try:
        xml_data = create_xml_export(st.session_state.annotation_points)
        
        st.download_button(
            label="⬇️ Download XML",
            data=xml_data,
            file_name="annotations.xml",
            mime="text/xml",
            key="download_coordinates_xml"
        )
        
    except Exception as e:
        st.error(f"Error saving XML: {str(e)}")


def save_annotated_image():
    """Save current annotated image"""
    if not st.session_state.annotation_current_image or not st.session_state.annotation_points:
        st.warning("No image or annotations to save")
        return
    
    try:
        annotated_image = draw_points_on_image_clean(  # CHANGED: use clean version
            st.session_state.annotation_current_image,
            st.session_state.annotation_points
        )
        
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=img_bytes,
            file_name="annotated_image.png",
            mime="image/png",
            key="download_annotated_image"
        )
        
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")

def save_coordinates_csv():
    """Save coordinates as CSV"""
    if not st.session_state.annotation_points:
        st.warning("No annotations to save")
        return
    
    try:
        csv_data = create_csv_export(st.session_state.annotation_points)
        
        st.download_button(
            label="⬇️ Download Coordinates CSV",
            data=csv_data,
            file_name="annotations.csv",
            mime="text/csv",
            key="download_coordinates_csv"
        )
        
    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")

def show_batch_download():
    """Download batch annotations"""
    # Save current image before creating download
    if st.session_state.annotation_current_image_path:
        current_path = st.session_state.annotation_current_image_path
        
        if hasattr(st.session_state, 'annotation_batch_original_names'):
            try:
                index = st.session_state.annotation_image_files.index(current_path)
                original_name = st.session_state.annotation_batch_original_names[index]
            except (ValueError, IndexError):
                original_name = os.path.basename(current_path)
        else:
            original_name = os.path.basename(current_path)
        
        st.session_state.annotation_batch_results[current_path] = {
            'points': st.session_state.annotation_points.copy(),
            'image_name': original_name
        }
    
    if not st.session_state.annotation_batch_results:
        st.warning("No batch annotations to download")
        return
    
    try:
        import zipfile
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Summary CSV
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Image', 'Total_Points'])
            
            for image_path, results in st.session_state.annotation_batch_results.items():
                points = results['points']
                original_name = results['image_name']
                base_name = os.path.splitext(original_name)[0]
                
                # Add to summary
                summary_writer.writerow([original_name, len(points)])
                
                # Individual CSV
                coord_output = io.StringIO()
                coord_writer = csv.writer(coord_output)
                coord_writer.writerow(['x', 'y', 'id', 'manual'])
                
                for point in points:
                    coord_writer.writerow([
                        point['x'], point['y'], 
                        point.get('id', 1), 
                        point.get('manual', False)
                    ])
                
                zip_file.writestr(f"csv/{base_name}_annotations.csv", coord_output.getvalue())
                
                # Individual XML
                xml_data = create_xml_export(points)
                zip_file.writestr(f"xml/{base_name}_annotations.xml", xml_data)
                
                # Annotated image
                try:
                    original_image = Image.open(image_path)
                    annotated_image = draw_points_on_image_clean(original_image, points)  # Use clean version
                    img_buffer = io.BytesIO()
                    annotated_image.save(img_buffer, format='PNG')
                    zip_file.writestr(f"images/{base_name}_annotated.png", img_buffer.getvalue())
                except Exception as e:
                    st.warning(f"Could not create annotated image for {original_name}: {str(e)}")
            
            # Add summary
            zip_file.writestr("batch_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Complete Batch Annotations",
            data=zip_content,
            file_name="annotation_batch_results.zip",
            mime="application/zip",
            key="download_complete_annotation_batch"
        )
        
        st.success(f"Batch download ready with {len(st.session_state.annotation_batch_results)} annotated images")
        
    except Exception as e:
        st.error(f"Error creating batch download: {str(e)}")




