# utils/file_utils.py

import os
import csv
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from PIL import Image
import io

def save_detections_to_csv(detections: List[Dict], 
                          filename: Optional[str] = None) -> str:
    """
    Save detections to CSV format
    
    Args:
        detections: List of detection dictionaries
        filename: Optional filename, if None returns CSV content as string
        
    Returns:
        CSV content as string
    """
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['x', 'y', 'confidence', 'method', 'type', 'cell_row', 'cell_col', 'roi_coords'])
    
    # Data rows
    for detection in detections:
        cell_info = detection.get('cell', (None, None))
        roi_coords = detection.get('roi_coords', None)
        
        writer.writerow([
            detection['x'],
            detection['y'],
            detection.get('conf', 1.0),
            detection.get('method', 'unknown'),
            'manual' if detection.get('manual', False) else 'automatic',
            cell_info[0] if isinstance(cell_info, tuple) else '',
            cell_info[1] if isinstance(cell_info, tuple) else '',
            str(roi_coords) if roi_coords else ''
        ])
    
    csv_content = output.getvalue()
    
    # Save to file if filename provided
    if filename:
        with open(filename, 'w', newline='') as f:
            f.write(csv_content)
    
    return csv_content

def save_image_with_detections(image: Image.Image,
                              detections: List[Dict],
                              filename: Optional[str] = None,
                              format: str = 'PNG') -> bytes:
    """
    Save image with detections marked
    
    Args:
        image: PIL Image
        detections: List of detections
        filename: Optional filename to save to
        format: Image format (PNG, JPEG)
        
    Returns:
        Image bytes
    """
    
    from utils.visualization_utils import draw_detections_on_image
    
    # Create image with detections
    annotated_image = draw_detections_on_image(image, detections)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    annotated_image.save(img_buffer, format=format)
    img_bytes = img_buffer.getvalue()
    
    # Save to file if filename provided
    if filename:
        annotated_image.save(filename, format=format)
    
    return img_bytes

def create_batch_results_zip(batch_results: Dict[str, Dict],
                           include_images: bool = True,
                           include_summary: bool = True) -> bytes:
    """
    Create ZIP file with batch processing results
    
    Args:
        batch_results: Dictionary of batch results
        include_images: Whether to include annotated images
        include_summary: Whether to include summary CSV
        
    Returns:
        ZIP file bytes
    """
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Summary CSV
        if include_summary:
            summary_content = create_batch_summary_csv(batch_results)
            zip_file.writestr("batch_summary.csv", summary_content)
        
        # Individual result files
        for image_path, results in batch_results.items():
            image_name = results.get('image_name', os.path.basename(image_path))
            base_name = os.path.splitext(image_name)[0]
            detections = results.get('detections', [])
            
            # Individual coordinates CSV
            csv_content = save_detections_to_csv(detections)
            zip_file.writestr(f"coordinates/{base_name}_coordinates.csv", csv_content)
            
            # Individual annotated image (if requested)
            if include_images and 'original_image' in results:
                try:
                    img_bytes = save_image_with_detections(
                        results['original_image'], 
                        detections
                    )
                    zip_file.writestr(f"images/{base_name}_detected.png", img_bytes)
                except Exception as e:
                    st.warning(f"Could not save image for {image_name}: {e}")
    
    return zip_buffer.getvalue()

def create_batch_summary_csv(batch_results: Dict[str, Dict]) -> str:
    """
    Create summary CSV for batch results
    
    Args:
        batch_results: Dictionary of batch results
        
    Returns:
        CSV content as string
    """
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Image', 'Total_Tips', 'Auto_Tips', 'Manual_Tips', 
        'Method', 'Model_Info', 'Coordinates_File'
    ])
    
    # Data rows
    for image_path, results in batch_results.items():
        detections = results.get('detections', [])
        image_name = results.get('image_name', os.path.basename(image_path))
        base_name = os.path.splitext(image_name)[0]
        
        total = len(detections)
        manual = sum(1 for d in detections if d.get('manual', False))
        auto = total - manual
        
        method = results.get('method', 'unknown')
        model_info = results.get('model_info', '')
        
        writer.writerow([
            image_name,
            total,
            auto,
            manual,
            method,
            model_info,
            f"coordinates/{base_name}_coordinates.csv"
        ])
    
    return output.getvalue()

def load_image_files(directory: str, 
                    supported_formats: List[str] = None) -> List[str]:
    """
    Load image file paths from directory
    
    Args:
        directory: Directory path
        supported_formats: List of supported file extensions
        
    Returns:
        List of image file paths
    """
    
    if supported_formats is None:
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    image_files = []
    
    for ext in supported_formats:
        pattern = f"*{ext}"
        image_files.extend(Path(directory).glob(pattern))
        # Also check uppercase
        pattern = f"*{ext.upper()}"
        image_files.extend(Path(directory).glob(pattern))
    
    return [str(f) for f in sorted(image_files)]

def extract_zip_file(zip_file_bytes: bytes, 
                    extract_dir: Optional[str] = None) -> str:
    """
    Extract ZIP file and return extraction directory
    
    Args:
        zip_file_bytes: ZIP file bytes
        extract_dir: Directory to extract to (creates temp dir if None)
        
    Returns:
        Path to extraction directory
    """
    
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp()
    
    # Save ZIP bytes to temporary file
    zip_path = os.path.join(extract_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_file_bytes)
    
    # Extract ZIP
    extract_path = os.path.join(extract_dir, "extracted")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Clean up ZIP file
    os.remove(zip_path)
    
    return extract_path

def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def create_download_filename(base_name: str, 
                           suffix: str = "", 
                           extension: str = ".csv") -> str:
    """
    Create standardized download filename
    
    Args:
        base_name: Base filename
        suffix: Optional suffix
        extension: File extension
        
    Returns:
        Formatted filename
    """
    
    # Remove existing extension
    base_name = os.path.splitext(base_name)[0]
    
    # Add suffix if provided
    if suffix:
        filename = f"{base_name}_{suffix}{extension}"
    else:
        filename = f"{base_name}{extension}"
    
    # Replace spaces and special characters
    filename = filename.replace(" ", "_")
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    
    return filename

def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files and directories"""
    
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {e}")

# File format validation
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
SUPPORTED_ARCHIVE_FORMATS = ['.zip']
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB

def validate_uploaded_file(uploaded_file, 
                          expected_type: str = "image") -> bool:
    """
    Validate uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        expected_type: Expected file type ("image" or "archive")
        
    Returns:
        True if valid, False otherwise
    """
    
    if uploaded_file is None:
        return False
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
        return False
    
    # Check file extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if expected_type == "image":
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            st.error(f"Unsupported image format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}")
            return False
    elif expected_type == "archive":
        if file_ext not in SUPPORTED_ARCHIVE_FORMATS:
            st.error(f"Unsupported archive format. Supported: {', '.join(SUPPORTED_ARCHIVE_FORMATS)}")
            return False
    
    return True