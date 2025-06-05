# utils/image_utils.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List
import streamlit as st
import io

def load_image(image_path: str) -> Image.Image:
    """
    Load image from path with proper format handling
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image in RGB format
    """
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, (0, 0), image.convert('RGBA'))
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

def resize_image(image: Image.Image, 
                max_size: int = 1536,
                maintain_aspect: bool = True) -> Tuple[Image.Image, float]:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image
        max_size: Maximum dimension size
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    
    original_width, original_height = image.size
    
    if maintain_aspect:
        # Calculate scale factor
        scale = min(max_size / original_width, max_size / original_height)
        
        if scale >= 1.0:
            # No need to resize
            return image, 1.0
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    else:
        scale = max_size / max(original_width, original_height)
        new_width = new_height = max_size
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image, scale

def crop_image(image: Image.Image, 
               crop_coords: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop image to specified coordinates
    
    Args:
        image: PIL Image
        crop_coords: (x1, y1, x2, y2) coordinates
        
    Returns:
        Cropped PIL Image
    """
    
    x1, y1, x2, y2 = crop_coords
    
    # Validate coordinates
    img_width, img_height = image.size
    
    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
        raise ValueError(f"Crop coordinates {crop_coords} are outside image bounds {image.size}")
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop coordinates: x2 must be > x1 and y2 must be > y1")
    
    return image.crop(crop_coords)

def convert_image_to_bytes(image: Image.Image, 
                          format: str = 'PNG',
                          quality: int = 95) -> bytes:
    """
    Convert PIL Image to bytes
    
    Args:
        image: PIL Image
        format: Output format (PNG, JPEG)
        quality: JPEG quality (1-100)
        
    Returns:
        Image bytes
    """
    
    img_buffer = io.BytesIO()
    
    if format.upper() == 'JPEG':
        # Ensure RGB mode for JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(img_buffer, format=format, quality=quality, optimize=True)
    else:
        image.save(img_buffer, format=format)
    
    return img_buffer.getvalue()

def get_image_info(image: Image.Image) -> dict:
    """
    Get comprehensive image information
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image information
    """
    
    width, height = image.size
    
    return {
        'width': width,
        'height': height,
        'mode': image.mode,
        'format': image.format,
        'size_bytes': len(convert_image_to_bytes(image)),
        'aspect_ratio': width / height,
        'megapixels': (width * height) / 1_000_000
    }

def validate_image_size(image: Image.Image,
                       max_width: int = 4000,
                       max_height: int = 4000,
                       max_megapixels: float = 50.0) -> bool:
    """
    Validate image size constraints
    
    Args:
        image: PIL Image
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        max_megapixels: Maximum megapixels
        
    Returns:
        True if valid, False otherwise
    """
    
    width, height = image.size
    megapixels = (width * height) / 1_000_000
    
    if width > max_width or height > max_height:
        return False
    
    if megapixels > max_megapixels:
        return False
    
    return True

def create_thumbnail(image: Image.Image, 
                    size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """
    Create thumbnail of image
    
    Args:
        image: PIL Image
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail PIL Image
    """
    
    # Create copy to avoid modifying original
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.LANCZOS)
    
    return thumbnail

def add_image_metadata(image: Image.Image, 
                      metadata: dict) -> Image.Image:
    """
    Add metadata to image (for PNG format)
    
    Args:
        image: PIL Image
        metadata: Dictionary of metadata to add
        
    Returns:
        PIL Image with metadata
    """
    
    # Create copy
    img_with_metadata = image.copy()
    
    # Add metadata to info dictionary
    for key, value in metadata.items():
        img_with_metadata.info[key] = str(value)
    
    return img_with_metadata

def extract_image_region(image: Image.Image,
                        center: Tuple[int, int],
                        size: Tuple[int, int]) -> Image.Image:
    """
    Extract region around center point
    
    Args:
        image: PIL Image
        center: (x, y) center coordinates
        size: (width, height) of region to extract
        
    Returns:
        Extracted region as PIL Image
    """
    
    center_x, center_y = center
    width, height = size
    
    # Calculate crop coordinates
    x1 = max(0, center_x - width // 2)
    y1 = max(0, center_y - height // 2)
    x2 = min(image.width, center_x + width // 2)
    y2 = min(image.height, center_y + height // 2)
    
    return crop_image(image, (x1, y1, x2, y2))

def blend_images(image1: Image.Image,
                image2: Image.Image,
                alpha: float = 0.5) -> Image.Image:
    """
    Blend two images together
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        alpha: Blending factor (0.0 to 1.0)
        
    Returns:
        Blended PIL Image
    """
    
    # Ensure both images are the same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.LANCZOS)
    
    # Ensure both images are in the same mode
    if image1.mode != image2.mode:
        image2 = image2.convert(image1.mode)
    
    return Image.blend(image1, image2, alpha)

def apply_image_filter(image: Image.Image, 
                      filter_type: str = 'sharpen') -> Image.Image:
    """
    Apply basic image filters
    
    Args:
        image: PIL Image
        filter_type: Type of filter ('sharpen', 'blur', 'edge')
        
    Returns:
        Filtered PIL Image
    """
    
    from PIL import ImageFilter
    
    filter_map = {
        'sharpen': ImageFilter.SHARPEN,
        'blur': ImageFilter.BLUR,
        'edge': ImageFilter.FIND_EDGES,
        'smooth': ImageFilter.SMOOTH,
        'detail': ImageFilter.DETAIL
    }
    
    if filter_type not in filter_map:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    
    return image.filter(filter_map[filter_type])

def create_image_grid(images: List[Image.Image],
                     grid_size: Tuple[int, int],
                     cell_size: Tuple[int, int] = (200, 200),
                     spacing: int = 5) -> Image.Image:
    """
    Create grid layout of multiple images
    
    Args:
        images: List of PIL Images
        grid_size: (rows, cols) grid layout
        cell_size: (width, height) of each cell
        spacing: Spacing between images in pixels
        
    Returns:
        Grid image as PIL Image
    """
    
    rows, cols = grid_size
    cell_width, cell_height = cell_size
    
    # Calculate total grid dimensions
    total_width = cols * cell_width + (cols - 1) * spacing
    total_height = rows * cell_height + (rows - 1) * spacing
    
    # Create blank grid image
    grid_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Place images in grid
    for i, image in enumerate(images[:rows * cols]):
        row = i // cols
        col = i % cols
        
        # Resize image to cell size
        resized_image = image.resize(cell_size, Image.LANCZOS)
        
        # Calculate position
        x = col * (cell_width + spacing)
        y = row * (cell_height + spacing)
        
        # Paste image
        grid_image.paste(resized_image, (x, y))
    
    return grid_image

def get_dominant_colors(image: Image.Image, 
                       num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Get dominant colors from image
    
    Args:
        image: PIL Image
        num_colors: Number of dominant colors to return
        
    Returns:
        List of RGB color tuples
    """
    
    # Resize image for faster processing
    small_image = image.resize((100, 100), Image.LANCZOS)
    
    # Convert to RGB if needed
    if small_image.mode != 'RGB':
        small_image = small_image.convert('RGB')
    
    # Get color histogram
    colors = small_image.getcolors(10000)  # Get up to 10000 colors
    
    if colors is None:
        # Too many colors, sample pixels
        pixels = list(small_image.getdata())
        from collections import Counter
        color_counts = Counter(pixels)
        colors = [(count, color) for color, count in color_counts.most_common(num_colors)]
    
    # Sort by frequency and return top colors
    colors.sort(reverse=True)
    return [color for count, color in colors[:num_colors]]

def enhance_image_contrast(image: Image.Image, 
                          factor: float = 1.2) -> Image.Image:
    """
    Enhance image contrast
    
    Args:
        image: PIL Image
        factor: Contrast enhancement factor (1.0 = no change)
        
    Returns:
        Enhanced PIL Image
    """
    
    from PIL import ImageEnhance
    
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def enhance_image_brightness(image: Image.Image, 
                           factor: float = 1.1) -> Image.Image:
    """
    Enhance image brightness
    
    Args:
        image: PIL Image
        factor: Brightness enhancement factor (1.0 = no change)
        
    Returns:
        Enhanced PIL Image
    """
    
    from PIL import ImageEnhance
    
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)