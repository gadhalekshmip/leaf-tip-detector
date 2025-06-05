# utils/visualization_utils.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
def draw_detections_on_image(image: Image.Image, 
                           detections: List[Dict],
                           colors: Optional[Dict[str, str]] = None) -> Image.Image:
    """
    Draw detection points on image
    
    Args:
        image: PIL Image
        detections: List of detection dictionaries
        colors: Color mapping for different methods
        
    Returns:
        Image with detections drawn
    """
    if colors is None:
        colors = {
            'manual': 'green',
            'grid': 'red', 
            'roi': 'blue',
            'frcnn': 'purple',
            'yolo': 'red',
            'default': 'red'
        }
    
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
    
    # Draw each detection
    for i, detection in enumerate(detections):
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        method = detection.get('method', 'default')
        is_manual = detection.get('manual', False)
        
        # Determine color
        if is_manual:
            color = colors.get('manual', 'green')
        else:
            # Try to match method with color
            color = colors.get('default', 'red')
            for key in colors:
                if key in method.lower():
                    color = colors[key]
                    break
        
        # Calculate point size based on confidence
        point_size = int(4 + conf * 4)  # 4-8 pixel radius
        
        # Draw circle
        draw.ellipse([
            x - point_size, y - point_size,
            x + point_size, y + point_size
        ], fill=color, outline='white', width=2)
        
       
    
    return img_copy

def draw_grid_lines(image: Image.Image, 
                   grid_size: int,
                   line_color: str = 'cyan',
                   line_width: int = 2,
                   show_cell_ids: bool = True) -> Image.Image:
    """
    Draw grid lines on image
    
    Args:
        image: PIL Image
        grid_size: Grid size (3x3, 5x5, etc.)
        line_color: Color for grid lines
        line_width: Width of grid lines
        show_cell_ids: Whether to show cell ID labels
        
    Returns:
        Image with grid lines drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    img_width, img_height = image.size
    cell_width = img_width / grid_size
    cell_height = img_height / grid_size
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", max(10, int(min(cell_width, cell_height) / 8)))
    except:
        font = ImageFont.load_default()
    
    # Draw vertical lines
    for i in range(1, grid_size):
        x = int(i * cell_width)
        draw.line([(x, 0), (x, img_height)], fill=line_color, width=line_width)
    
    # Draw horizontal lines
    for i in range(1, grid_size):
        y = int(i * cell_height)
        draw.line([(0, y), (img_width, y)], fill=line_color, width=line_width)
    
    # Draw cell IDs if requested
    if show_cell_ids:
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate center of cell
                center_x = int(col * cell_width + cell_width / 2)
                center_y = int(row * cell_height + cell_height / 2)
                
                # Draw cell ID
                cell_id = f"({row},{col})"
                
                # Draw background for better visibility
                bbox = draw.textbbox((center_x, center_y), cell_id, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 128))
                
                # Draw text
                draw.text((center_x - 15, center_y - 8), cell_id, 
                         fill='white', font=font)
    
    return img_copy

def draw_roi_rectangle(image: Image.Image,
                      roi_coords: Tuple[int, int, int, int],
                      color: str = 'cyan',
                      width: int = 3) -> Image.Image:
    """
    Draw ROI rectangle on image
    
    Args:
        image: PIL Image
        roi_coords: (x1, y1, x2, y2) coordinates
        color: Rectangle color
        width: Line width
        
    Returns:
        Image with ROI rectangle drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x1, y1, x2, y2 = roi_coords
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # Add ROI label
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((x1 + 5, y1 + 5), "ROI", fill=color, font=font)
    
    return img_copy

def create_detection_summary_plot(detections: List[Dict]) -> go.Figure:
    """Create summary plot of detections by method"""
    
    if not detections:
        fig = go.Figure()
        fig.add_annotation(text="No detections to display", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count detections by method
    method_counts = {}
    for detection in detections:
        method = detection.get('method', 'unknown')
        if detection.get('manual', False):
            method = 'manual'
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # Create bar chart
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    fig = go.Figure(data=[
        go.Bar(x=methods, y=counts, 
               marker_color=['green' if m == 'manual' else 'blue' for m in methods])
    ])
    
    fig.update_layout(
        title="Detection Count by Method",
        xaxis_title="Detection Method",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_confidence_histogram(detections: List[Dict]) -> go.Figure:
    """Create histogram of detection confidences"""
    
    if not detections:
        fig = go.Figure()
        fig.add_annotation(text="No detections to display", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract confidences
    confidences = [d.get('conf', 0) for d in detections if not d.get('manual', False)]
    
    if not confidences:
        fig = go.Figure()
        fig.add_annotation(text="No automatic detections to display", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
    
    fig.update_layout(
        title="Distribution of Detection Confidences",
        xaxis_title="Confidence",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def create_interactive_image_plot(image: Image.Image, 
                                detections: List[Dict],
                                show_grid: bool = False,
                                grid_size: int = 3,
                                roi_coords: Optional[Tuple[int, int, int, int]] = None) -> go.Figure:
    """
    Create interactive plotly image with detections
    
    Args:
        image: PIL Image
        detections: List of detections
        show_grid: Whether to show grid overlay
        grid_size: Grid size for overlay
        roi_coords: ROI coordinates to highlight
        
    Returns:
        Plotly figure
    """
    
    # Start with base image
    display_image = image.copy()
    
    # Add grid if requested
    if show_grid:
        display_image = draw_grid_lines(display_image, grid_size)
    
    # Add ROI rectangle if provided
    if roi_coords:
        display_image = draw_roi_rectangle(display_image, roi_coords)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add image as background
    fig.add_layout_image(
        dict(
            source=display_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=image.width,
            sizey=image.height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    # Add detection points
    if detections:
        # Separate manual and automatic detections
        manual_points = [d for d in detections if d.get('manual', False)]
        auto_points = [d for d in detections if not d.get('manual', False)]
        
        # Add automatic detections
        if auto_points:
            x_coords = [d['x'] for d in auto_points]
            y_coords = [image.height - d['y'] for d in auto_points]  # Flip Y for plotly
            confidences = [d.get('conf', 1.0) for d in auto_points]
            methods = [d.get('method', 'unknown') for d in auto_points]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=[4 + c*6 for c in confidences],
                    color='red',
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[f"Method: {m}<br>Conf: {c:.3f}" for m, c in zip(methods, confidences)],
                hovertemplate="<b>Auto Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
                customdata=[d['y'] for d in auto_points],
                name="Automatic",
                showlegend=True
            ))
        
        # Add manual points
        if manual_points:
            x_coords = [d['x'] for d in manual_points]
            y_coords = [image.height - d['y'] for d in manual_points]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=10,
                    color='green',
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                text=["Manual point" for _ in manual_points],
                hovertemplate="<b>Manual Point</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
                customdata=[d['y'] for d in manual_points],
                name="Manual",
                showlegend=True
            ))
    
    # Configure layout
    fig.update_layout(
        xaxis=dict(
            range=[0, image.width], 
            showgrid=False, 
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[0, image.height], 
            showgrid=False, 
            zeroline=False,
            scaleanchor="x",
            showticklabels=False
        ),
        width=None,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def create_stitching_visualization_frames(image: Image.Image,
                                        raw_detections: List[Dict],
                                        final_detections: List[Dict],
                                        grid_size: int) -> List[Image.Image]:
    """
    Create frames for stitching process visualization
    
    Args:
        image: Original image
        raw_detections: Raw detections before deduplication
        final_detections: Final detections after deduplication
        grid_size: Grid size used
        
    Returns:
        List of PIL Images showing the stitching process
    """
    frames = []
    
    # Frame 1: Original image with grid
    frame1 = draw_grid_lines(image.copy(), grid_size, show_cell_ids=True)
    frames.append(frame1)
    
    # Frame 2: Raw detections (orange)
    frame2 = draw_grid_lines(image.copy(), grid_size, show_cell_ids=False)
    # Draw raw detections in orange
    draw = ImageDraw.Draw(frame2)
    for detection in raw_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        point_size = int(3 + conf * 3)
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], 
                    fill='orange', outline='white', width=1)
    frames.append(frame2)
    
    # Frame 3: Final detections (red) with raw in background (faded orange)
    frame3 = image.copy()
    draw = ImageDraw.Draw(frame3)
    
    # Draw raw detections faded
    for detection in raw_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        point_size = int(2 + conf * 2)
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], 
                    fill=(255, 165, 0, 128), outline=None)  # Faded orange
    
    # Draw final detections prominent
    for detection in final_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        point_size = int(4 + conf * 4)
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], 
                    fill='red', outline='white', width=2)
    
    frames.append(frame3)
    
    return frames

def display_detection_statistics(detections: List[Dict]) -> None:
    """Display detection statistics in Streamlit"""
    
    if not detections:
        st.info("No detections to analyze")
        return
    
    # Calculate statistics
    total = len(detections)
    manual_count = sum(1 for d in detections if d.get('manual', False))
    auto_count = total - manual_count
    
    # Confidence statistics for automatic detections
    auto_detections = [d for d in detections if not d.get('manual', False)]
    if auto_detections:
        confidences = [d.get('conf', 0) for d in auto_detections]
        avg_conf = np.mean(confidences)
        min_conf = min(confidences) 
        max_conf = max(confidences)
    else:
        avg_conf = min_conf = max_conf = 0
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", total)
    
    with col2:
        st.metric("Automatic", auto_count)
    
    with col3:
        st.metric("Manual", manual_count)
    
    with col4:
        if auto_count > 0:
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    # Method breakdown
    if auto_detections:
        st.write("**Detection Methods:**")
        method_counts = {}
        for d in auto_detections:
            method = d.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for method, count in method_counts.items():
            st.write(f"- {method}: {count} points")

def create_batch_results_summary(batch_results: Dict[str, Dict]) -> go.Figure:
    """Create summary visualization of batch processing results"""
    
    if not batch_results:
        fig = go.Figure()
        fig.add_annotation(text="No batch results to display", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract data
    image_names = []
    detection_counts = []
    methods = []
    
    for image_path, results in batch_results.items():
        image_names.append(os.path.basename(image_path))
        detection_counts.append(len(results.get('detections', [])))
        methods.append(results.get('method', 'unknown'))
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=image_names,
            y=detection_counts,
            text=methods,
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Batch Processing Results - Detection Count per Image",
        xaxis_title="Image",
        yaxis_title="Detection Count",
        xaxis={'tickangle': 45},
        height=400
    )
    
    return fig