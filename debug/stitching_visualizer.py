# debug/stitching_visualizer.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_stitching_visualization(image: Image.Image,
                                 raw_detections: List[Dict],
                                 final_detections: List[Dict],
                                 grid_size: int) -> None:
    """
    Create an interactive step-by-step visualization of the grid stitching process
    
    Args:
        image: Original PIL Image
        raw_detections: Raw detections before deduplication
        final_detections: Final detections after deduplication
        grid_size: Grid size used (3 or 5)
    """
    
    st.markdown("## 🧩 Grid Stitching Process Visualization")
    st.markdown("See how the grid-based detection works step by step:")
    
    # Create step selector
    step = st.selectbox(
        "Select Step to View",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Step 0: Original Image", 
            "Step 1: Grid Overlay",
            "Step 2: Raw Detections (Before Stitching)",
            "Step 3: Final Detections (After Stitching)"
        ][x],
        key="stitching_step"
    )
    
    # Create visualization based on selected step
    if step == 0:
        show_original_image(image)
    elif step == 1:
        show_grid_overlay(image, grid_size)
    elif step == 2:
        show_raw_detections(image, raw_detections, grid_size)
    elif step == 3:
        show_final_comparison(image, raw_detections, final_detections, grid_size)
    
    # Show statistics
    show_stitching_statistics(raw_detections, final_detections, grid_size)

def show_original_image(image: Image.Image):
    """Show the original image without any overlays"""
    
    st.markdown("### 📷 Original Image")
    st.markdown("This is the input image before any processing.")
    
    # Display using plotly for consistency
    fig = create_image_plot(image, title="Original Image")
    st.plotly_chart(fig, use_container_width=True)

def show_grid_overlay(image: Image.Image, grid_size: int):
    """Show the image with grid overlay"""
    
    st.markdown(f"### 📊 Grid Overlay ({grid_size}x{grid_size})")
    st.markdown(f"The image is divided into a {grid_size}x{grid_size} grid for processing. Each cell is processed independently.")
    
    # Create image with grid overlay
    grid_image = draw_grid_overlay(image, grid_size)
    
    # Display with grid information
    fig = create_image_plot(grid_image, title=f"{grid_size}x{grid_size} Grid Overlay")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show grid statistics
    img_width, img_height = image.size
    cell_width = img_width / grid_size
    cell_height = img_height / grid_size
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cells", grid_size * grid_size)
    with col2:
        st.metric("Cell Width", f"{cell_width:.0f}px")
    with col3:
        st.metric("Cell Height", f"{cell_height:.0f}px")

def show_raw_detections(image: Image.Image, raw_detections: List[Dict], grid_size: int):
    """Show raw detections before duplicate removal"""
    
    st.markdown("### 🔍 Raw Detections")
    st.markdown("These are all detections found in each grid cell **before** duplicate removal. "
               "Notice how some leaf tips appear multiple times due to overlapping detections.")
    
    # Create image with raw detections
    raw_image = draw_raw_detections(image, raw_detections, grid_size)
    
    # Display with detection information
    fig = create_detections_plot(raw_image, raw_detections, "Raw Detections (Orange)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detection breakdown by cell
    show_cell_breakdown(raw_detections, grid_size)

def show_final_comparison(image: Image.Image, 
                         raw_detections: List[Dict], 
                         final_detections: List[Dict], 
                         grid_size: int):
    """Show comparison between raw and final detections"""
    
    st.markdown("### ✨ Final Result")
    st.markdown("After applying duplicate removal (stitching), we get the final clean detections. "
               "**Orange dots** show raw detections, **Red dots** show final results.")
    
    # Create comparison image
    comparison_image = draw_comparison(image, raw_detections, final_detections, grid_size)
    
    # Display comparison
    fig = create_comparison_plot(comparison_image, raw_detections, final_detections)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show before/after metrics
    show_before_after_metrics(raw_detections, final_detections)

def draw_grid_overlay(image: Image.Image, grid_size: int) -> Image.Image:
    """Draw grid lines on image"""
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    img_width, img_height = image.size
    cell_width = img_width / grid_size
    cell_height = img_height / grid_size
    
    # Draw grid lines
    for i in range(1, grid_size):
        # Vertical lines
        x = int(i * cell_width)
        draw.line([(x, 0), (x, img_height)], fill='cyan', width=3)
        
        # Horizontal lines
        y = int(i * cell_height)
        draw.line([(0, y), (img_width, y)], fill='cyan', width=3)
    
    # Draw cell labels
    try:
        font = ImageFont.truetype("arial.ttf", max(12, int(min(cell_width, cell_height) / 10)))
    except:
        font = ImageFont.load_default()
    
    for row in range(grid_size):
        for col in range(grid_size):
            center_x = int(col * cell_width + cell_width / 2)
            center_y = int(row * cell_height + cell_height / 2)
            
            cell_id = f"({row},{col})"
            
            # Draw background for text
            bbox = draw.textbbox((center_x, center_y), cell_id, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((center_x - 15, center_y - 8), cell_id, fill='white', font=font)
    
    return img_copy

def draw_raw_detections(image: Image.Image, raw_detections: List[Dict], grid_size: int) -> Image.Image:
    """Draw raw detections on image with grid overlay"""
    
    # Start with grid overlay
    img_copy = draw_grid_overlay(image, grid_size)
    draw = ImageDraw.Draw(img_copy)
    
    # Draw raw detections in orange
    for detection in raw_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        
        # Point size based on confidence
        point_size = int(3 + conf * 4)
        
        # Draw orange dot
        draw.ellipse([
            x - point_size, y - point_size,
            x + point_size, y + point_size
        ], fill='orange', outline='white', width=1)
        
        # Draw cell info if available
        cell_info = detection.get('cell', None)
        if cell_info:
            row, col = cell_info
            draw.text((x + point_size + 2, y - point_size), 
                     f"({row},{col})", fill='yellow', font=None)
    
    return img_copy

def draw_comparison(image: Image.Image, 
                   raw_detections: List[Dict], 
                   final_detections: List[Dict], 
                   grid_size: int) -> Image.Image:
    """Draw comparison between raw and final detections"""
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Draw raw detections (faded orange)
    for detection in raw_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        point_size = int(2 + conf * 3)
        
        draw.ellipse([
            x - point_size, y - point_size,
            x + point_size, y + point_size
        ], fill=(255, 165, 0, 128), outline=None)  # Faded orange
    
    # Draw final detections (prominent red)
    for detection in final_detections:
        x, y = int(detection['x']), int(detection['y'])
        conf = detection.get('conf', 1.0)
        point_size = int(4 + conf * 4)
        
        draw.ellipse([
            x - point_size, y - point_size,
            x + point_size, y + point_size
        ], fill='red', outline='white', width=2)
    
    return img_copy

def create_image_plot(image: Image.Image, title: str = "") -> go.Figure:
    """Create plotly figure for image display"""
    
    fig = go.Figure()
    
    # Add image
    fig.add_layout_image(
        dict(
            source=image,
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
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, image.width], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, image.height], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        width=None,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    return fig

def create_detections_plot(image: Image.Image, detections: List[Dict], title: str = "") -> go.Figure:
    """Create plotly figure with detections overlay"""
    
    fig = create_image_plot(image, title)
    
    if detections:
        # Add detection points
        x_coords = [d['x'] for d in detections]
        y_coords = [image.height - d['y'] for d in detections]  # Flip Y for plotly
        confidences = [d.get('conf', 1.0) for d in detections]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=[4 + c*6 for c in confidences],
                color='orange',
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=[f"Conf: {c:.3f}" for c in confidences],
            hovertemplate="<b>Raw Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
            customdata=[d['y'] for d in detections],
            name="Raw Detections",
            showlegend=True
        ))
    
    return fig

def create_comparison_plot(image: Image.Image, 
                          raw_detections: List[Dict], 
                          final_detections: List[Dict]) -> go.Figure:
    """Create plotly figure comparing raw vs final detections"""
    
    fig = create_image_plot(image, "Raw vs Final Detections Comparison")
    
    # Add raw detections (orange)
    if raw_detections:
        x_coords = [d['x'] for d in raw_detections]
        y_coords = [image.height - d['y'] for d in raw_detections]
        confidences = [d.get('conf', 1.0) for d in raw_detections]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=[3 + c*4 for c in confidences],
                color='orange',
                line=dict(width=1, color='white'),
                opacity=0.6
            ),
            text=[f"Raw - Conf: {c:.3f}" for c in confidences],
            hovertemplate="<b>Raw Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
            customdata=[d['y'] for d in raw_detections],
            name="Raw Detections",
            showlegend=True
        ))
    
    # Add final detections (red)
    if final_detections:
        x_coords = [d['x'] for d in final_detections]
        y_coords = [image.height - d['y'] for d in final_detections]
        confidences = [d.get('conf', 1.0) for d in final_detections]
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=[5 + c*6 for c in confidences],
                color='red',
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=[f"Final - Conf: {c:.3f}" for c in confidences],
            hovertemplate="<b>Final Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
            customdata=[d['y'] for d in final_detections],
            name="Final Detections",
            showlegend=True
        ))
    
    return fig

def show_cell_breakdown(raw_detections: List[Dict], grid_size: int):
    """Show breakdown of detections by grid cell"""
    
    st.markdown("#### 📊 Detection Breakdown by Grid Cell")
    
    # Count detections per cell
    cell_counts = {}
    for detection in raw_detections:
        cell_info = detection.get('cell', None)
        if cell_info:
            row, col = cell_info
            cell_key = f"({row},{col})"
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
    
    if not cell_counts:
        st.info("No cell information available in detections")
        return
    
    # Create grid visualization of counts
    counts_grid = np.zeros((grid_size, grid_size))
    for detection in raw_detections:
        cell_info = detection.get('cell', None)
        if cell_info:
            row, col = cell_info
            if 0 <= row < grid_size and 0 <= col < grid_size:
                counts_grid[row, col] += 1
    
    # Display as heatmap
    fig = go.Figure(data=go.Heatmap(
        z=counts_grid,
        text=counts_grid.astype(int),
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Detections")
    ))
    
    fig.update_layout(
        title=f"Detections per Grid Cell ({grid_size}x{grid_size})",
        xaxis_title="Column",
        yaxis_title="Row",
        width=400,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top cells
    sorted_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown("**Top 5 Cells by Detection Count:**")
    for i, (cell, count) in enumerate(sorted_cells[:5]):
        st.write(f"{i+1}. Cell {cell}: {count} detections")

def show_before_after_metrics(raw_detections: List[Dict], final_detections: List[Dict]):
    """Show before/after metrics comparison"""
    
    st.markdown("#### 📈 Before/After Comparison")
    
    raw_count = len(raw_detections)
    final_count = len(final_detections)
    reduction_count = raw_count - final_count
    reduction_pct = (reduction_count / raw_count * 100) if raw_count > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Raw Detections", raw_count)
    
    with col2:
        st.metric("Final Detections", final_count)
    
    with col3:
        st.metric("Duplicates Removed", reduction_count)
    
    with col4:
        st.metric("Reduction", f"{reduction_pct:.1f}%")
    
    # Efficiency analysis
    if raw_count > 0:
        efficiency = final_count / raw_count
        if efficiency > 0.8:
            efficiency_status = "🟢 Efficient"
            efficiency_msg = "Low duplicate rate - good grid sizing"
        elif efficiency > 0.6:
            efficiency_status = "🟡 Moderate"
            efficiency_msg = "Some overlap detected - normal for grid processing"
        else:
            efficiency_status = "🔴 High Overlap"
            efficiency_msg = "Significant overlap - consider adjusting grid size"
        
        st.info(f"**Processing Efficiency:** {efficiency_status} ({efficiency:.1%})")
        st.caption(efficiency_msg)

def show_stitching_statistics(raw_detections: List[Dict], final_detections: List[Dict], grid_size: int):
    """Show comprehensive stitching statistics"""
    
    st.markdown("---")
    st.markdown("### 📊 Stitching Statistics")
    
    if not raw_detections:
        st.info("No detection data available for statistics")
        return
    
    # Basic stats
    raw_count = len(raw_detections)
    final_count = len(final_detections)
    
    # Confidence analysis
    raw_confidences = [d.get('conf', 1.0) for d in raw_detections]
    final_confidences = [d.get('conf', 1.0) for d in final_detections]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Raw Detection Stats")
        st.write(f"- **Count:** {raw_count}")
        if raw_confidences:
            st.write(f"- **Avg Confidence:** {np.mean(raw_confidences):.3f}")
            st.write(f"- **Min Confidence:** {min(raw_confidences):.3f}")
            st.write(f"- **Max Confidence:** {max(raw_confidences):.3f}")
    
    with col2:
        st.markdown("#### Final Detection Stats")
        st.write(f"- **Count:** {final_count}")
        if final_confidences:
            st.write(f"- **Avg Confidence:** {np.mean(final_confidences):.3f}")
            st.write(f"- **Min Confidence:** {min(final_confidences):.3f}")
            st.write(f"- **Max Confidence:** {max(final_confidences):.3f}")
    
    # Create confidence distribution comparison
    if raw_confidences and final_confidences:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=raw_confidences,
            name="Raw Detections",
            opacity=0.7,
            nbinsx=20,
            marker_color='orange'
        ))
        
        fig.add_trace(go.Histogram(
            x=final_confidences,
            name="Final Detections",
            opacity=0.7,
            nbinsx=20,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Confidence Distribution Comparison",
            xaxis_title="Confidence",
            yaxis_title="Count",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing summary
    st.markdown("#### 🔄 Processing Summary")
    
    processing_time_est = raw_count * 0.1  # Estimate based on detection count
    cells_processed = grid_size * grid_size
    avg_detections_per_cell = raw_count / cells_processed if cells_processed > 0 else 0
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Grid Cells Processed", cells_processed)
    
    with summary_col2:
        st.metric("Avg Detections/Cell", f"{avg_detections_per_cell:.1f}")
    
    with summary_col3:
        st.metric("Est. Processing Time", f"{processing_time_est:.1f}s")
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    reduction_pct = ((raw_count - final_count) / raw_count * 100) if raw_count > 0 else 0
    
    if reduction_pct > 50:
        st.warning("⚠️ **High duplicate rate detected** - Consider using a larger grid size for better efficiency")
    elif reduction_pct > 30:
        st.info("ℹ️ **Moderate overlap** - Normal for overlapping detection patterns")
    elif reduction_pct < 10:
        st.success("✅ **Efficient processing** - Low duplicate rate indicates good grid sizing")
    
    if avg_detections_per_cell > 10:
        st.info("💡 **Tip:** High detection density - Consider increasing confidence threshold if getting false positives")
    elif avg_detections_per_cell < 1:
        st.info("💡 **Tip:** Low detection density - Consider decreasing confidence threshold if missing detections")