# components/grid_debug_visualizer.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GridDebugVisualizer:
    """Component for visualizing grid detection stitching process"""
    
    def __init__(self):
        self.current_step = 0
        self.total_steps = 4  # Original, Grid Lines, Raw Detections, Final Result
        
    def show_stitching_process(self, 
                            original_image: Image.Image,
                            raw_detections: List[Dict],
                            final_detections: List[Dict],
                            grid_size: int,
                            key_prefix: str = "debug"):
        """Show complete stitching process visualization"""
        
        if not original_image:
            st.warning("No image available for debug visualization")
            return
            
        st.markdown("### 🧩 Grid Detection Process")
        
        # Step controls
        close_result = self._show_step_controls(key_prefix)
        if close_result == "close":
            return "close"
        
        # Show current step
        self._show_current_step(original_image, raw_detections, final_detections, grid_size)
        
        # Step description - NOW PASS grid_size
        self._show_step_description(len(raw_detections), len(final_detections), grid_size)
    def _show_step_controls(self, key_prefix: str):
        """Show navigation controls for steps"""
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=self.current_step <= 0):
                self.current_step = max(0, self.current_step - 1)
                st.rerun()
        
        with col2:
            if st.button("Next ▶", key=f"{key_prefix}_next", disabled=self.current_step >= self.total_steps - 1):
                self.current_step = min(self.total_steps - 1, self.current_step + 1)
                st.rerun()
        
        with col3:
            st.markdown(f"**Step {self.current_step + 1} of {self.total_steps}**")
        
        with col4:
            if st.button("Reset", key=f"{key_prefix}_reset"):
                self.current_step = 0
                st.rerun()
        
        with col5:
            if st.button("❌ Close", key=f"{key_prefix}_close"):
                return "close"
    
    def _show_current_step(self, 
                      original_image: Image.Image,
                      raw_detections: List[Dict],
                      final_detections: List[Dict],
                      grid_size: int):
        """Show the current step visualization"""
        
        if self.current_step == 0:
            # Step 1: Original Image
            self._show_original_image(original_image)
            
        elif self.current_step == 1:
            # Step 2: Original Image with Grid Lines
            self._show_image_with_grid(original_image, grid_size)
            
        elif self.current_step == 2:
            # Step 3: Raw Detections (Orange dots on original image)
            self._show_raw_detections(original_image, raw_detections, grid_size)
            
        elif self.current_step == 3:
            # Step 4: Final Result (Red dots on original image)
            self._show_final_result(original_image, raw_detections, final_detections, grid_size)
            self._show_detailed_reports(raw_detections, final_detections, grid_size)
            # DON'T call _show_step_description here - it's called in show_stitching_process
    
    def _show_original_image(self, image: Image.Image):
        """Show original image - Step 1"""
        st.image(image, caption="Step 1: Original Image", use_container_width=True)
    
    def _show_image_with_grid(self, image: Image.Image, grid_size: int):
        """Show image with grid lines - Step 2"""
        
        # Create image with grid lines
        img_with_grid = image.copy()
        draw = ImageDraw.Draw(img_with_grid)
        
        img_width, img_height = img_with_grid.size
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size
        
        # Draw grid lines
        line_color = (0, 255, 255)  # Cyan
        line_width = max(2, int(min(img_width, img_height) / 500))  # Scale line width
        
        # Vertical lines
        for i in range(1, grid_size):
            x = int(i * cell_width)
            draw.line([(x, 0), (x, img_height)], fill=line_color, width=line_width)
        
        # Horizontal lines
        for i in range(1, grid_size):
            y = int(i * cell_height)
            draw.line([(0, y), (img_width, y)], fill=line_color, width=line_width)
        
        # Add cell IDs
        for row in range(grid_size):
            for col in range(grid_size):
                x = int(col * cell_width + cell_width / 2)
                y = int(row * cell_height + cell_height / 2)
                
                # Draw text with background for visibility
                text = f"({row},{col})"
                bbox = draw.textbbox((0, 0), text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Background rectangle
                draw.rectangle([
                    x - text_width//2 - 5, y - text_height//2 - 2,
                    x + text_width//2 + 5, y + text_height//2 + 2
                ], fill=(0, 0, 0, 128))
                
                # Text
                draw.text((x - text_width//2, y - text_height//2), text, fill=(255, 255, 255))
        
        st.image(img_with_grid, caption=f"Step 2: Grid Layout ({grid_size}x{grid_size})", use_container_width=True)
    
    def _show_raw_detections(self, 
                            image: Image.Image, 
                            raw_detections: List[Dict], 
                            grid_size: int):
        """Show raw detections - Step 3"""
        
        # Create image with raw detections
        img_with_raw = image.copy()
        draw = ImageDraw.Draw(img_with_raw)
        
        # Draw light grid lines first
        self._draw_light_grid(draw, img_with_raw.size, grid_size)
        
        # Draw raw detections as orange dots
        for detection in raw_detections:
            x, y = int(detection['x']), int(detection['y'])
            conf = detection.get('conf', 1.0)
            
            # Point size based on confidence
            point_size = max(3, int(4 + conf * 6))
            
            # Orange dot with white outline
            draw.ellipse([
                x - point_size, y - point_size,
                x + point_size, y + point_size
            ], fill=(255, 165, 0), outline=(255, 255, 255), width=2)  # Orange
            
            # Show cell info if available
            if 'cell' in detection:
                cell = detection['cell']
                if cell and len(cell) >= 2:
                    # Small text showing cell coordinates
                    cell_text = f"({cell[0]},{cell[1]})"
                    draw.text((x + point_size + 2, y - point_size), cell_text, 
                             fill=(255, 255, 0))  # Yellow text
        
        st.image(img_with_raw, 
                caption=f"Step 3: Raw Detections ({len(raw_detections)} points) - Orange dots show all detections before deduplication", 
                use_container_width=True)
    
    def _show_final_result(self, 
                          image: Image.Image,
                          raw_detections: List[Dict],
                          final_detections: List[Dict],
                          grid_size: int):
        """Show final result with comparison - Step 4"""
        
        # Create image with both raw and final detections
        img_final = image.copy()
        draw = ImageDraw.Draw(img_final)
        
        # Draw light grid lines
        self._draw_light_grid(draw, img_final.size, grid_size)
        
        # Draw raw detections as smaller orange dots (background)
        for detection in raw_detections:
            x, y = int(detection['x']), int(detection['y'])
            conf = detection.get('conf', 1.0)
            point_size = max(2, int(2 + conf * 3))  # Smaller
            
            draw.ellipse([
                x - point_size, y - point_size,
                x + point_size, y + point_size
            ], fill=(255, 165, 0), outline=None)  # Orange, no outline
        
        # Draw final detections as larger red dots (foreground)
        for detection in final_detections:
            x, y = int(detection['x']), int(detection['y'])
            conf = detection.get('conf', 1.0)
            point_size = max(4, int(5 + conf * 6))  # Larger
            
            draw.ellipse([
                x - point_size, y - point_size,
                x + point_size, y + point_size
            ], fill=(255, 0, 0), outline=(255, 255, 255), width=3)  # Red with white outline
        
        duplicates_removed = len(raw_detections) - len(final_detections)
        st.image(img_final, 
                caption=f"Step 4: Final Result - {len(final_detections)} unique detections (removed {duplicates_removed} duplicates)", 
                use_container_width=True)
        
        # Show statistics
        self._show_statistics(raw_detections, final_detections)
    
    def _draw_light_grid(self, draw, image_size, grid_size):
        """Draw light grid lines for reference"""
        
        img_width, img_height = image_size
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size
        
        line_color = (0, 255, 255, 100)  # Light cyan
        
        # Vertical lines
        for i in range(1, grid_size):
            x = int(i * cell_width)
            draw.line([(x, 0), (x, img_height)], fill=line_color, width=1)
        
        # Horizontal lines
        for i in range(1, grid_size):
            y = int(i * cell_height)
            draw.line([(0, y), (img_width, y)], fill=line_color, width=1)
    
    def _show_step_description(self, raw_count: int, final_count: int, grid_size: int):
        """Show description for current step"""
        
        descriptions = [
            # Step 1
            f"**Original Image**: This is the input image that will be processed using grid-based detection.",
            
            # Step 2  
            f"**Grid Layout**: The image is divided into a {grid_size}×{grid_size} grid. Each cell will be processed separately with overlapping regions to ensure no leaf tips are missed at grid boundaries.",
            
            # Step 3
            f"**Raw Detections**: The model detected **{raw_count} total points** across all grid cells. Orange dots show all detections before deduplication. Notice that some leaf tips appear multiple times due to overlapping grid cells.",
            
            # Step 4
            f"**Final Result**: After deduplication, **{final_count} unique leaf tips** remain. The stitching process removed **{raw_count - final_count} duplicate detections** by analyzing proximity and confidence scores. Orange dots show original detections, red dots show the final unique detections."
        ]
        
        st.info(descriptions[self.current_step])
    
    def _show_statistics(self, raw_detections: List[Dict], final_detections: List[Dict]):
        """Show processing statistics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Raw Detections", len(raw_detections))
        
        with col2:
            st.metric("Final Detections", len(final_detections))
        
        with col3:
            duplicates = len(raw_detections) - len(final_detections)
            st.metric("Duplicates Removed", duplicates)
        
        with col4:
            if len(raw_detections) > 0:
                efficiency = (len(final_detections) / len(raw_detections)) * 100
                st.metric("Efficiency", f"{efficiency:.1f}%")
    def _show_detailed_reports(self, raw_detections: List[Dict], final_detections: List[Dict], grid_size: int):
        """Show detailed analysis reports"""
        
        st.markdown("#### 📊 Detailed Analysis Reports")
        
        # Processing efficiency report
        with st.expander("🔍 Processing Efficiency Report", expanded=False):
            duplicates_removed = len(raw_detections) - len(final_detections)
            efficiency_percent = (len(final_detections) / len(raw_detections) * 100) if raw_detections else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Detections", len(raw_detections))
            with col2:
                st.metric("Final Detections", len(final_detections))
            with col3:
                st.metric("Efficiency", f"{efficiency_percent:.1f}%")
            
            st.write(f"**Deduplication Process:** Removed {duplicates_removed} duplicate detections")
            st.write(f"**Grid Configuration:** {grid_size}×{grid_size} cells")
        
        # Grid cell distribution report
        with st.expander("📋 Grid Cell Distribution Report", expanded=False):
            cell_counts = {}
            for detection in final_detections:
                cell = detection.get('cell', (None, None))
                if cell and len(cell) >= 2:
                    cell_key = f"({cell[0]},{cell[1]})"
                    cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
            
            if cell_counts:
                st.write("**Detections per Grid Cell:**")
                
                # Create a grid visualization of cell counts
                import pandas as pd
                grid_data = []
                for row in range(grid_size):
                    row_data = []
                    for col in range(grid_size):
                        cell_key = f"({row},{col})"
                        count = cell_counts.get(cell_key, 0)
                        row_data.append(count)
                    grid_data.append(row_data)
                
                df = pd.DataFrame(grid_data, 
                                index=[f"Row {i}" for i in range(grid_size)],
                                columns=[f"Col {i}" for i in range(grid_size)])
                st.dataframe(df, use_container_width=True)
                
                # Show statistics
                non_empty_cells = len([c for c in cell_counts.values() if c > 0])
                avg_per_cell = sum(cell_counts.values()) / len(cell_counts) if cell_counts else 0
                max_cell = max(cell_counts.values()) if cell_counts else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Active Cells", f"{non_empty_cells}/{grid_size*grid_size}")
                with col2:
                    st.metric("Avg per Cell", f"{avg_per_cell:.1f}")
                with col3:
                    st.metric("Max in Cell", max_cell)
        
        # Confidence distribution report
        with st.expander("📈 Confidence Distribution Report", expanded=False):
            confidences = [d.get('conf', 1.0) for d in final_detections if not d.get('manual', False)]
            
            if confidences:
                import numpy as np
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min Confidence", f"{min(confidences):.3f}")
                with col2:
                    st.metric("Max Confidence", f"{max(confidences):.3f}")
                with col3:
                    st.metric("Mean Confidence", f"{np.mean(confidences):.3f}")
                with col4:
                    st.metric("Std Dev", f"{np.std(confidences):.3f}")
                
                # Confidence histogram
                import plotly.express as px
                import pandas as pd
                
                df_conf = pd.DataFrame({'confidence': confidences})
                fig = px.histogram(df_conf, x='confidence', nbins=20, 
                                title="Confidence Score Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No automatic detections with confidence scores found")
# Factory function to create the visualizer
def create_grid_debug_visualizer() -> GridDebugVisualizer:
    """Create a new GridDebugVisualizer instance"""
    return GridDebugVisualizer()

# Main function to be called from grid analysis tab
def show_grid_debug_visualization(image: Image.Image,
                                 raw_detections: List[Dict],
                                 final_detections: List[Dict],
                                 grid_size: int,
                                 key_prefix: str = "grid_debug"):
    """Show the grid debug visualization"""
    
    if 'grid_debug_visualizer' not in st.session_state:
        st.session_state.grid_debug_visualizer = create_grid_debug_visualizer()
    
    visualizer = st.session_state.grid_debug_visualizer
    
    result = visualizer.show_stitching_process(
        image, raw_detections, final_detections, grid_size, key_prefix
    )
    
    return result == "close"