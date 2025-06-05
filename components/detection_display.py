# components/detection_display.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from typing import List, Dict, Optional, Tuple
import numpy as np

class DetectionDisplay:
    """Component for displaying detection results with various visualization options"""
    
    def __init__(self):
        self.current_image = None
        self.detections = []
        self.display_options = {
            'show_confidence': True,
            'show_grid': False,
            'show_cell_ids': False,
            'point_size_multiplier': 1.0,
            'color_scheme': 'method'
        }
    
    def display_detections_overlay(self, 
                                 image: Image.Image,
                                 detections: List[Dict],
                                 key: str = "detection_display",
                                 height: int = 600,
                                 show_controls: bool = True) -> Dict:
        """
        Display image with detection overlay and controls
        
        Returns:
            Dict with interaction results (clicked points, etc.)
        """
        
        self.current_image = image
        self.detections = detections
        
        # Display controls if requested
        if show_controls:
            self._display_controls(key)
        
        # Create interactive plot
        fig = self._create_detection_plot()
        
        # Display plot
        plot_result = st.plotly_chart(
            fig, 
            use_container_width=True,
            key=f"{key}_plot"
        )
        
        # Display statistics
        if self.display_options.get('show_statistics', True):
            self._display_detection_statistics()
        
        return {"plot_result": plot_result}
    
    def _display_controls(self, key: str):
        """Display detection visualization controls"""
        
        with st.expander("🎨 Display Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self.display_options['show_confidence'] = st.checkbox(
                    "Show Confidence Values",
                    value=self.display_options['show_confidence'],
                    key=f"{key}_show_conf"
                )
                
                self.display_options['show_grid'] = st.checkbox(
                    "Show Grid Lines",
                    value=self.display_options['show_grid'],
                    key=f"{key}_show_grid"
                )
            
            with col2:
                self.display_options['show_cell_ids'] = st.checkbox(
                    "Show Cell IDs",
                    value=self.display_options['show_cell_ids'],
                    key=f"{key}_show_cells"
                )
                
                self.display_options['point_size_multiplier'] = st.slider(
                    "Point Size",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"{key}_point_size"
                )
            
            with col3:
                self.display_options['color_scheme'] = st.selectbox(
                    "Color Scheme",
                    options=['method', 'confidence', 'single'],
                    index=0,
                    key=f"{key}_color_scheme"
                )
    
    def _create_detection_plot(self) -> go.Figure:
        """Create plotly figure with detections"""
        
        fig = go.Figure()
        
        # Add image background
        fig.add_layout_image(
            dict(
                source=self.current_image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=self.current_image.width,
                sizey=self.current_image.height,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        
        # Add grid overlay if enabled
        if self.display_options['show_grid']:
            self._add_grid_overlay(fig)
        
        # Group detections by type for better visualization
        detection_groups = self._group_detections_by_type()
        
        # Add detection points for each group
        for group_name, group_detections in detection_groups.items():
            if group_detections:
                self._add_detection_group_to_plot(fig, group_name, group_detections)
        
        # Configure layout
        self._configure_plot_layout(fig)
        
        return fig
    
    def _group_detections_by_type(self) -> Dict[str, List[Dict]]:
        """Group detections by type for visualization"""
        
        groups = {
            'manual': [],
            'grid_3x3': [],
            'grid_5x5': [],
            'frcnn': [],
            'yolo': [],
            'roi': [],
            'other': []
        }
        
        for detection in self.detections:
            if detection.get('manual', False):
                groups['manual'].append(detection)
            elif 'grid' in detection.get('method', ''):
                if '3x3' in detection.get('method', '') or detection.get('grid_size') == 3:
                    groups['grid_3x3'].append(detection)
                elif '5x5' in detection.get('method', '') or detection.get('grid_size') == 5:
                    groups['grid_5x5'].append(detection)
                else:
                    groups['other'].append(detection)
            elif 'frcnn' in detection.get('method', ''):
                groups['frcnn'].append(detection)
            elif 'yolo' in detection.get('method', ''):
                groups['yolo'].append(detection)
            elif 'roi' in detection.get('method', ''):
                groups['roi'].append(detection)
            else:
                groups['other'].append(detection)
        
        return groups
    
    def _add_detection_group_to_plot(self, fig: go.Figure, group_name: str, detections: List[Dict]):
        """Add a group of detections to the plot"""
        
        if not detections:
            return
        
        # Extract coordinates
        x_coords = [d['x'] for d in detections]
        y_coords = [self.current_image.height - d['y'] for d in detections]  # Flip Y
        confidences = [d.get('conf', 1.0) for d in detections]
        
        # Determine colors and sizes
        colors, sizes = self._get_detection_colors_and_sizes(group_name, detections, confidences)
        
        # Create hover text
        hover_text = []
        for i, detection in enumerate(detections):
            text = f"<b>{group_name.replace('_', ' ').title()}</b><br>"
            text += f"X: {detection['x']:.1f}<br>"
            text += f"Y: {detection['y']:.1f}<br>"
            text += f"Confidence: {confidences[i]:.3f}<br>"
            text += f"Method: {detection.get('method', 'unknown')}"
            
            if 'cell' in detection:
                cell = detection['cell']
                text += f"<br>Cell: ({cell[0]}, {cell[1]})"
            
            hover_text.append(text)
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name=group_name.replace('_', ' ').title(),
            showlegend=True
        ))
    
    def _get_detection_colors_and_sizes(self, group_name: str, detections: List[Dict], confidences: List[float]) -> Tuple[List, List]:
        """Get colors and sizes for detection points"""
        
        color_scheme = self.display_options['color_scheme']
        size_multiplier = self.display_options['point_size_multiplier']
        
        # Define color maps
        method_colors = {
            'manual': 'green',
            'grid_3x3': 'blue',
            'grid_5x5': 'red',
            'frcnn': 'purple',
            'yolo': 'orange',
            'roi': 'cyan',
            'other': 'gray'
        }
        
        # Calculate sizes based on confidence
        base_sizes = [4 + conf * 6 for conf in confidences]
        sizes = [size * size_multiplier for size in base_sizes]
        
        # Determine colors
        if color_scheme == 'method':
            colors = [method_colors.get(group_name, 'gray')] * len(detections)
        elif color_scheme == 'confidence':
            colors = confidences  # Will use colorscale
        else:  # single color
            colors = ['red'] * len(detections)
        
        return colors, sizes
    
    def _add_grid_overlay(self, fig: go.Figure):
        """Add grid overlay to the plot"""
        
        # Try to determine grid size from detections
        grid_size = self._infer_grid_size()
        
        if grid_size > 0:
            img_width = self.current_image.width
            img_height = self.current_image.height
            
            cell_width = img_width / grid_size
            cell_height = img_height / grid_size
            
            # Add vertical lines
            for i in range(1, grid_size):
                x = i * cell_width
                fig.add_vline(
                    x=x,
                    line_color="cyan",
                    line_width=2,
                    opacity=0.7
                )
            
            # Add horizontal lines
            for i in range(1, grid_size):
                y = self.current_image.height - (i * cell_height)  # Flip Y
                fig.add_hline(
                    y=y,
                    line_color="cyan",
                    line_width=2,
                    opacity=0.7
                )
            
            # Add cell IDs if enabled
            if self.display_options['show_cell_ids']:
                self._add_cell_id_annotations(fig, grid_size, cell_width, cell_height)
    
    def _infer_grid_size(self) -> int:
        """Infer grid size from detection data"""
        
        for detection in self.detections:
            method = detection.get('method', '')
            if '3x3' in method:
                return 3
            elif '5x5' in method:
                return 5
            elif 'cell' in detection:
                # Try to infer from cell coordinates
                cell = detection['cell']
                if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                    max_cell = max(cell[0], cell[1])
                    if max_cell < 3:
                        return 3
                    elif max_cell < 5:
                        return 5
        
        return 0  # No grid detected
    
    def _add_cell_id_annotations(self, fig: go.Figure, grid_size: int, cell_width: float, cell_height: float):
        """Add cell ID annotations to grid"""
        
        for row in range(grid_size):
            for col in range(grid_size):
                x = col * cell_width + cell_width / 2
                y = self.current_image.height - (row * cell_height + cell_height / 2)  # Flip Y
                
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=f"({row},{col})",
                    showarrow=False,
                    font=dict(color="white", size=12),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="white",
                    borderwidth=1
                )
    
    def _configure_plot_layout(self, fig: go.Figure):
        """Configure plot layout and styling"""
        
        fig.update_layout(
            xaxis=dict(
                range=[0, self.current_image.width],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[0, self.current_image.height],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x"
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
            ),
            hovermode='closest'
        )
    
    def _display_detection_statistics(self):
        """Display detection statistics below the plot"""
        
        if not self.detections:
            return
        
        # Calculate statistics
        total = len(self.detections)
        manual_count = sum(1 for d in self.detections if d.get('manual', False))
        auto_count = total - manual_count
        
        # Method breakdown
        methods = {}
        for detection in self.detections:
            method = detection.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        # Confidence statistics
        confidences = [d.get('conf', 1.0) for d in self.detections if not d.get('manual', False)]
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Points", total)
        
        with col2:
            st.metric("Automatic", auto_count)
        
        with col3:
            st.metric("Manual", manual_count)
        
        with col4:
            if confidences:
                avg_conf = np.mean(confidences)
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        # Method breakdown
        if len(methods) > 1:
            st.markdown("**Detection Methods:**")
            method_cols = st.columns(len(methods))
            for i, (method, count) in enumerate(methods.items()):
                with method_cols[i]:
                    st.write(f"**{method}:** {count}")

def create_detection_display() -> DetectionDisplay:
    """Factory function to create DetectionDisplay instance"""
    return DetectionDisplay()