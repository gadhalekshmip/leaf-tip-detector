# components/image_viewer.py

import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from typing import Optional, Tuple, Dict, Any, Callable
import base64
import io

class ImageViewer:
    """Advanced image viewer component with zoom, pan, and interaction capabilities"""
    
    def __init__(self):
        self.current_image = None
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        self.view_settings = {
            'show_toolbar': True,
            'enable_zoom': True,
            'enable_pan': True,
            'enable_select': False,
            'fit_on_load': True
        }
    
    def display_image(self,
                     image: Image.Image,
                     key: str = "image_viewer",
                     height: int = 600,
                     enable_zoom_controls: bool = True,
                     on_click: Optional[Callable] = None,
                     overlay_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Display image with advanced viewing capabilities
        
        Args:
            image: PIL Image to display
            key: Unique key for the component
            height: Height of the viewer
            enable_zoom_controls: Show zoom control buttons
            on_click: Callback for click events
            overlay_data: Data to overlay on image (detections, annotations)
            
        Returns:
            Dict with interaction results
        """
        
        self.current_image = image
        
        # Display zoom controls if enabled
        if enable_zoom_controls:
            self._display_zoom_controls(key)
        
        # Create interactive plot
        fig = self._create_image_plot(overlay_data)
        
        # Configure interaction callbacks
        if on_click:
            self._setup_click_callback(fig, on_click)
        
        # Display the plot
        plot_result = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{key}_plot",
            config={
                'displayModeBar': self.view_settings['show_toolbar'],
                'scrollZoom': self.view_settings['enable_zoom'],
                'doubleClick': 'reset+autosize'
            }
        )
        
        # Display image information
        self._display_image_info()
        
        return {
            'plot_result': plot_result,
            'zoom_level': self.zoom_level,
            'pan_offset': self.pan_offset
        }
    
    def _display_zoom_controls(self, key: str):
        """Display zoom and navigation controls"""
        
        st.markdown("**🔍 View Controls**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🔍+", key=f"{key}_zoom_in", help="Zoom In"):
                self._zoom_in()
                st.rerun()
        
        with col2:
            if st.button("🔍-", key=f"{key}_zoom_out", help="Zoom Out"):
                self._zoom_out()
                st.rerun()
        
        with col3:
            if st.button("📐", key=f"{key}_fit_view", help="Fit to View"):
                self._fit_to_view()
                st.rerun()
        
        with col4:
            if st.button("🔄", key=f"{key}_reset", help="Reset View"):
                self._reset_view()
                st.rerun()
        
        with col5:
            # Zoom level display
            st.metric("Zoom", f"{self.zoom_level:.1f}x")
        
        # Zoom slider
        new_zoom = st.slider(
            "Zoom Level",
            min_value=0.1,
            max_value=5.0,
            value=self.zoom_level,
            step=0.1,
            key=f"{key}_zoom_slider"
        )
        
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            st.rerun()
    
    def _create_image_plot(self, overlay_data: Optional[Dict] = None) -> go.Figure:
        """Create plotly figure for image display"""
        
        fig = go.Figure()
        
        # Add image as background
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
        
        # Add overlay data if provided
        if overlay_data:
            self._add_overlay_data(fig, overlay_data)
        
        # Configure layout with current view settings
        self._configure_plot_layout(fig)
        
        return fig
    
    def _add_overlay_data(self, fig: go.Figure, overlay_data: Dict):
        """Add overlay data to the plot"""
        
        # Handle different types of overlay data
        if 'detections' in overlay_data:
            self._add_detection_overlay(fig, overlay_data['detections'])
        
        if 'annotations' in overlay_data:
            self._add_annotation_overlay(fig, overlay_data['annotations'])
        
        if 'shapes' in overlay_data:
            self._add_shape_overlay(fig, overlay_data['shapes'])
        
        if 'grid' in overlay_data:
            self._add_grid_overlay(fig, overlay_data['grid'])
    
    def _add_detection_overlay(self, fig: go.Figure, detections: list):
        """Add detection points to the plot"""
        
        if not detections:
            return
        
        x_coords = [d['x'] for d in detections]
        y_coords = [self.current_image.height - d['y'] for d in detections]  # Flip Y
        confidences = [d.get('conf', 1.0) for d in detections]
        
        # Color by detection type
        colors = []
        for detection in detections:
            if detection.get('manual', False):
                colors.append('green')
            elif 'grid' in detection.get('method', ''):
                colors.append('red')
            elif 'roi' in detection.get('method', ''):
                colors.append('blue')
            else:
                colors.append('orange')
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=[6 + c*8 for c in confidences],
                color=colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[f"Conf: {c:.3f}" for c in confidences],
            hovertemplate="<b>Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
            customdata=[d['y'] for d in detections],
            name="Detections",
            showlegend=True
        ))
    
    def _add_annotation_overlay(self, fig: go.Figure, annotations: list):
        """Add text annotations to the plot"""
        
        for annotation in annotations:
            fig.add_annotation(
                x=annotation['x'],
                y=self.current_image.height - annotation['y'],  # Flip Y
                text=annotation['text'],
                showarrow=annotation.get('show_arrow', True),
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=annotation.get('color', 'red'),
                font=dict(
                    color=annotation.get('color', 'red'),
                    size=annotation.get('font_size', 12)
                ),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=annotation.get('color', 'red'),
                borderwidth=1
            )
    
    def _add_shape_overlay(self, fig: go.Figure, shapes: list):
        """Add geometric shapes to the plot"""
        
        for shape in shapes:
            shape_type = shape.get('type', 'rect')
            
            if shape_type == 'rect':
                fig.add_shape(
                    type="rect",
                    x0=shape['x1'],
                    y0=self.current_image.height - shape['y2'],  # Flip Y
                    x1=shape['x2'],
                    y1=self.current_image.height - shape['y1'],  # Flip Y
                    line=dict(
                        color=shape.get('color', 'red'),
                        width=shape.get('width', 2)
                    ),
                    fillcolor=shape.get('fill_color', 'rgba(255,0,0,0.1)')
                )
            elif shape_type == 'circle':
                # Convert circle to scatter point
                fig.add_trace(go.Scatter(
                    x=[shape['x']],
                    y=[self.current_image.height - shape['y']],  # Flip Y
                    mode='markers',
                    marker=dict(
                        size=shape.get('radius', 10) * 2,
                        color=shape.get('color', 'red'),
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False
                ))
    
    def _add_grid_overlay(self, fig: go.Figure, grid_config: dict):
        """Add grid overlay to the plot"""
        
        grid_size = grid_config.get('size', 3)
        color = grid_config.get('color', 'cyan')
        width = grid_config.get('width', 2)
        
        img_width = self.current_image.width
        img_height = self.current_image.height
        
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size
        
        # Add vertical lines
        for i in range(1, grid_size):
            x = i * cell_width
            fig.add_vline(x=x, line_color=color, line_width=width, opacity=0.7)
        
        # Add horizontal lines
        for i in range(1, grid_size):
            y = self.current_image.height - (i * cell_height)  # Flip Y
            fig.add_hline(y=y, line_color=color, line_width=width, opacity=0.7)
    
    def _configure_plot_layout(self, fig: go.Figure):
        """Configure plot layout based on current view settings"""
        
        img_width = self.current_image.width
        img_height = self.current_image.height
        
        # Calculate view range based on zoom and pan
        center_x = img_width / 2 + self.pan_offset[0]
        center_y = img_height / 2 + self.pan_offset[1]
        
        view_width = img_width / self.zoom_level
        view_height = img_height / self.zoom_level
        
        x_range = [
            max(0, center_x - view_width / 2),
            min(img_width, center_x + view_width / 2)
        ]
        y_range = [
            max(0, center_y - view_height / 2),
            min(img_height, center_y + view_height / 2)
        ]
        
        fig.update_layout(
            xaxis=dict(
                range=x_range,
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=y_range,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x"
            ),
            width=None,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            hovermode='closest'
        )
    
    def _setup_click_callback(self, fig: go.Figure, callback: Callable):
        """Setup click event callback"""
        # Note: Plotly click callbacks in Streamlit are limited
        # This is a placeholder for click handling
        pass
    
    def _zoom_in(self):
        """Zoom in by 25%"""
        self.zoom_level = min(5.0, self.zoom_level * 1.25)
    
    def _zoom_out(self):
        """Zoom out by 25%"""
        self.zoom_level = max(0.1, self.zoom_level * 0.8)
    
    def _fit_to_view(self):
        """Reset zoom to fit entire image"""
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
    
    def _reset_view(self):
        """Reset all view parameters"""
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
    
    def _display_image_info(self):
        """Display image information panel"""
        
        if not self.current_image:
            return
        
        with st.expander("ℹ️ Image Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Dimensions:** {self.current_image.width} × {self.current_image.height}")
                st.write(f"**Mode:** {self.current_image.mode}")
            
            with col2:
                aspect_ratio = self.current_image.width / self.current_image.height
                st.write(f"**Aspect Ratio:** {aspect_ratio:.2f}")
                megapixels = (self.current_image.width * self.current_image.height) / 1_000_000
                st.write(f"**Megapixels:** {megapixels:.1f}MP")
            
            with col3:
                st.write(f"**Current Zoom:** {self.zoom_level:.1f}x")
                if self.pan_offset != (0, 0):
                    st.write(f"**Pan Offset:** ({self.pan_offset[0]:.0f}, {self.pan_offset[1]:.0f})")
    
    def get_current_view_bounds(self) -> Tuple[int, int, int, int]:
        """Get current view bounds in image coordinates"""
        
        img_width = self.current_image.width
        img_height = self.current_image.height
        
        center_x = img_width / 2 + self.pan_offset[0]
        center_y = img_height / 2 + self.pan_offset[1]
        
        view_width = img_width / self.zoom_level
        view_height = img_height / self.zoom_level
        
        x1 = max(0, int(center_x - view_width / 2))
        y1 = max(0, int(center_y - view_height / 2))
        x2 = min(img_width, int(center_x + view_width / 2))
        y2 = min(img_height, int(center_y + view_height / 2))
        
        return (x1, y1, x2, y2)
    
    def set_view_to_region(self, x1: int, y1: int, x2: int, y2: int):
        """Set view to focus on specific region"""
        
        region_width = x2 - x1
        region_height = y2 - y1
        
        if region_width <= 0 or region_height <= 0:
            return
        
        # Calculate zoom level to fit region
        zoom_x = self.current_image.width / region_width
        zoom_y = self.current_image.height / region_height
        self.zoom_level = min(zoom_x, zoom_y)
        
        # Calculate pan offset to center on region
        region_center_x = (x1 + x2) / 2
        region_center_y = (y1 + y2) / 2
        
        img_center_x = self.current_image.width / 2
        img_center_y = self.current_image.height / 2
        
        self.pan_offset = (
            region_center_x - img_center_x,
            region_center_y - img_center_y
        )

def create_image_viewer() -> ImageViewer:
    """Factory function to create ImageViewer instance"""
    return ImageViewer()