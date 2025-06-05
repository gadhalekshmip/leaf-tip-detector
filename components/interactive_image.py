# components/interactive_image.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from streamlit_drawable_canvas import st_canvas
import streamlit_image_coordinates as sic

class InteractiveImageEditor:
    """Interactive image editor with point editing and ROI selection capabilities"""
    
    def __init__(self):
        self.image = None
        self.detections = []
        self.manual_points = []
        self.roi_coords = None
        
    def display_image_with_points(self, 
                                 image: Image.Image,
                                 detections: List[Dict],
                                 key: str = "image_editor",
                                 height: int = 600,
                                 enable_editing: bool = True) -> Dict:
        """
        Display image with detection points and enable interactive editing
        Returns: {"clicked_point": (x, y), "action": "add"/"remove", "roi": coordinates}
        """
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create plotly figure
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
        
        # Add detection points
        if detections:
            x_coords = [d['x'] for d in detections]
            y_coords = [image.height - d['y'] for d in detections]  # Flip Y for plotly
            confidences = [d.get('conf', 1.0) for d in detections]
            
            # Color points by confidence or method
            colors = []
            for d in detections:
                if d.get('manual', False):
                    colors.append('green')
                elif 'grid' in d.get('method', ''):
                    colors.append('red')
                elif 'roi' in d.get('method', ''):
                    colors.append('blue')
                else:
                    colors.append('red')
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=[4 + c*6 for c in confidences],  # Size based on confidence
                    color=colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[f"Conf: {c:.3f}" for c in confidences],
                hovertemplate="<b>Detection</b><br>X: %{x}<br>Y: %{customdata}<br>%{text}<extra></extra>",
                customdata=[d['y'] for d in detections],  # Original Y coordinates
                name="Detections"
            ))
        
        # Configure layout
        fig.update_layout(
            xaxis=dict(range=[0, image.width], showgrid=False, zeroline=False),
            yaxis=dict(range=[0, image.height], showgrid=False, zeroline=False, scaleanchor="x"),
            width=None,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        # Display interactive plot
        clicked_data = st.plotly_chart(
            fig, 
            use_container_width=True,
            key=f"{key}_plot"
        )
        
        result = {"clicked_point": None, "action": None, "roi": None}
        
        if enable_editing:
            # Add editing controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                add_mode = st.button("➕ Add Point Mode", key=f"{key}_add")
            with col2:
                remove_mode = st.button("➖ Remove Point Mode", key=f"{key}_remove") 
            with col3:
                roi_mode = st.button("🎯 ROI Selection", key=f"{key}_roi")
            
            # Handle click events (this is a simplified version - in practice you'd need more sophisticated event handling)
            if clicked_data and hasattr(clicked_data, 'last_clicked'):
                click_data = clicked_data.last_clicked
                if click_data:
                    x = click_data.get('x', 0)
                    y = image.height - click_data.get('y', 0)  # Flip Y back
                    
                    if add_mode:
                        result["clicked_point"] = (x, y)
                        result["action"] = "add"
                    elif remove_mode:
                        result["clicked_point"] = (x, y)
                        result["action"] = "remove"
        
        return result

class ROISelector:
    """ROI selection component using drawable canvas"""
    
    def __init__(self):
        self.roi_coords = None
        
    def draw_roi_selector(self, 
                         image: Image.Image,
                         key: str = "roi_selector",
                         height: int = 600) -> Optional[Tuple[int, int, int, int]]:
        """
        Draw ROI selector on image
        Returns: (x1, y1, x2, y2) coordinates or None
        """
        
        st.write("**Draw ROI:** Click and drag to select region of interest")
        
        # Create canvas for ROI drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",  # Semi-transparent red
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image,
            height=height,
            width=image.width * (height / image.height) if image.height > height else image.width,
            drawing_mode="rect",
            key=f"{key}_canvas",
            display_toolbar=True
        )
        
        # Extract ROI coordinates if drawn
        roi_coords = None
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                # Get the last drawn rectangle
                rect = objects[-1]
                if rect["type"] == "rect":
                    # Convert canvas coordinates to image coordinates
                    scale_x = image.width / canvas_result.image_data.shape[1]
                    scale_y = image.height / canvas_result.image_data.shape[0]
                    
                    x1 = int(rect["left"] * scale_x)
                    y1 = int(rect["top"] * scale_y)
                    x2 = int((rect["left"] + rect["width"]) * scale_x)
                    y2 = int((rect["top"] + rect["height"]) * scale_y)
                    
                    roi_coords = (x1, y1, x2, y2)
                    
                    st.success(f"ROI selected: ({x1}, {y1}) to ({x2}, {y2})")
        
        return roi_coords

class ManualPointEditor:
    """Manual point addition/removal using coordinate inputs and image clicks"""
    
    def __init__(self):
        pass
        
    def edit_points_interface(self, 
                             detections: List[Dict],
                             image_size: Tuple[int, int],
                             key: str = "point_editor") -> Dict:
        """
        Interface for manually editing detection points
        Returns: {"action": "add"/"remove", "point": (x, y), "index": int}
        """
        
        result = {"action": None, "point": None, "index": None}
        
        st.write("**Manual Point Editing**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Add Point**")
            add_x = st.number_input("X coordinate", min_value=0, max_value=image_size[0], key=f"{key}_add_x")
            add_y = st.number_input("Y coordinate", min_value=0, max_value=image_size[1], key=f"{key}_add_y")
            
            if st.button("Add Point", key=f"{key}_add_btn"):
                result["action"] = "add"
                result["point"] = (add_x, add_y)
        
        with col2:
            st.write("**Remove Point**")
            if detections:
                # Create list of points for selection
                point_options = [f"Point {i+1}: ({int(d['x'])}, {int(d['y'])})" for i, d in enumerate(detections)]
                selected_point = st.selectbox("Select point to remove", point_options, key=f"{key}_remove_select")
                
                if st.button("Remove Point", key=f"{key}_remove_btn") and selected_point:
                    # Extract index from selection
                    point_index = int(selected_point.split(":")[0].split()[1]) - 1
                    result["action"] = "remove"
                    result["index"] = point_index
            else:
                st.info("No points to remove")
        
        return result

def create_interactive_image_editor() -> InteractiveImageEditor:
    """Create interactive image editor instance"""
    return InteractiveImageEditor()

def create_roi_selector() -> ROISelector:
    """Create ROI selector instance"""
    return ROISelector()

def create_manual_point_editor() -> ManualPointEditor:
    """Create manual point editor instance"""
    return ManualPointEditor()