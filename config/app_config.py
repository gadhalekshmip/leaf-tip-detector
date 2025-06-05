# config/app_config.py

import streamlit as st

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Leaf Tip Detector",
    "page_icon": "🌿",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# UI Configuration
UI_CONFIG = {
    "modes": {
        "quick": {
            "title": "🌱 Quick Detection",
            "description": "Simple & Fast Detection",
            "icon": "🌱"
        },
        "research": {
            "title": "🔬 Research Mode", 
            "description": "All Features & Experiments",
            "icon": "🔬"
        },
        "annotation": {  # ADD THIS
            "title": "✏️ Annotation",
            "description": "Manual Labeling & Ground Truth",
            "icon": "✏️"
        }
    },
    "tabs": {
        "entire_image": {
            "title": "Entire Image",
            "icon": "🖼️",
            "description": "Process the complete image at once"
        },
        "grid_analysis": {
            "title": "Grid Analysis", 
            "icon": "📊",
            "description": "Grid-based detection with 3x3 or 5x5 patterns"
        },
        "roi_analysis": {
            "title": "ROI Analysis",
            "icon": "🎯", 
            "description": "Region of Interest focused detection"
        }
    },
    "colors": {
        "primary": "#4CAF50",
        "secondary": "#2196F3", 
        "accent": "#FF5722",
        "warning": "#FF9800",
        "success": "#4CAF50",
        "error": "#F44336"
    }
}

# Session state keys
SESSION_KEYS = {
    "mode": "app_mode",
    "current_image": "current_image",
    "detections": "detections",
    "batch_results": "batch_results",
    "models": "loaded_models",
    "settings": "app_settings"
}

def init_session_state():
    """Initialize Streamlit session state variables"""
    
    # App mode
    if SESSION_KEYS["mode"] not in st.session_state:
        st.session_state[SESSION_KEYS["mode"]] = None
    
    # Current image and detections
    if SESSION_KEYS["current_image"] not in st.session_state:
        st.session_state[SESSION_KEYS["current_image"]] = None
    
    if SESSION_KEYS["detections"] not in st.session_state:
        st.session_state[SESSION_KEYS["detections"]] = {}
    
    # Batch processing results
    if SESSION_KEYS["batch_results"] not in st.session_state:
        st.session_state[SESSION_KEYS["batch_results"]] = {}
    
    # Loaded models cache
    if SESSION_KEYS["models"] not in st.session_state:
        st.session_state[SESSION_KEYS["models"]] = {}
    
    # App settings
    if SESSION_KEYS["settings"] not in st.session_state:
        st.session_state[SESSION_KEYS["settings"]] = {
            "auto_load_models": True,
            "show_debug": False,
            "max_image_size": 1536
        }

def get_session_state(key):
    """Get value from session state"""
    return st.session_state.get(SESSION_KEYS.get(key, key), None)

def set_session_state(key, value):
    """Set value in session state"""
    st.session_state[SESSION_KEYS.get(key, key)] = value

def clear_session_state(key=None):
    """Clear session state - all or specific key"""
    if key:
        if SESSION_KEYS.get(key, key) in st.session_state:
            del st.session_state[SESSION_KEYS.get(key, key)]
    else:
        # Clear all app-related session state
        for session_key in SESSION_KEYS.values():
            if session_key in st.session_state:
                del st.session_state[session_key]

# CSS Styles for the app
CUSTOM_CSS = """
<style>
/* Main app styling */
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.mode-card {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.mode-card:hover {
    border-color: #4CAF50;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.quick-mode {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
}

.research-mode {
    background: linear-gradient(45deg, #2196F3, #1976D2);
    color: white;
}

.detection-count {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E7D32;
    text-align: center;
    padding: 1rem;
    background-color: #f0f8f0;
    border-radius: 8px;
    margin: 1rem 0;
}

.model-status {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    font-weight: bold;
}

.model-loaded {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.model-not-loaded {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.results-panel {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.batch-progress {
    margin: 1rem 0;
}

.debug-panel {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.annotation-mode {
    background: linear-gradient(45deg, #FF5722, #E64A19);
    color: white;
}
</style>
"""