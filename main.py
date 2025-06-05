# main.py
import torch
import streamlit as st
import os
from pathlib import Path
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_CLIENT_SHOW_ERROR_DETAILS"] = "false"

# Configure page
st.set_page_config(
    page_title="🌿 Leaf Tip Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import configurations and modules
from config.app_config import init_session_state, CUSTOM_CSS, UI_CONFIG
from config.model_config import get_available_models, get_best_model

def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Leaf Tip Detector</h1>
        <p>Advanced Computer Vision Tool for Leaf Tip Detection and Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if mode is already selected
    if st.session_state.get('app_mode') is None:
        show_mode_selection()
    else:
        # Show the selected mode
        mode = st.session_state.app_mode
        if mode == 'quick':
            show_quick_detection_page()
        elif mode == 'research':
            show_research_mode_page()
        elif mode == 'annotation':
            show_annotation_page()

def show_mode_selection():
    """Display mode selection interface"""
    
    st.markdown("## Choose Your Experience Level")
    st.markdown("Select the mode that best fits your needs:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="mode-card quick-mode">
            <h3>🌱 Quick Detection</h3>
            <p><strong>Simple & Fast</strong></p>
            <ul>
                <li>Single image or batch processing</li>
                <li>One-click detection</li>
                <li>Download results</li>
            </ul>
            <p><em>Perfect for: Quick analysis</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🌱 Start Quick Detection", key="quick_mode", use_container_width=True):
            st.session_state.app_mode = 'quick'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card research-mode">
            <h3>🔬 Research Mode</h3>
            <p><strong>All Features & Experiments</strong></p>
            <ul>
                <li>Multiple detection models</li>
                <li>Grid analysis (3x3, 5x5)</li>
                <li>ROI-based detection</li>
            </ul>
            <p><em>Perfect for: Research, model comparison</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Start Research Mode", key="research_mode", use_container_width=True):
            st.session_state.app_mode = 'research'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="mode-card annotation-mode">
            <h3>✏️ Annotation</h3>
            <p><strong>Manual Labeling</strong></p>
            <ul>
                <li>Manual point annotation</li>
                <li>Add/remove leaf tips</li>
                <li>Export CSV/XML formats</li>
            </ul>
            <p><em>Perfect for: Ground truth creation</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✏️ Start Annotation", key="annotation_mode", use_container_width=True):
            st.session_state.app_mode = 'annotation'
            st.rerun()
    
    # Show available models info
    with st.expander("🤖 Available Models", expanded=False):
        available_models = get_available_models()
        
        for model_key, config in available_models.items():
            available = config.get('available', True)
            status = "✅" if available else "❌"
            
            st.write(f"{status} **{config['name']}**")
            st.write(f"   - Path: `{config['path']}`")
            st.write(f"   - Type: {config['type']}")
            if config.get('is_best'):
                st.write("   - 🏆 **Best Model**")
            st.write("")

def show_quick_detection_page():
    """Display Quick Detection page"""
    
    # Add mode switcher in sidebar
    with st.sidebar:
        st.markdown("### 🌱 Quick Detection Mode")
        if st.button("🔄 Switch to Research Mode"):
            st.session_state.app_mode = 'research'
            st.rerun()
        
        st.markdown("---")
        if st.button("✏️ Go to Annotation", key="quick_to_annotation"):
            st.session_state.app_mode = 'annotation'
            st.rerun()
    
    # Import and show quick detection page
    try:
        from pages.quick_detection import show_quick_detection_interface
        show_quick_detection_interface()
    except ImportError as e:
        st.error(f"Failed to load Quick Detection page: {e}")
        st.info("Make sure all required modules are installed.")

def show_research_mode_page():
    """Display Research Mode page"""
    
    # Add mode switcher in sidebar
    with st.sidebar:
        st.markdown("### 🔬 Research Mode")
        if st.button("🔄 Switch to Quick Mode"):
            st.session_state.app_mode = 'quick'
            st.rerun()
        
        st.markdown("---")
        if st.button("✏️ Go to Annotation", key="research_to_annotation"):
            st.session_state.app_mode = 'annotation'
            st.rerun()
    
    # Import and show research mode page
    try:
        from pages.research_mode import show_research_mode_interface
        show_research_mode_interface()
    except ImportError as e:
        st.error(f"Failed to load Research Mode page: {e}")
        st.info("Make sure all required modules are installed.")

def show_annotation_page():
    """Display Annotation page"""
    
    # Add mode switcher in sidebar
    with st.sidebar:
        st.markdown("### ✏️ Annotation Mode")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Quick Mode"):
                st.session_state.app_mode = 'quick'
                st.rerun()
        with col2:
            if st.button("🔄 Research Mode"):
                st.session_state.app_mode = 'research'
                st.rerun()
        
        st.markdown("---")
    
    # Import and show annotation page
    try:
        from pages.annotation import show_annotation_interface
        show_annotation_interface()
    except ImportError as e:
        st.error(f"Failed to load Annotation page: {e}")
        st.info("Make sure the annotation module is available.")

def check_requirements():
    """Check if all required files and models are available"""
    
    issues = []
    
    # Check model directory
    models_dir = Path("models")
    if not models_dir.exists():
        issues.append("❌ Models directory not found. Please create 'models/' directory.")
    
    # Check for model files
    from config.model_config import MODEL_CONFIGS
    for model_key, config in MODEL_CONFIGS.items():
        if not os.path.exists(config["path"]):
            issues.append(f"⚠️ Model file not found: {config['path']}")
    
    # Check for required Python packages
    required_packages = [
        'ultralytics', 'torch', 'torchvision', 'PIL', 'numpy', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"❌ Missing packages: {', '.join(missing_packages)}")
    
    return issues

if __name__ == "__main__":
    # 🚨 FIXED: Check if user already bypassed requirements check
    if not st.session_state.get('bypass_requirements_check', False):
        
        # Check requirements
        issues = check_requirements()
        
        if issues:
            st.error("⚠️ **Setup Issues Detected**")
            for issue in issues:
                st.write(issue)
            
            st.markdown("---")
            st.markdown("### 📋 Setup Instructions")
            st.markdown("""
            1. **Install required packages:**
               ```bash
               pip install ultralytics torch torchvision pillow numpy plotly streamlit
               ```
            
            2. **Create models directory and add your model files:**
               ```
               models/
               ├── best(grid_syn_keypt).pt      # 3x3 Grid model
               ├── key_grid_syn_5x5.pt          # 5x5 Grid model (best)
               ├── best.pt                      # YOLO entire image
               └── fold_4_best_map50_aug.pth    # Faster R-CNN model
               ```
            
            3. **Restart the application**
            """)
            
            st.info("The application will work with available models, but some features may be limited.")
            
            # 🚨 FIXED: Set bypass flag when user clicks continue
            if st.button("🚀 Continue Anyway"):
                st.session_state.bypass_requirements_check = True
                st.rerun()
        else:
            # All good, run the app
            main()
    else:
        # Requirements check bypassed, run the app
        main()