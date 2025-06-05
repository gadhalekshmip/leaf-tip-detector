# components/model_selector.py

import streamlit as st
import os
from typing import Dict, List, Optional, Callable
from config.model_config import get_available_models, get_model_config, MODEL_CONFIGS

class ModelSelector:
    """Component for selecting and managing detection models"""
    
    def __init__(self):
        self.available_models = get_available_models()
        self.selected_models = {}
        self.model_status = {}
        self.load_callbacks = {}
    
    def display_model_selector(self,
                             key: str = "model_selector",
                             model_types: List[str] = None,
                             enable_auto_load: bool = True,
                             show_status: bool = True,
                             on_model_loaded: Optional[Callable] = None) -> Dict[str, str]:
        """
        Display model selection interface
        
        Args:
            key: Unique key for the component
            model_types: List of model types to show (None = all)
            enable_auto_load: Show auto-load buttons
            show_status: Show model status indicators
            on_model_loaded: Callback when model is loaded
            
        Returns:
            Dict of selected model keys
        """
        
        if model_types is None:
            model_types = ['yolo_grid', 'yolo_entire', 'frcnn']
        
        self.load_callbacks[key] = on_model_loaded
        
        st.markdown("### 🤖 Model Selection")
        
        # Filter models by type
        filtered_models = self._filter_models_by_type(model_types)
        
        if not filtered_models:
            st.warning("No models available for the specified types")
            return {}
        
        # Display model categories
        selected_models = {}
        
        if 'yolo_grid' in model_types:
            selected_models.update(self._display_grid_models(key, enable_auto_load, show_status))
        
        if 'yolo_entire' in model_types:
            selected_models.update(self._display_entire_models(key, enable_auto_load, show_status))
        
        if 'frcnn' in model_types:
            selected_models.update(self._display_frcnn_models(key, enable_auto_load, show_status))
        
        # Display overall model status summary
        if show_status:
            self._display_model_summary()
        
        return selected_models
    
    def _filter_models_by_type(self, model_types: List[str]) -> Dict[str, Dict]:
        """Filter available models by type"""
        
        filtered = {}
        
        for model_key, config in self.available_models.items():
            model_type = config.get('type', '')
            
            if 'yolo_grid' in model_types and model_type == 'yolo_grid':
                filtered[model_key] = config
            elif 'yolo_entire' in model_types and model_type == 'yolo_entire':
                filtered[model_key] = config
            elif 'frcnn' in model_types and model_type == 'frcnn':
                filtered[model_key] = config
        
        return filtered
    
    def _display_grid_models(self, key: str, enable_auto_load: bool, show_status: bool) -> Dict[str, str]:
        """Display grid model selection"""
        
        st.markdown("#### 📊 Grid Models")
        
        selected = {}
        
        # Grid model selection
        col1, col2 = st.columns(2)
        
        with col1:
            # 3x3 Grid Model
            st.markdown("**3x3 Grid Model**")
            
            config_3x3 = get_model_config("grid_3x3")
            if config_3x3:
                available = os.path.exists(config_3x3["path"])
                
                if enable_auto_load:
                    if st.button("📥 Load 3x3 Model", key=f"{key}_load_3x3"):
                        self._load_model("grid_3x3", key)
                
                if show_status:
                    self._display_model_status("grid_3x3", available)
                
                # Selection radio
                if available and st.checkbox("Use 3x3 Grid", key=f"{key}_select_3x3"):
                    selected["grid"] = "grid_3x3"
        
        with col2:
            # 5x5 Grid Model
            st.markdown("**5x5 Grid Model** 🏆")
            
            config_5x5 = get_model_config("grid_5x5")
            if config_5x5:
                available = os.path.exists(config_5x5["path"])
                
                if enable_auto_load:
                    if st.button("📥 Load 5x5 Model", key=f"{key}_load_5x5"):
                        self._load_model("grid_5x5", key)
                
                if show_status:
                    self._display_model_status("grid_5x5", available)
                
                # Selection radio (default selected if available)
                default_selected = available and "grid" not in selected
                if available and st.checkbox("Use 5x5 Grid", value=default_selected, key=f"{key}_select_5x5"):
                    selected["grid"] = "grid_5x5"
        
        return selected
    
    def _display_entire_models(self, key: str, enable_auto_load: bool, show_status: bool) -> Dict[str, str]:
        """Display entire image model selection"""
        
        st.markdown("#### 🖼️ Entire Image Models")
        
        selected = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            # YOLO Entire Model
            st.markdown("**YOLO Entire Image**")
            
            config_yolo = get_model_config("yolo_entire")
            if config_yolo:
                available = os.path.exists(config_yolo["path"])
                
                if enable_auto_load:
                    if st.button("📥 Load YOLO Model", key=f"{key}_load_yolo_entire"):
                        self._load_model("yolo_entire", key)
                
                if show_status:
                    self._display_model_status("yolo_entire", available)
                
                if available and st.radio(
                    "Entire Image Model",
                    ["YOLO", "Faster R-CNN"],
                    key=f"{key}_entire_model_type"
                ) == "YOLO":
                    selected["entire"] = "yolo_entire"
        
        with col2:
            # FRCNN Model (will be handled in _display_frcnn_models)
            pass
        
        return selected
    
    def _display_frcnn_models(self, key: str, enable_auto_load: bool, show_status: bool) -> Dict[str, str]:
        """Display Faster R-CNN model selection"""
        
        st.markdown("#### 🎯 Faster R-CNN Model")
        
        selected = {}
        
        config_frcnn = get_model_config("frcnn")
        if config_frcnn:
            available = os.path.exists(config_frcnn["path"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if enable_auto_load:
                    if st.button("📥 Load Faster R-CNN", key=f"{key}_load_frcnn"):
                        self._load_model("frcnn", key)
            
            with col2:
                if show_status:
                    self._display_model_status("frcnn", available)
            
            # Check if FRCNN is selected from radio in entire models
            if available and hasattr(st.session_state, f"{key}_entire_model_type"):
                if st.session_state[f"{key}_entire_model_type"] == "Faster R-CNN":
                    selected["entire"] = "frcnn"
        
        return selected
    
    def _display_model_status(self, model_key: str, available: bool):
        """Display status for a specific model"""
        
        config = get_model_config(model_key)
        if not config:
            return
        
        # Check if model is loaded in session state
        session_key = f"{model_key}_processor"
        loaded = st.session_state.get(session_key) is not None
        
        if loaded:
            st.success("✅ Loaded & Ready")
        elif available:
            st.warning("⚪ Available (Click to Load)")
        else:
            st.error("❌ Model File Missing")
            st.caption(f"Expected: {config['path']}")
    
    def _load_model(self, model_key: str, component_key: str):
        """Load a specific model"""
        
        config = get_model_config(model_key)
        if not config:
            st.error(f"No configuration found for {model_key}")
            return
        
        if not os.path.exists(config["path"]):
            st.error(f"Model file not found: {config['path']}")
            return
        
        try:
            with st.spinner(f"Loading {config['name']}..."):
                success = False
                
                if config["type"] == "yolo_grid" or config["type"] == "yolo_entire":
                    success = self._load_yolo_model(model_key, config)
                elif config["type"] == "frcnn":
                    success = self._load_frcnn_model(model_key, config)
                
                if success:
                    st.success(f"✅ {config['name']} loaded successfully!")
                    
                    # Call callback if provided
                    callback = self.load_callbacks.get(component_key)
                    if callback:
                        callback(model_key, config)
                else:
                    st.error(f"❌ Failed to load {config['name']}")
                    
        except Exception as e:
            st.error(f"Error loading {config['name']}: {str(e)}")
    
    def _load_yolo_model(self, model_key: str, config: Dict) -> bool:
        """Load YOLO model"""
        
        try:
            from core.yolo_processor import create_yolo_processor
            
            processor = create_yolo_processor(config)
            if processor.is_loaded():
                # Store in session state
                session_key = f"{model_key}_processor"
                st.session_state[session_key] = processor
                
                # Also store for grid processing if applicable
                if config["type"] == "yolo_grid":
                    if "3x3" in model_key:
                        st.session_state["grid_3x3_processor"] = processor
                    elif "5x5" in model_key:
                        st.session_state["grid_5x5_processor"] = processor
                elif config["type"] == "yolo_entire":
                    st.session_state["entire_yolo_processor"] = processor
                
                return True
                
        except Exception as e:
            st.error(f"YOLO model loading error: {str(e)}")
        
        return False
    
    def _load_frcnn_model(self, model_key: str, config: Dict) -> bool:
        """Load Faster R-CNN model"""
        
        try:
            from core.frcnn_processor import create_frcnn_processor
            
            processor = create_frcnn_processor(config)
            if processor.is_loaded():
                # Store in session state
                session_key = f"{model_key}_processor"
                st.session_state[session_key] = processor
                st.session_state["entire_frcnn_processor"] = processor
                
                return True
                
        except Exception as e:
            st.error(f"FRCNN model loading error: {str(e)}")
        
        return False
    
    def _display_model_summary(self):
        """Display overall model status summary"""
        
        st.markdown("---")
        st.markdown("#### 📋 Model Status Summary")
        
        status_data = []
        
        for model_key, config in MODEL_CONFIGS.items():
            available = os.path.exists(config["path"])
            session_key = f"{model_key}_processor"
            loaded = st.session_state.get(session_key) is not None
            
            status_data.append({
                "Model": config["name"],
                "Type": config["type"].replace("_", " ").title(),
                "Available": "✅" if available else "❌",
                "Loaded": "✅" if loaded else "⚪",
                "File": os.path.basename(config["path"])
            })
        
        # Display as dataframe
        import pandas as pd
        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary metrics
        total_models = len(status_data)
        available_count = sum(1 for item in status_data if item["Available"] == "✅")
        loaded_count = sum(1 for item in status_data if item["Loaded"] == "✅")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Available", available_count)
        with col3:
            st.metric("Loaded", loaded_count)
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all loaded models"""
        
        loaded_models = {}
        
        for model_key in MODEL_CONFIGS.keys():
            session_key = f"{model_key}_processor"
            loaded_models[model_key] = st.session_state.get(session_key) is not None
        
        return loaded_models
    
    def auto_load_best_model(self, model_type: str = "grid") -> bool:
        """Auto-load the best available model of specified type"""
        
        if model_type == "grid":
            # Try 5x5 first (best), then 3x3
            for model_key in ["grid_5x5", "grid_3x3"]:
                config = get_model_config(model_key)
                if config and os.path.exists(config["path"]):
                    return self._load_yolo_model(model_key, config)
        
        elif model_type == "entire":
            # Try FRCNN first (recommended), then YOLO
            for model_key in ["frcnn", "yolo_entire"]:
                config = get_model_config(model_key)
                if config and os.path.exists(config["path"]):
                    if model_key == "frcnn":
                        return self._load_frcnn_model(model_key, config)
                    else:
                        return self._load_yolo_model(model_key, config)
        
        return False
    
    def unload_model(self, model_key: str):
        """Unload a specific model from memory"""
        
        session_key = f"{model_key}_processor"
        if session_key in st.session_state:
            del st.session_state[session_key]
        
        # Also remove from specific processor stores
        related_keys = [
            "grid_3x3_processor", "grid_5x5_processor",
            "entire_yolo_processor", "entire_frcnn_processor"
        ]
        
        for key in related_keys:
            if key in st.session_state:
                processor = st.session_state[key]
                if hasattr(processor, 'model_path') and model_key in str(processor.model_path):
                    del st.session_state[key]

def create_model_selector() -> ModelSelector:
    """Factory function to create ModelSelector instance"""
    return ModelSelector()