# pages/research_mode_enhanced.py

import streamlit as st
from config.app_config import UI_CONFIG

def show_research_mode_interface():
    """FIXED Research Mode - removed top switch menu, removed conf threshold"""
    
    st.markdown("## 🔬 Research Mode - Advanced Analysis")
    st.markdown("Comprehensive leaf tip detection with multiple models and analysis methods.")
    
    # Initialize session state for research mode
    if 'research_tab' not in st.session_state:
        st.session_state.research_tab = 'entire_image'
    
    # REMOVED: Top switch menu (keep only sidebar one)
    
    # Create tabs
    tab_configs = UI_CONFIG['tabs']
    
    tab1, tab2, tab3 = st.tabs([
        f"{tab_configs['entire_image']['icon']} {tab_configs['entire_image']['title']}",
        f"{tab_configs['grid_analysis']['icon']} {tab_configs['grid_analysis']['title']}", 
        f"{tab_configs['roi_analysis']['icon']} {tab_configs['roi_analysis']['title']}"
    ])
    
    with tab1:
        show_entire_image_tab()
    
    with tab2:
        show_grid_analysis_tab()
    
    with tab3:
        show_roi_analysis_tab()
    
    # Sidebar with common controls
    show_research_sidebar()

def show_switch_menu():
    """Switch Menu - Switch to Quick Mode"""
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "🔄 Switch to Quick Mode", 
                key="switch_to_quick_mode",
                use_container_width=True,
                type="secondary"
            ):
                # Clear research mode state when switching (as per documentation)
                clear_research_mode_state()
                
                # Switch to quick mode
                st.session_state.app_mode = 'quick'
                st.rerun()
        
        st.markdown("---")

def clear_research_mode_state():
    """Clear all research mode state when switching modes (Mode Switching Behavior)"""
    
    # Clear all tab-specific states
    research_keys = [
        # Entire Image tab
        'entire_detections', 'entire_current_image', 'entire_current_image_path',
        'entire_batch_results', 'entire_image_files', 'entire_current_image_index',
        'entire_yolo_processor', 'entire_frcnn_processor', 'entire_selected_model',
        'entire_zoom_level',
        
        # Grid Analysis tab
        'grid_detections', 'grid_raw_detections', 'grid_current_image', 
        'grid_current_image_path', 'grid_batch_results', 'grid_image_files',
        'grid_current_image_index', 'grid_3x3_processor', 'grid_5x5_processor',
        'grid_selected_size', 'grid_show_visualization', 'grid_zoom_level',
        
        # ROI Analysis tab
        'roi_detections', 'roi_current_image', 'roi_current_image_path',
        'roi_batch_results', 'roi_image_files', 'roi_current_image_index',
        'roi_coordinates', 'roi_processor', 'roi_zoom_level', 'roi_drawing_mode',
        
        # Common research state
        'research_tab'
    ]
    
    for key in research_keys:
        if key in st.session_state:
            del st.session_state[key]

def show_entire_image_tab():
    """Show entire image processing tab"""
    try:
        from tabs.entire_image_tab import show_entire_image_interface
        show_entire_image_interface()
    except ImportError as e:
        st.error(f"Failed to load Entire Image tab: {e}")
        st.info("Please ensure all tab modules are properly installed.")

def show_grid_analysis_tab():
    """Show enhanced grid analysis tab with debug features"""
    try:
        from tabs.grid_analysis_tab import show_grid_analysis_interface
        show_grid_analysis_interface()
    except ImportError as e:
        st.error(f"Failed to load Grid Analysis tab: {e}")
        st.info("Please ensure all tab modules are properly installed.")

def show_roi_analysis_tab():
    """Show ROI analysis tab"""
    try:
        from tabs.roi_analysis_tab import show_roi_analysis_interface
        show_roi_analysis_interface()
    except ImportError as e:
        st.error(f"Failed to load ROI Analysis tab: {e}")
        st.info("Please ensure all tab modules are properly installed.")

def show_research_sidebar():
    """Show research mode sidebar with enhanced controls"""
    
    with st.sidebar:
        st.markdown("### 🔬 Research Controls")
        
        # Model management
        st.markdown("#### 🤖 Model Status")
        show_model_status_sidebar()
        
        # Common settings
        st.markdown("#### ⚙️ Common Settings")
        show_common_settings()
        
        
        # Batch operations
        st.markdown("#### 📦 Batch Operations")
        show_batch_operations()

def show_model_status_sidebar():
    """Show comprehensive model loading status in sidebar"""
    
    from config.model_config import get_available_models
    
    available_models = get_available_models()
    
    for model_key, config in available_models.items():
        available = config.get('available', True)
        
        if available:
            status_color = "🟢"
            status_text = "Ready"
        else:
            status_color = "🔴" 
            status_text = "Missing"
        
        st.write(f"{status_color} **{config['name']}**")
        st.caption(f"Status: {status_text}")
        
        if not available:
            st.caption(f"⚠️ File not found: {config['path']}")

def show_common_settings():
    """Show enhanced common settings across tabs"""
    
    # Image display settings
    st.selectbox(
        "Image Display Size",
        options=["Small", "Medium", "Large", "Fit to View"],  # Changed "Full Width" to "Fit to View"
        index=3,  # Default to "Fit to View"
        key="display_size"
    )
    
    
    


def show_batch_operations():
    """Show enhanced batch processing operations"""
    
    # Check if any tab has batch results
    has_batch_results = False
    batch_counts = {}
    
    for key, name in [
        ('entire_batch_results', 'Entire Image'),
        ('grid_batch_results', 'Grid Analysis'), 
        ('roi_batch_results', 'ROI Analysis')
    ]:
        if st.session_state.get(key):
            has_batch_results = True
            batch_counts[name] = len(st.session_state[key])
    
    if has_batch_results:
        st.success("✅ Batch results available")
        
        # Show counts for each method
        for method, count in batch_counts.items():
            st.write(f"- {method}: {count} images")
        
        # Batch operation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 View Summary", key="view_batch_summary"):
                show_batch_summary_modal()
        
        with col2:
            if st.button("💾 Download All", key="download_all_results"):
                download_all_batch_results()
        
        # Additional batch controls
        st.markdown("**Batch Controls:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear All", key="clear_all_batch"):
                clear_all_batch_results()
        
        with col2:
            if st.button("📋 Compare", key="compare_methods"):
                show_method_comparison()
                
    else:
        st.info("No batch results yet")
        st.caption("Process images in batch mode to see results here")

def show_batch_summary_modal():
    """Show enhanced batch processing summary - FIXED for sidebar"""
    
    st.markdown("### 📊 Comprehensive Batch Summary")
    
    # Collect results from all tabs
    all_results = {}
    
    # Entire image results
    if 'entire_batch_results' in st.session_state and st.session_state.entire_batch_results:
        all_results['Entire Image'] = st.session_state.entire_batch_results
    
    # Grid analysis results
    if 'grid_batch_results' in st.session_state and st.session_state.grid_batch_results:
        all_results['Grid Analysis'] = st.session_state.grid_batch_results
    
    # ROI analysis results
    if 'roi_batch_results' in st.session_state and st.session_state.roi_batch_results:
        all_results['ROI Analysis'] = st.session_state.roi_batch_results
    
    if not all_results:
        st.info("No batch results to display")
        return
    
    # Display enhanced summary for each method (NO COLUMNS - use metrics instead)
    for method, results in all_results.items():
        st.markdown(f"#### {method}")
        
        if results:
            total_images = len(results)
            total_detections = sum(len(r.get('detections', [])) for r in results.values())
            avg_detections = total_detections / total_images if total_images > 0 else 0
            
            # Enhanced metrics
            detection_counts = [len(r.get('detections', [])) for r in results.values()]
            min_detections = min(detection_counts) if detection_counts else 0
            max_detections = max(detection_counts) if detection_counts else 0
            
            # Use individual metrics instead of columns
            st.metric("Images", total_images)
            st.metric("Total Tips", total_detections)
            st.metric("Avg/Image", f"{avg_detections:.1f}")
            st.metric("Range", f"{min_detections}-{max_detections}")
                
            # Processing efficiency (if raw detections available)
            if method == "Grid Analysis":
                raw_counts = [len(r.get('raw_detections', [])) for r in results.values()]
                if raw_counts:
                    total_raw = sum(raw_counts)
                    efficiency = (total_detections / total_raw * 100) if total_raw > 0 else 0
                    st.metric("Efficiency", f"{efficiency:.1f}%")
            
            st.markdown("---")  # Separator between methods
        else:
            st.info("No results for this method")
def download_all_batch_results():
    """Download comprehensive results from all tabs - FIXED implementation"""
    
    # Collect all batch results
    all_results = {}
    total_files = 0
    
    if 'entire_batch_results' in st.session_state and st.session_state.entire_batch_results:
        all_results['entire_image'] = st.session_state.entire_batch_results
        total_files += len(st.session_state.entire_batch_results)
    
    if 'grid_batch_results' in st.session_state and st.session_state.grid_batch_results:
        all_results['grid_analysis'] = st.session_state.grid_batch_results
        total_files += len(st.session_state.grid_batch_results)
    
    if 'roi_batch_results' in st.session_state and st.session_state.roi_batch_results:
        all_results['roi_analysis'] = st.session_state.roi_batch_results
        total_files += len(st.session_state.roi_batch_results)
    
    if not all_results:
        st.warning("No batch results to download")
        return
    
    try:
        import zipfile
        import io
        import csv
        import os
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Create comprehensive summary
            summary_output = io.StringIO()
            summary_writer = csv.writer(summary_output)
            summary_writer.writerow(['Method', 'Image', 'Total_Tips', 'Details'])
            
            for method, results in all_results.items():
                for image_path, result in results.items():
                    detections = result.get('detections', [])
                    image_name = result.get('image_name', os.path.basename(image_path))
                    
                    summary_writer.writerow([
                        method.replace('_', ' ').title(),
                        image_name,
                        len(detections),
                        result.get('method', method)
                    ])
                    
                    # Individual method CSV
                    base_name = os.path.splitext(image_name)[0]
                    coord_output = io.StringIO()
                    coord_writer = csv.writer(coord_output)
                    coord_writer.writerow(['x', 'y', 'confidence', 'method', 'type'])
                    
                    for detection in detections:
                        coord_writer.writerow([
                            detection['x'],
                            detection['y'],
                            detection.get('conf', 1.0),
                            detection.get('method', method),
                            'manual' if detection.get('manual', False) else 'automatic'
                        ])
                    
                    zip_file.writestr(f"{method}/{base_name}_coordinates.csv", coord_output.getvalue())
            
            zip_file.writestr("comprehensive_summary.csv", summary_output.getvalue())
        
        zip_content = zip_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Comprehensive Results",
            data=zip_content,
            file_name="comprehensive_batch_results.zip",
            mime="application/zip",
            key="download_comprehensive_results"
        )
        
        st.success(f"Download includes results from {len(all_results)} methods with {total_files} total files")
        
    except Exception as e:
        st.error(f"Error creating comprehensive download: {str(e)}")
def clear_all_batch_results():
    """Clear batch results from all tabs with confirmation"""
    
    # Show confirmation
    if st.button("⚠️ Confirm Clear All Batch Results", key="confirm_clear_all"):
        batch_keys = ['entire_batch_results', 'grid_batch_results', 'roi_batch_results']
        
        cleared_count = 0
        for key in batch_keys:
            if key in st.session_state and st.session_state[key]:
                cleared_count += len(st.session_state[key])
                del st.session_state[key]
        
        st.success(f"Cleared {cleared_count} batch results from all tabs")
        st.rerun()
    else:
        st.warning("Click 'Confirm' button above to clear all batch results")

def show_method_comparison():
    """Show enhanced comparison between different detection methods"""
    
    st.markdown("### 🔍 Advanced Method Comparison")
    
    # Collect results for comparison
    methods_data = {}
    
    if 'entire_batch_results' in st.session_state and st.session_state.entire_batch_results:
        methods_data['Entire Image'] = st.session_state.entire_batch_results
    
    if 'grid_batch_results' in st.session_state and st.session_state.grid_batch_results:
        methods_data['Grid Analysis'] = st.session_state.grid_batch_results
    
    if 'roi_batch_results' in st.session_state and st.session_state.roi_batch_results:
        methods_data['ROI Analysis'] = st.session_state.roi_batch_results
    
    if len(methods_data) < 2:
        st.warning("Need results from at least 2 methods to compare")
        st.info("Run batch processing on multiple tabs to enable comparison")
        return
    
    # Create enhanced comparison visualization
    create_enhanced_method_comparison_chart(methods_data)

def create_enhanced_method_comparison_chart(methods_data: dict):
    """Create enhanced comparison chart between methods - FIXED sidebar columns issue"""
    
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    # Prepare data for comparison
    comparison_data = []
    detection_distributions = {}
    
    for method, results in methods_data.items():
        if results:
            detection_counts = [len(r.get('detections', [])) for r in results.values()]
            
            if detection_counts:
                comparison_data.append({
                    'Method': method,
                    'Mean Detections': np.mean(detection_counts),
                    'Std Detections': np.std(detection_counts),
                    'Min Detections': min(detection_counts),
                    'Max Detections': max(detection_counts),
                    'Total Images': len(results),
                    'Total Detections': sum(detection_counts),
                    'Median Detections': np.median(detection_counts)
                })
                
                detection_distributions[method] = detection_counts
    
    if not comparison_data:
        st.info("No data available for comparison")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # FIXED: Create subplots with correct specs for pie chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Detections per Image', 'Detection Distribution', 
                       'Total Detections by Method', 'Detection Range'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "xy"}]]
    )
    
    # 1. Bar chart with error bars
    fig.add_trace(
        go.Bar(
            x=df['Method'],
            y=df['Mean Detections'],
            error_y=dict(type='data', array=df['Std Detections']),
            name='Average Detections',
            text=df['Mean Detections'].round(1),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # 2. Box plots for distribution
    for method, counts in detection_distributions.items():
        fig.add_trace(
            go.Box(
                y=counts,
                name=method,
                boxpoints='outliers',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Total detections pie chart
    fig.add_trace(
        go.Pie(
            labels=df['Method'],
            values=df['Total Detections'],
            name="Total Detections",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Range comparison
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['Method'], row['Method']],
                y=[row['Min Detections'], row['Max Detections']],
                mode='lines+markers',
                name=f"{row['Method']} Range",
                line=dict(width=4),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Comprehensive Method Comparison Analysis",
        height=800,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed comparison table
    st.markdown("#### 📋 Detailed Performance Metrics")
    
    # Format the dataframe for better display
    display_df = df.copy()
    for col in ['Mean Detections', 'Std Detections', 'Median Detections']:
        display_df[col] = display_df[col].round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Statistical insights - FIXED: No columns in sidebar
    st.markdown("#### 📈 Statistical Insights")
    
    best_avg = df.loc[df['Mean Detections'].idxmax(), 'Method']
    most_consistent = df.loc[df['Std Detections'].idxmin(), 'Method']
    highest_total = df.loc[df['Total Detections'].idxmax(), 'Method']
    
    # FIXED: Use individual metrics instead of columns
    st.metric("Best Average", best_avg, f"{df['Mean Detections'].max():.1f} tips/image")
    st.metric("Most Consistent", most_consistent, f"±{df['Std Detections'].min():.1f} std dev")
    st.metric("Highest Total", highest_total, f"{df['Total Detections'].max()} total tips")