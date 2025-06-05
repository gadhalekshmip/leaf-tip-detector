# components/result_panel.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from utils.file_utils import save_detections_to_csv, save_image_with_detections
import os
class ResultPanel:
    """Component for displaying comprehensive detection results and analytics"""
    
    def __init__(self):
        self.current_detections = []
        self.batch_results = {}
        self.display_mode = "summary"
        
    def display_results_panel(self,
                            detections: List[Dict],
                            key: str = "result_panel",
                            show_statistics: bool = True,
                            show_charts: bool = True,
                            show_downloads: bool = True,
                            batch_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Display comprehensive results panel
        
        Args:
            detections: Current detection results
            key: Unique key for the component
            show_statistics: Show statistical summary
            show_charts: Show visualization charts
            show_downloads: Show download options
            batch_results: Optional batch processing results
            
        Returns:
            Dict with user interactions and selected options
        """
        
        self.current_detections = detections
        self.batch_results = batch_results or {}
        
        st.markdown("### 📊 Results Panel")
        
        # Result mode selector
        mode = st.radio(
            "Display Mode",
            options=["summary", "detailed", "comparison"],
            format_func=lambda x: {
                "summary": "📋 Summary",
                "detailed": "🔍 Detailed Analysis", 
                "comparison": "📊 Comparison"
            }[x],
            horizontal=True,
            key=f"{key}_mode"
        )
        
        self.display_mode = mode
        
        # Display content based on mode
        results = {}
        
        if mode == "summary":
            results.update(self._display_summary_mode(key, show_statistics, show_downloads))
        elif mode == "detailed":
            results.update(self._display_detailed_mode(key, show_charts, show_downloads))
        elif mode == "comparison":
            results.update(self._display_comparison_mode(key))
        
        return results
    
    def _display_summary_mode(self, key: str, show_statistics: bool, show_downloads: bool) -> Dict:
        """Display summary results mode"""
        
        results = {}
        
        # Quick statistics
        if show_statistics:
            self._display_quick_statistics()
        
        # Detection count with visual indicator
        self._display_detection_count_visual()
        
        # Method breakdown if multiple methods
        if self._has_multiple_methods():
            self._display_method_breakdown()
        
        # Download options
        if show_downloads:
            results.update(self._display_download_options(key))
        
        # Batch summary if available
        if self.batch_results:
            self._display_batch_summary()
        
        return results
    
    def _display_detailed_mode(self, key: str, show_charts: bool, show_downloads: bool) -> Dict:
        """Display detailed analysis mode"""
        
        results = {}
        
        # Detailed statistics table
        self._display_detailed_statistics()
        
        # Charts and visualizations
        if show_charts:
            self._display_analysis_charts()
        
        # Detection list with details
        self._display_detection_list(key)
        
        # Advanced download options
        if show_downloads:
            results.update(self._display_advanced_download_options(key))
        
        return results
    
    def _display_comparison_mode(self, key: str) -> Dict:
        """Display comparison mode for multiple methods/batches"""
        
        results = {}
        
        if not self.batch_results:
            st.info("Comparison mode requires batch processing results")
            return results
        
        # Method comparison
        self._display_method_comparison()
        
        # Performance metrics comparison
        self._display_performance_comparison()
        
        # Batch vs batch comparison
        self._display_batch_comparison()
        
        return results
    
    def _display_quick_statistics(self):
        """Display quick statistics overview"""
        
        if not self.current_detections:
            st.info("No detections to analyze")
            return
        
        total = len(self.current_detections)
        manual_count = sum(1 for d in self.current_detections if d.get('manual', False))
        auto_count = total - manual_count
        
        # Confidence statistics
        auto_detections = [d for d in self.current_detections if not d.get('manual', False)]
        confidences = [d.get('conf', 1.0) for d in auto_detections]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Detections",
                total,
                delta=f"+{total}" if total > 0 else None
            )
        
        with col2:
            st.metric(
                "Automatic",
                auto_count,
                delta=f"{auto_count/total*100:.1f}%" if total > 0 else None
            )
        
        with col3:
            st.metric(
                "Manual",
                manual_count,
                delta=f"{manual_count/total*100:.1f}%" if total > 0 else None
            )
        
        with col4:
            if confidences:
                avg_conf = np.mean(confidences)
                st.metric(
                    "Avg Confidence",
                    f"{avg_conf:.3f}",
                    delta=f"Range: {min(confidences):.2f}-{max(confidences):.2f}"
                )
            else:
                st.metric("Avg Confidence", "N/A")
    
    def _display_detection_count_visual(self):
        """Display detection count with visual progress indicator"""
        
        total = len(self.current_detections)
        
        # Create a visual indicator based on detection count
        if total == 0:
            st.error("🔍 No detections found")
        elif total < 10:
            st.warning(f"🌱 {total} leaf tips detected - Low density")
        elif total < 50:
            st.success(f"🌿 {total} leaf tips detected - Normal density")
        elif total < 100:
            st.info(f"🍃 {total} leaf tips detected - High density")
        else:
            st.warning(f"🌳 {total} leaf tips detected - Very high density")
        
        # Progress bar visualization (for visual appeal)
        max_expected = 100  # Adjust based on typical use case
        progress = min(total / max_expected, 1.0)
        st.progress(progress)
    
    def _has_multiple_methods(self) -> bool:
        """Check if detections come from multiple methods"""
        
        methods = set()
        for detection in self.current_detections:
            method = detection.get('method', 'unknown')
            if detection.get('manual', False):
                method = 'manual'
            methods.add(method)
        
        return len(methods) > 1
    
    def _display_method_breakdown(self):
        """Display breakdown by detection method"""
        
        st.markdown("**🔬 Detection Methods:**")
        
        method_counts = {}
        for detection in self.current_detections:
            method = detection.get('method', 'unknown')
            if detection.get('manual', False):
                method = 'manual'
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Display as columns
        methods = list(method_counts.keys())
        if len(methods) <= 4:
            cols = st.columns(len(methods))
            for i, (method, count) in enumerate(method_counts.items()):
                with cols[i]:
                    percentage = count / len(self.current_detections) * 100
                    st.metric(
                        method.replace('_', ' ').title(),
                        count,
                        delta=f"{percentage:.1f}%"
                    )
        else:
            # Too many methods, use a different layout
            for method, count in method_counts.items():
                percentage = count / len(self.current_detections) * 100
                st.write(f"**{method.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)")
    
    def _display_download_options(self, key: str) -> Dict:
        """Display basic download options"""
        
        results = {}
        
        if not self.current_detections:
            return results
        
        st.markdown("**💾 Download Options:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 CSV File", key=f"{key}_download_csv"):
                results['download_csv'] = True
        
        with col2:
            if st.button("🖼️ Annotated Image", key=f"{key}_download_image"):
                results['download_image'] = True
        
        with col3:
            if st.button("📦 Full Report", key=f"{key}_download_report"):
                results['download_report'] = True
        
        return results
    
    def _display_batch_summary(self):
        """Display batch processing summary"""
        
        if not self.batch_results:
            return
        
        st.markdown("**📦 Batch Processing Summary:**")
        
        total_images = len(self.batch_results)
        total_detections = sum(len(result.get('detections', [])) for result in self.batch_results.values())
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Images Processed", total_images)
        
        with col2:
            st.metric("Total Detections", total_detections)
        
        with col3:
            st.metric("Average per Image", f"{avg_detections:.1f}")
    
    def _display_detailed_statistics(self):
        """Display detailed statistical analysis"""
        
        if not self.current_detections:
            st.info("No detections for detailed analysis")
            return
        
        st.markdown("**📈 Detailed Statistics**")
        
        # Create statistics dataframe
        stats_data = self._calculate_detailed_statistics()
        
        # Display as table
        df = pd.DataFrame([stats_data])
        st.dataframe(df, use_container_width=True)
        
        # Additional analysis
        self._display_spatial_analysis()
        self._display_confidence_analysis()
    
    def _calculate_detailed_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        
        total = len(self.current_detections)
        manual_count = sum(1 for d in self.current_detections if d.get('manual', False))
        auto_count = total - manual_count
        
        # Confidence statistics
        confidences = [d.get('conf', 1.0) for d in self.current_detections if not d.get('manual', False)]
        
        # Spatial statistics
        x_coords = [d['x'] for d in self.current_detections]
        y_coords = [d['y'] for d in self.current_detections]
        
        stats = {
            'Total Detections': total,
            'Automatic': auto_count,
            'Manual': manual_count,
            'Auto Percentage': f"{auto_count/total*100:.1f}%" if total > 0 else "0%",
        }
        
        if confidences:
            stats.update({
                'Avg Confidence': f"{np.mean(confidences):.3f}",
                'Min Confidence': f"{min(confidences):.3f}",
                'Max Confidence': f"{max(confidences):.3f}",
                'Std Confidence': f"{np.std(confidences):.3f}"
            })
        
        if x_coords and y_coords:
            stats.update({
                'X Range': f"{min(x_coords):.0f} - {max(x_coords):.0f}",
                'Y Range': f"{min(y_coords):.0f} - {max(y_coords):.0f}",
                'Center X': f"{np.mean(x_coords):.1f}",
                'Center Y': f"{np.mean(y_coords):.1f}"
            })
        
        return stats
    
    def _display_spatial_analysis(self):
        """Display spatial distribution analysis"""
        
        if len(self.current_detections) < 2:
            return
        
        st.markdown("**🗺️ Spatial Distribution**")
        
        x_coords = [d['x'] for d in self.current_detections]
        y_coords = [d['y'] for d in self.current_detections]
        
        # Create scatter plot of detection positions
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=8,
                color=[d.get('conf', 1.0) for d in self.current_detections],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=[f"Conf: {d.get('conf', 1.0):.3f}" for d in self.current_detections],
            hovertemplate="X: %{x}<br>Y: %{y}<br>%{text}<extra></extra>",
            name="Detections"
        ))
        
        fig.update_layout(
            title="Detection Spatial Distribution",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            height=400,
            yaxis=dict(scaleanchor="x", scaleratio=1)  # Maintain aspect ratio
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_confidence_analysis(self):
        """Display confidence distribution analysis"""
        
        confidences = [d.get('conf', 1.0) for d in self.current_detections if not d.get('manual', False)]
        
        if not confidences:
            return
        
        st.markdown("**📊 Confidence Distribution**")
        
        # Create histogram
        fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
        
        fig.update_layout(
            title="Detection Confidence Distribution",
            xaxis_title="Confidence",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Conf (<0.5)", sum(1 for c in confidences if c < 0.5))
        with col2:
            st.metric("Medium Conf (0.5-0.8)", sum(1 for c in confidences if 0.5 <= c < 0.8))
        with col3:
            st.metric("High Conf (≥0.8)", sum(1 for c in confidences if c >= 0.8))
    
    def _display_analysis_charts(self):
        """Display various analysis charts"""
        
        if not self.current_detections:
            return
        
        # Method comparison chart if multiple methods
        if self._has_multiple_methods():
            self._create_method_comparison_chart()
        
        # Temporal analysis if batch data available
        if self.batch_results:
            self._create_batch_trend_chart()
    
    def _create_method_comparison_chart(self):
        """Create chart comparing different detection methods"""
        
        method_data = {}
        for detection in self.current_detections:
            method = detection.get('method', 'unknown')
            if detection.get('manual', False):
                method = 'manual'
            
            if method not in method_data:
                method_data[method] = []
            method_data[method].append(detection.get('conf', 1.0))
        
        # Create box plot
        fig = go.Figure()
        
        for method, confidences in method_data.items():
            fig.add_trace(go.Box(
                y=confidences,
                name=method.replace('_', ' ').title(),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Confidence Distribution by Method",
            yaxis_title="Confidence",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_batch_trend_chart(self):
        """Create trend chart for batch processing results"""
        
        if not self.batch_results:
            return
        
        # Extract batch data
        image_names = []
        detection_counts = []
        
        for image_path, result in self.batch_results.items():
            image_names.append(os.path.basename(image_path))
            detection_counts.append(len(result.get('detections', [])))
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(image_names))),
            y=detection_counts,
            mode='lines+markers',
            name='Detection Count',
            text=image_names,
            hovertemplate="Image: %{text}<br>Detections: %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Detection Count Trend Across Batch",
            xaxis_title="Image Index",
            yaxis_title="Detection Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_detection_list(self, key: str):
        """Display detailed list of all detections"""
        
        if not self.current_detections:
            return
        
        st.markdown("**📋 Detection Details**")
        
        # Create dataframe
        detection_data = []
        for i, detection in enumerate(self.current_detections):
            detection_data.append({
                'ID': i + 1,
                'X': f"{detection['x']:.1f}",
                'Y': f"{detection['y']:.1f}",
                'Confidence': f"{detection.get('conf', 1.0):.3f}",
                'Method': detection.get('method', 'unknown'),
                'Type': 'Manual' if detection.get('manual', False) else 'Auto',
                'Cell': str(detection.get('cell', '')) if detection.get('cell') else ''
            })
        
        df = pd.DataFrame(detection_data)
        
        # Display with filtering options
        col1, col2 = st.columns(2)
        
        with col1:
            method_filter = st.selectbox(
                "Filter by Method",
                options=['All'] + list(df['Method'].unique()),
                key=f"{key}_method_filter"
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                options=['All', 'Auto', 'Manual'],
                key=f"{key}_type_filter"
            )
        
        # Apply filters
        filtered_df = df.copy()
        if method_filter != 'All':
            filtered_df = filtered_df[filtered_df['Method'] == method_filter]
        if type_filter != 'All':
            filtered_df = filtered_df[filtered_df['Type'] == type_filter]
        
        # Display table
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Selection for detailed view
        if len(filtered_df) > 0:
            selected_row = st.selectbox(
                "Select detection for details",
                options=filtered_df['ID'].tolist(),
                format_func=lambda x: f"Detection {x}",
                key=f"{key}_detection_select"
            )
            
            if selected_row:
                detection = self.current_detections[selected_row - 1]
                self._display_single_detection_details(detection)
    
    def _display_single_detection_details(self, detection: Dict):
        """Display details for a single detection"""
        
        st.markdown("**🔍 Detection Details**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Position:** ({detection['x']:.2f}, {detection['y']:.2f})")
            st.write(f"**Confidence:** {detection.get('conf', 1.0):.4f}")
            st.write(f"**Method:** {detection.get('method', 'unknown')}")
        
        with col2:
            st.write(f"**Type:** {'Manual' if detection.get('manual', False) else 'Automatic'}")
            if 'cell' in detection:
                st.write(f"**Grid Cell:** {detection['cell']}")
            if 'roi_coords' in detection:
                st.write(f"**ROI:** {detection['roi_coords']}")
    
    def _display_advanced_download_options(self, key: str) -> Dict:
        """Display advanced download options"""
        
        results = {}
        
        st.markdown("**💾 Advanced Downloads**")
        
        with st.expander("Download Options", expanded=True):
            
            # CSV options
            st.markdown("**CSV Export:**")
            col1, col2 = st.columns(2)
            
            with col1:
                include_metadata = st.checkbox("Include Metadata", value=True, key=f"{key}_csv_meta")
                include_confidence = st.checkbox("Include Confidence", value=True, key=f"{key}_csv_conf")
            
            with col2:
                include_method = st.checkbox("Include Method Info", value=True, key=f"{key}_csv_method")
                include_cell_info = st.checkbox("Include Cell Info", value=True, key=f"{key}_csv_cell")
            
            if st.button("📊 Download Custom CSV", key=f"{key}_custom_csv"):
                results['download_custom_csv'] = {
                    'include_metadata': include_metadata,
                    'include_confidence': include_confidence,
                    'include_method': include_method,
                    'include_cell_info': include_cell_info
                }
            
            # Image options
            st.markdown("**Image Export:**")
            col1, col2 = st.columns(2)
            
            with col1:
                image_format = st.selectbox("Format", ["PNG", "JPEG"], key=f"{key}_img_format")
                image_quality = st.slider("Quality", 50, 100, 95, key=f"{key}_img_quality")
            
            with col2:
                include_grid = st.checkbox("Include Grid", value=False, key=f"{key}_img_grid")
                include_labels = st.checkbox("Include Labels", value=False, key=f"{key}_img_labels")
            
            if st.button("🖼️ Download Custom Image", key=f"{key}_custom_image"):
                results['download_custom_image'] = {
                    'format': image_format,
                    'quality': image_quality,
                    'include_grid': include_grid,
                    'include_labels': include_labels
                }
        
        return results
    
    def _display_method_comparison(self):
        """Display comparison between different methods"""
        
        if not self.batch_results:
            st.info("No batch results for method comparison")
            return
        
        st.markdown("**🔍 Method Performance Comparison**")
        
        # Group results by method
        method_performance = {}
        
        for image_path, result in self.batch_results.items():
            method = result.get('method', 'unknown')
            detections = result.get('detections', [])
            
            if method not in method_performance:
                method_performance[method] = []
            
            method_performance[method].append(len(detections))
        
        # Create comparison chart
        fig = go.Figure()
        
        for method, counts in method_performance.items():
            fig.add_trace(go.Box(
                y=counts,
                name=method.replace('_', ' ').title(),
                boxpoints='all'
            ))
        
        fig.update_layout(
            title="Detection Count Distribution by Method",
            yaxis_title="Detection Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics table
        comparison_data = []
        for method, counts in method_performance.items():
            comparison_data.append({
                'Method': method.replace('_', ' ').title(),
                'Images': len(counts),
                'Total Detections': sum(counts),
                'Avg per Image': f"{np.mean(counts):.1f}",
                'Std Dev': f"{np.std(counts):.1f}",
                'Min': min(counts),
                'Max': max(counts)
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _display_performance_comparison(self):
        """Display performance metrics comparison"""
        
        st.markdown("**⚡ Performance Metrics**")
        
        # This would include timing, accuracy metrics, etc.
        # Placeholder for now
        st.info("Performance metrics will be displayed here when available")
    
    def _display_batch_comparison(self):
        """Display batch-to-batch comparison"""
        
        st.markdown("**📦 Batch Analysis**")
        
        # This would compare different batch processing runs
        # Placeholder for now
        st.info("Batch comparison features will be available here")
    
    def generate_report(self, include_charts: bool = True) -> Dict[str, Any]:
        """Generate comprehensive report data"""
        
        report = {
            'summary': self._calculate_detailed_statistics(),
            'detections': self.current_detections,
            'batch_results': self.batch_results
        }
        
        if include_charts:
            # Add chart data
            report['charts'] = {
                'method_breakdown': self._get_method_breakdown_data(),
                'confidence_distribution': self._get_confidence_distribution_data(),
                'spatial_distribution': self._get_spatial_distribution_data()
            }
        
        return report
    
    def _get_method_breakdown_data(self) -> Dict:
        """Get method breakdown data for charts"""
        
        method_counts = {}
        for detection in self.current_detections:
            method = detection.get('method', 'unknown')
            if detection.get('manual', False):
                method = 'manual'
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return method_counts
    
    def _get_confidence_distribution_data(self) -> List[float]:
        """Get confidence distribution data"""
        
        return [d.get('conf', 1.0) for d in self.current_detections if not d.get('manual', False)]
    
    def _get_spatial_distribution_data(self) -> Dict:
        """Get spatial distribution data"""
        
        return {
            'x_coords': [d['x'] for d in self.current_detections],
            'y_coords': [d['y'] for d in self.current_detections],
            'confidences': [d.get('conf', 1.0) for d in self.current_detections]
        }

def create_result_panel() -> ResultPanel:
    """Factory function to create ResultPanel instance"""
    return ResultPanel()