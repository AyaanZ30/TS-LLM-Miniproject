import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.anomaly_detector import AdvancedAnomalyDetector
from src.forecasting import AdvancedForecaster
from src.vector_db import AnomalyVectorDB
from src.llm_analyzer import LLMAnalyzer
from src.visualizations import AdvancedVisualizations

# Page configuration
st.set_page_config(
    page_title="Advanced System Monitoring Dashboard",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SystemMonitoringDashboard:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.forecaster = AdvancedForecaster()
        self.vector_db = AnomalyVectorDB()
        self.llm_analyzer = LLMAnalyzer(api_key = 'xyz')
        self.visualizer = AdvancedVisualizations()
        
        # Sample data generation (replace with your actual data source)
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample system monitoring data"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'CPU Usage (%)': np.random.normal(45, 15, n_samples).clip(0, 100),
            'CPU Temperature (°C)': np.random.normal(65, 10, n_samples).clip(40, 100),
            'Clock Speed (GHz)': np.random.normal(3.5, 0.5, n_samples).clip(2.0, 4.5),
            'Cache Miss Rate (%)': np.random.normal(15, 5, n_samples).clip(5, 30),
            'Power Consumption (W)': np.random.normal(85, 20, n_samples).clip(40, 150),
            'Disk Read Speed (MB/s)': np.random.normal(350, 100, n_samples).clip(100, 600),
            'Disk Write Speed (MB/s)': np.random.normal(250, 80, n_samples).clip(50, 500)
        }
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, 50, replace=False)
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                data['CPU Usage (%)'][idx] = np.random.uniform(90, 100)
                data['CPU Temperature (°C)'][idx] = np.random.uniform(85, 100)
            else:
                data['Disk Read Speed (MB/s)'][idx] = np.random.uniform(10, 50)
                data['Disk Write Speed (MB/s)'][idx] = np.random.uniform(5, 30)
        
        df = pd.DataFrame(data)
        return self.data_processor.create_derived_features(df)
    
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">🖥️ Advanced System Monitoring Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Configuration")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload System Metrics CSV", 
            type=['csv'],
            help="Upload your system monitoring data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                self.df = self.data_processor.load_and_clean_data(uploaded_file)
                ##############################################
                self.df = self.data_processor.prepare_time_series_data(self.df)   
                #############################################
                self.df = self.data_processor.create_derived_features(self.df)
                st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                self.df = self.sample_data
                st.sidebar.info("Using sample data instead.")
        else:
            self.df = self.sample_data
            st.sidebar.info("Using sample system data. Upload a CSV file to use your own data.")
        
        # Analysis parameters
        st.sidebar.subheader("Analysis Parameters")
        contamination = st.sidebar.slider(
            "Anomaly Contamination", 
            min_value=0.01, 
            max_value=0.3, 
            value=0.1,
            help="Expected proportion of anomalies in the data"
        )
        
        forecast_steps = st.sidebar.slider(
            "Forecast Steps", 
            min_value=5, 
            max_value=50, 
            value=10,
            help="Number of future steps to forecast"
        )
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard", 
            "🔍 Anomaly Analysis", 
            "📈 Forecasting",
            "🤖 AI Insights",
            "💾 Vector DB",
            "📋 Reports"
        ])
        
        with tab1:
            self._show_dashboard()
        
        with tab2:
            self._show_anomaly_analysis(contamination)
        
        with tab3:
            self._show_forecasting(forecast_steps)
        
        with tab4:
            self._show_ai_insights()
        
        with tab5:
            self._show_vector_db_interface()
        
        with tab6:
            self._show_reports()
    
    def _show_dashboard(self):
        st.header("Real-time System Monitoring Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_cpu = self.df['CPU Usage (%)'].mean()
            st.metric("Average CPU Usage", f"{avg_cpu:.1f}%")
        
        with col2:
            max_temp = self.df['CPU Temperature (°C)'].max()
            st.metric("Max CPU Temperature", f"{max_temp:.1f}°C")
        
        with col3:
            avg_power = self.df['Power Consumption (W)'].mean()
            st.metric("Average Power", f"{avg_power:.1f}W")
        
        with col4:
            total_anomalies = len(self._detect_anomalies()['combined']['anomalies'])
            st.metric("Total Anomalies", total_anomalies)
        
        # Main dashboard visualization
        anomalies = self._detect_anomalies()
        fig = self.visualizer.create_system_dashboard(self.df, anomalies)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("System Metrics Correlation")
        corr_fig = self.visualizer.create_correlation_matrix(self.df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    def _detect_anomalies(self):
        """Cache anomaly detection results"""
        if not hasattr(self, '_cached_anomalies'):
            self._cached_anomalies = self.anomaly_detector.comprehensive_anomaly_detection(self.df)
        return self._cached_anomalies
    
    def _show_anomaly_analysis(self, contamination):
        st.header("Advanced Anomaly Analysis")
        
        with st.spinner("Detecting anomalies..."):
            anomalies = self._detect_anomalies()
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        
        total_anomalies = np.sum(anomalies['combined']['anomalies'])
        anomaly_percentage = (total_anomalies / len(self.df)) * 100
        
        with col1:
            st.metric("Total Anomalies", total_anomalies)
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
        with col3:
            avg_confidence = np.mean(anomalies['combined']['confidence'])
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        # Anomaly visualization
        fig = self.visualizer.create_anomaly_analysis_chart(self.df, anomalies)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed anomaly table
        st.subheader("Detailed Anomaly Report")
        if total_anomalies > 0:
            anomaly_indices = np.where(anomalies['combined']['anomalies'] == 1)[0]
            anomaly_data = []
            
            for idx in anomaly_indices[:20]:  # Show first 20
                row = self.df.iloc[idx].copy()
                row['Anomaly_Score'] = anomalies['combined']['anomaly_score'][idx]
                row['Confidence'] = anomalies['combined']['confidence'][idx]
                anomaly_data.append(row)
            
            anomaly_df = pd.DataFrame(anomaly_data)
            st.dataframe(anomaly_df, use_container_width=True)
            
            # Store anomalies in vector DB
            if st.button("Store Anomalies in Vector DB"):
                self._store_detected_anomalies(anomalies)
                st.success("Anomalies stored in vector database!")
        else:
            st.info("No anomalies detected in the current dataset.")
    
    def _store_detected_anomalies(self, anomalies):
        """Store detected anomalies in vector database"""
        anomaly_indices = np.where(anomalies['combined']['anomalies'] == 1)[0]
        
        for idx in anomaly_indices:
            # Find the metric with highest deviation
            max_dev_metric = None
            max_dev = 0
            
            z_scores_df = pd.DataFrame(anomalies['statistical']['z_scores'])
            for col in self.df.select_dtypes(include=[np.number]).columns:
                if col in z_scores_df.columns:
                    dev = z_scores_df.iloc[idx][col]
                    if abs(dev) > max_dev:
                        max_dev = abs(dev)  
                        max_dev_metric = col
            
            anomaly_data = {
                'metric': max_dev_metric,
                'anomaly_type': 'combined_detection',
                'severity': 'high' if anomalies['combined']['confidence'][idx] > 0.7 else 'medium',
                'actual_value': float(self.df.iloc[idx][max_dev_metric]),
                'expected_value': float(self.df[max_dev_metric].mean()),
                'deviation': float(anomalies['combined']['anomaly_score'][idx]),
                'confidence': float(anomalies['combined']['confidence'][idx]),
                'additional_context': f"Detected by ensemble method at index {idx}"
            }
            
            self.vector_db.store_anomaly(anomaly_data)
    
    def _show_forecasting(self, steps):
        st.header("System Metrics Forecasting")
        
        # Metric selection
        target_metric = st.selectbox(
            "Select Metric to Forecast",
            options=self.df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        if target_metric:
            with st.spinner("Generating forecasts..."):
                series = self.df[target_metric]
                forecast_result = self.forecaster.multi_step_forecasting(
                    self.df, target_metric, steps
                )
            
            if forecast_result and forecast_result['best_forecast'] is not None:
                # Forecast visualization
                fig = self.visualizer.create_forecast_visualization(
                    series, forecast_result, f"{target_metric} Forecast"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast metrics
                st.subheader("Forecast Performance Metrics")
                metrics_df = pd.DataFrame(forecast_result['combined_metrics']).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Forecast values
                st.subheader("Forecasted Values")
                forecast_df = pd.DataFrame({
                    'Step': range(1, steps + 1),
                    'Forecasted_Value': forecast_result['best_forecast'].values
                })
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error("Forecasting failed for the selected metric.")
    
    def _show_ai_insights(self):
        st.header("AI-Powered System Insights")
        
        # Get recent anomalies for analysis
        recent_anomalies = self.vector_db.get_recent_anomalies(hours=24)
        
        if recent_anomalies:
            # Anomaly pattern analysis
            if st.button("Analyze Anomaly Patterns"):
                with st.spinner("Analyzing anomaly patterns with AI..."):
                    analysis = self.llm_analyzer.analyze_anomaly_patterns(recent_anomalies)
                
                st.subheader("Anomaly Pattern Analysis")
                st.write(analysis)
            
            # System insights
            if st.button("Generate System Insights"):
                with st.spinner("Generating system insights..."):
                    insights = self.llm_analyzer.generate_insights(self.df, recent_anomalies)
                
                st.subheader("System Performance Insights")
                st.write(insights)
            
            # Question answering
            st.subheader("Ask Questions About Anomalies")
            question = st.text_input("Enter your question about system anomalies:")
            
            if question:
                with st.spinner("Searching for relevant information..."):
                    # Search for relevant anomalies
                    relevant_anomalies = self.vector_db.search_anomalies(question, n_results=5)
                    answer = self.llm_analyzer.answer_question(question, relevant_anomalies)
                
                st.subheader("Answer")
                st.write(answer)
                
                if relevant_anomalies:
                    st.subheader("Relevant Anomaly Records")
                    for i, anomaly in enumerate(relevant_anomalies):
                        with st.expander(f"Anomaly Record {i+1}"):
                            st.write(anomaly['document'])
        else:
            st.info("No recent anomalies found in the database. Run anomaly detection first.")
    
    def _show_vector_db_interface(self):
        st.header("Vector Database Management")
        
        # Database statistics
        stats = self.vector_db.get_anomaly_statistics()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Anomalies", stats['total_anomalies'])
            with col2:
                st.metric("Avg Confidence", f"{stats['average_confidence']:.3f}")
            with col3:
                st.metric("Earliest", stats['time_range']['earliest'][:10])
            with col4:
                st.metric("Latest", stats['time_range']['latest'][:10])
            
            # Anomaly distribution
            st.subheader("Anomaly Distribution by Metric")
            metric_dist = pd.DataFrame.from_dict(
                stats['by_metric'], orient='index', columns=['Count']
            )
            st.bar_chart(metric_dist)
            
            # Search interface
            st.subheader("Search Anomalies")
            search_query = st.text_input("Enter search query:")
            if search_query:
                results = self.vector_db.search_anomalies(search_query, n_results=5)
                
                if results:
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Distance: {result['distance']:.3f})"):
                            st.write("**Document:**")
                            st.write(result['document'])
                            st.write("**Metadata:**")
                            st.json(result['metadata'])
                else:
                    st.info("No results found.")
        else:
            st.info("Vector database is empty. Run anomaly detection to store anomalies.")
    
    def _show_reports(self):
        st.header("Comprehensive System Reports")
        
        # Generate report
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Generating comprehensive system report..."):
                report = self._generate_comprehensive_report()
            
            st.subheader("System Health Report")
            st.write(report)
            
            # Download report
            st.download_button(
                "Download Report as PDF",
                data=report,
                file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive system health report"""
        anomalies = self._detect_anomalies()
        stats = self.vector_db.get_anomaly_statistics()
        
        report_lines = [
            "COMPREHENSIVE SYSTEM HEALTH REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SYSTEM OVERVIEW",
            "-" * 20,
            f"Total Data Points: {len(self.df)}",
            f"Monitoring Period: {len(self.df)} minutes",
            f"Metrics Monitored: {len(self.df.columns)}",
            "",
            "PERFORMANCE METRICS",
            "-" * 20,
        ]
        
        # Add key metrics
        for col in self.df.select_dtypes(include=[np.number]).columns:
            report_lines.append(
                f"{col}: Mean={self.df[col].mean():.2f}, "
                f"Std={self.df[col].std():.2f}, "
                f"Max={self.df[col].max():.2f}"
            )
        
        report_lines.extend([
            "",
            "ANOMALY ANALYSIS",
            "-" * 20,
            f"Total Anomalies Detected: {np.sum(anomalies['combined']['anomalies'])}",
            f"Anomaly Rate: {(np.sum(anomalies['combined']['anomalies']) / len(self.df)) * 100:.2f}%",
            "",
            "RECOMMENDATIONS",
            "-" * 20,
        ])
        
        # Add AI-generated recommendations if available
        recent_anomalies = self.vector_db.get_recent_anomalies(hours=24)
        if recent_anomalies:
            insights = self.llm_analyzer.generate_insights(self.df, recent_anomalies)
            report_lines.extend([insights])
        
        return "\n".join(report_lines)

def main():
    dashboard = SystemMonitoringDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()