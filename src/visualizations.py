import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List

class AdvancedVisualizations:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'anomaly': '#d62728',
            'forecast': '#2ca02c',
            'background': '#f8f9fa'
        }
    
    def create_system_dashboard(self, df: pd.DataFrame, 
                              anomalies: Dict = None) -> go.Figure:
        """Create comprehensive system monitoring dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'CPU Usage & Temperature', 
                'Disk Performance',
                'Power Consumption', 
                'System Efficiency Metrics',
                'Anomaly Distribution',
                'Performance Trends'
            ),
            specs=[
                [{"secondary_y": True}, {}],
                [{"secondary_y": False}, {}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add timestamp if not present
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        
        # 1. CPU Usage & Temperature
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['CPU Usage (%)'], 
                      name='CPU Usage', line=dict(color=self.color_palette['primary'])),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['CPU Temperature (°C)'], 
                      name='CPU Temp', line=dict(color=self.color_palette['secondary'])),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Disk Performance
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['Disk Read Speed (MB/s)'], 
                      name='Read Speed', line=dict(color='#17becf')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['Disk Write Speed (MB/s)'], 
                      name='Write Speed', line=dict(color='#bcbd22')),
            row=1, col=2
        )
        
        # 3. Power Consumption
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['Power Consumption (W)'], 
                      name='Power', line=dict(color='#e377c2')),
            row=2, col=1
        )
        
        # 4. System Efficiency (if derived features exist)
        if 'Performance_Efficiency' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['Timestamp'], y=df['Performance_Efficiency'], 
                          name='Perf Efficiency', line=dict(color='#7f7f7f')),
                row=2, col=2
            )
        
        # 5. Anomaly Distribution
        if anomalies and 'combined' in anomalies:
            anomaly_indices = np.where(anomalies['combined']['anomalies'] == 1)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Histogram(x=anomaly_indices, name='Anomaly Distribution',
                               marker_color=self.color_palette['anomaly']),
                    row=3, col=1
                )
        
        # 6. Performance Trends
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['Clock Speed (GHz)'], 
                      name='Clock Speed', line=dict(color='#9467bd')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Comprehensive System Monitoring Dashboard",
            showlegend=True,
            template="plotly_white",
            font=dict(size=10)
        )
        
        return fig
    
    def create_anomaly_analysis_chart(self, df: pd.DataFrame, 
                                   anomalies: Dict) -> go.Figure:
        """Create detailed anomaly analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Anomaly Timeline',
                'Anomaly Severity Distribution',
                'Metric-wise Anomalies',
                'Anomaly Clusters'
            ),
            specs=[
                [{"secondary_y": False}, {}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Add timestamp if not present
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        
        # 1. Anomaly Timeline
        if 'combined' in anomalies:
            anomaly_scores = anomalies['combined']['anomaly_score']
            fig.add_trace(
                go.Scatter(x=df['Timestamp'], y=anomaly_scores, 
                          name='Anomaly Score', line=dict(color=self.color_palette['anomaly'])),
                row=1, col=1
            )
            
            # Highlight actual anomalies
            anomaly_indices = np.where(anomalies['combined']['anomalies'] == 1)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Scatter(x=df['Timestamp'].iloc[anomaly_indices], 
                              y=anomaly_scores[anomaly_indices],
                              mode='markers', name='Detected Anomalies',
                              marker=dict(color='red', size=8, symbol='x')),
                    row=1, col=1
                )
        
        # 2. Anomaly Severity Distribution
        if 'combined' in anomalies:
            fig.add_trace(
                go.Histogram(x=anomalies['combined']['anomaly_score'], 
                           name='Anomaly Score Distribution',
                           marker_color=self.color_palette['primary']),
                row=1, col=2
            )
        
        # 3. Metric-wise Anomalies
        if 'statistical' in anomalies:
            metric_anomalies = anomalies['statistical']['anomalies'].sum()
            fig.add_trace(
                go.Bar(x=metric_anomalies.index, y=metric_anomalies.values,
                      name='Anomalies per Metric',
                      marker_color=self.color_palette['secondary']),
                row=2, col=1
            )
        
        # 4. Anomaly Clusters (if available)
        if 'isolation_forest' in anomalies:
            scores = anomalies['isolation_forest']['scores']
            fig.add_trace(
                go.Histogram(x=scores, name='Isolation Forest Scores',
                           marker_color='#ff7f0e'),
                row=2, col=2
            )
        
        fig.update_layout(height=700, title_text="Advanced Anomaly Analysis")
        return fig
    
    def create_forecast_visualization(self, historical_data: pd.Series,
                                   forecast_result: Dict,
                                   title: str = "Forecast Results") -> go.Figure:
        """Create forecast visualization with confidence intervals"""
        fig = go.Figure()
        
        ###########################################
        historical_data.index = pd.to_datetime(historical_data.index)
        ###########################################
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            name='Historical',
            line=dict(color=self.color_palette['primary'])
        ))
        
        # Forecast
        if forecast_result and 'best_forecast' in forecast_result:
            forecast = forecast_result['best_forecast']
            forecast_index = pd.date_range(
                start=historical_data.index[-1] + pd.Timedelta(minutes=1),
                periods=len(forecast),
                freq='1min'
            )
            
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast.values,
                name='Forecast',
                line=dict(color=self.color_palette['forecast'], dash='dash')
            ))
            
            # Confidence intervals if available
            best_model = forecast_result['best_model']
            if best_model and 'confidence_intervals' in forecast_result['all_models'][best_model]:
                ci = forecast_result['all_models'][best_model]['confidence_intervals']
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=ci.iloc[:, 0],
                    fill=None,
                    mode='lines',
                    line=dict(color='gray', width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=ci.iloc[:, 1],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='gray', width=0),
                    name='Confidence Interval'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template="plotly_white"
        )
        
        return fig
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="System Metrics Correlation Matrix",
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            height=600
        )
        
        return fig