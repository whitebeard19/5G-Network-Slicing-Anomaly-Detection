import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

class NetworkMonitoringDashboard:
    def __init__(self):
        # Load data and initialize
        self.load_data()
        self.setup_alerts()
        
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load main datasets
            self.network_data = pd.read_csv('Dataset/network_data.csv')
            self.results = pd.read_csv('results/anomaly_detection_results.csv')
            self.analysis = pd.read_csv('results/anomaly_analysis_report.csv')
            
            # Process timestamps
            self.network_data['timestamp'] = pd.to_datetime('now') - pd.to_timedelta(
                self.network_data.index * 5, unit='minutes')
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
    def setup_alerts(self):
        """Initialize alert thresholds"""
        self.alert_thresholds = {
            'critical': 20.0,  # Red alert
            'warning': 10.0,   # Yellow alert
            'normal': 5.0      # Green status
        }

    def run_dashboard(self):
        """Main dashboard layout and functionality"""
        # Page config
        st.set_page_config(
            page_title="Network Slice Monitoring",
            page_icon="🌐",
            layout="wide"
        )
        
        # Header
        st.title("🌐 Network Slice Monitoring Dashboard")
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main dashboard sections
        self.render_key_metrics()
        self.render_anomaly_monitoring()
        self.render_network_analysis()
        self.render_detailed_metrics()

    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("Dashboard Controls")
        
        # Time window selection
        self.selected_window = st.sidebar.slider(
            "Time Window (hours)",
            min_value=1,
            max_value=24,
            value=6
        )
        
        # Feature selection
        self.selected_features = st.sidebar.multiselect(
            "Select Metrics",
            ['bandwidth', 'packets_rate', 'delay', 'jitter', 'loss_rate'],
            default=['bandwidth', 'packets_rate']
        )
        
        # Auto-refresh toggle
        self.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

    def render_key_metrics(self):
        """Display key performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Current network status
        with col1:
            current_error = self.results['reconstruction_error'].iloc[-1]
            status = self.get_health_status(current_error)
            st.metric(
                "Network Health",
                status,
                delta="Normal" if status == "🟢 Healthy" else "Alert"
            )
            
        # Anomaly count
        with col2:
            anomaly_count = sum(self.results['is_anomaly'])
            st.metric(
                "Detected Anomalies",
                anomaly_count,
                delta=f"{anomaly_count} in last 24h"
            )
            
        # Average metrics
        with col3:
            avg_bandwidth = self.network_data['bandwidth'].mean()
            st.metric(
                "Avg Bandwidth",
                f"{avg_bandwidth:.2f} Mbps",
                delta=f"{(self.network_data['bandwidth'].iloc[-1] - avg_bandwidth):.2f}"
            )
            
        # Packet loss
        with col4:
            avg_loss = self.network_data['loss_rate'].mean()
            st.metric(
                "Packet Loss",
                f"{avg_loss:.2%}",
                delta=f"{(self.network_data['loss_rate'].iloc[-1] - avg_loss):.2%}"
            )

    def render_anomaly_monitoring(self):
        """Display anomaly detection visualization"""
        st.subheader("Anomaly Detection Monitor")
        
        # Create error timeline
        fig = px.line(
            self.results,
            y='reconstruction_error',
            title='Reconstruction Error Timeline'
        )
        
        # Add threshold lines
        fig.add_hline(
            y=self.alert_thresholds['critical'],
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold"
        )
        
        fig.add_hline(
            y=self.alert_thresholds['warning'],
            line_dash="dash",
            line_color="yellow",
            annotation_text="Warning Threshold"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_network_analysis(self):
        """Display network analysis metrics"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traffic Distribution")
            # Traffic heatmap
            fig = px.density_heatmap(
                self.network_data,
                x='src_node',
                y='dst_node',
                title='Network Traffic Heatmap'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Feature Correlations")
            # Correlation matrix
            corr = self.network_data[self.selected_features].corr()
            fig = px.imshow(
                corr,
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_detailed_metrics(self):
        """Display detailed network metrics"""
        st.subheader("Detailed Network Metrics")
        
        # Create tabs for different metric groups
        tab1, tab2, tab3 = st.tabs([
            "Traffic Metrics",
            "Performance Metrics",
            "Anomaly Details"
        ])
        
        with tab1:
            self.plot_traffic_metrics()
            
        with tab2:
            self.plot_performance_metrics()
            
        with tab3:
            self.show_anomaly_details()

    def get_health_status(self, error):
        """Determine network health status"""
        if error > self.alert_thresholds['critical']:
            return "🔴 Critical"
        elif error > self.alert_thresholds['warning']:
            return "🟡 Warning"
        return "🟢 Healthy"

    def plot_traffic_metrics(self):
        """Plot traffic-related metrics"""
        fig = px.line(
            self.network_data,
            y=['bandwidth', 'packets_rate', 'throughput'],
            title='Network Traffic Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_performance_metrics(self):
        """Plot performance-related metrics"""
        fig = px.line(
            self.network_data,
            y=['delay', 'jitter', 'loss_rate'],
            title='Network Performance Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_anomaly_details(self):
        """Display detailed anomaly information"""
        anomalies = self.results[self.results['is_anomaly'] == True].tail(10)
        if not anomalies.empty:
            st.dataframe(anomalies)
        else:
            st.info("No anomalies detected in the selected time window")

def main():
    dashboard = NetworkMonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()