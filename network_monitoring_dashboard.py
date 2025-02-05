import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set the page configuration as the very first Streamlit command
st.set_page_config(
    page_title="Network Slice Monitoring",
    page_icon="🌐",
    layout="wide"
)

@st.cache_data
def load_csv_data(file_path):
    """Optimize the CSV loading by reading data in chunks and limiting memory usage"""
    try:
        # Use chunks to load large data files more efficiently
        chunk_size = 50000  # Adjust this as needed
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

class NetworkMonitoringDashboard:
    def __init__(self):
        # Load data and initialize
        self.load_data()
        self.setup_alerts()

    def load_data(self):
        """Load all necessary data files using caching"""
        try:
            # Load main datasets via caching
            self.network_data = load_csv_data('Dataset/network_data.csv')
            self.results = load_csv_data('results/anomaly_detection_results.csv')
            self.analysis = load_csv_data('results/anomaly_analysis_report.csv')
            
            # Process timestamps once the data is loaded
            self.network_data['timestamp'] = pd.to_datetime('now') - pd.to_timedelta(
                self.network_data.index * 5, unit='minutes')
        except Exception as e:
            st.error(f"Error loading data: {e}")

    def setup_alerts(self):
        """Initialize alert thresholds for triggering color codes"""
        self.alert_thresholds = {
            'critical': 20.0,  # Red alert threshold
            'warning': 10.0,   # Yellow alert threshold
            'normal': 5.0      # Green status threshold
        }

    def run_dashboard(self):
        """Main dashboard layout and functionality"""
        # Dashboard header
        st.title("🌐 Network Slice Monitoring Dashboard")
        
        # Render sidebar controls
        self.render_sidebar()
        
        # Render main dashboard sections
        self.render_key_metrics()
        self.render_anomaly_monitoring()
        self.render_network_analysis()
        self.render_detailed_metrics()

    def render_sidebar(self):
        """Render sidebar controls for filtering and configuration"""
        st.sidebar.header("Dashboard Controls")
        
        # Time window selection slider
        self.selected_window = st.sidebar.slider(
            "Time Window (hours)",
            min_value=1,
            max_value=24,
            value=6
        )
        
        # Multi-select for choosing key metrics to display in correlations
        self.selected_features = st.sidebar.multiselect(
            "Select Metrics",
            ['bandwidth', 'packets_rate', 'delay', 'jitter', 'loss_rate'],
            default=['bandwidth', 'packets_rate']
        )
        
        # Toggle for enabling auto-refresh
        self.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

    def render_key_metrics(self):
        """Display key performance metrics on the dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Current network status based on the latest reconstruction error
        with col1:
            current_error = self.results['reconstruction_error'].iloc[-1]
            status = self.get_health_status(current_error)
            st.metric(
                "Network Health",
                status,
                delta="Normal" if status == "🟢 Healthy" else "Alert"
            )
            
        # Number of anomalies detected
        with col2:
            anomaly_count = sum(self.results['is_anomaly'])
            st.metric(
                "Detected Anomalies",
                anomaly_count,
                delta=f"{anomaly_count} in last 24h"
            )
            
        # Average network bandwidth
        with col3:
            avg_bandwidth = self.network_data['bandwidth'].mean()
            st.metric(
                "Avg Bandwidth",
                f"{avg_bandwidth:.2f} Mbps",
                delta=f"{(self.network_data['bandwidth'].iloc[-1] - avg_bandwidth):.2f}"
            )
            
        # Average network packet loss
        with col4:
            avg_loss = self.network_data['loss_rate'].mean()
            st.metric(
                "Packet Loss",
                f"{avg_loss:.2%}",
                delta=f"{(self.network_data['loss_rate'].iloc[-1] - avg_loss):.2%}"
            )

    def render_anomaly_monitoring(self):
        """Visualize anomaly detection via reconstruction error timeline"""
        st.subheader("Anomaly Detection Monitor")
        
        # Sample the data to prevent large datasets from rendering all points
        sampled_results = self.results.tail(500)  # You can adjust the number of rows here
        
        # Plot the reconstruction error over time
        fig = px.line(
            sampled_results,
            y='reconstruction_error',
            title='Reconstruction Error Timeline'
        )
        
        # Draw a horizontal red line for the critical threshold
        fig.add_hline(
            y=self.alert_thresholds['critical'],
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold"
        )
        
        # Draw a horizontal yellow line for the warning threshold
        fig.add_hline(
            y=self.alert_thresholds['warning'],
            line_dash="dash",
            line_color="yellow",
            annotation_text="Warning Threshold"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_network_analysis(self):
        """Visualize network analysis metrics including traffic distribution and correlations"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traffic Distribution")
            # Generate a heatmap to display network traffic flows between nodes
            fig = px.density_heatmap(
                self.network_data,
                x='src_node',
                y='dst_node',
                title='Network Traffic Heatmap'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Feature Correlations")
            # Compute correlation matrix for selected features
            corr = self.network_data[self.selected_features].corr()
            fig = px.imshow(
                corr,
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_detailed_metrics(self):
        """Display detailed network metrics using tabs"""
        st.subheader("Detailed Network Metrics")
        
        # Create tabs for traffic, performance, and anomaly details
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
        """Determine network health status based on the current error metric"""
        if error > self.alert_thresholds['critical']:
            return "🔴 Critical"
        elif error > self.alert_thresholds['warning']:
            return "🟡 Warning"
        return "🟢 Healthy"

    def plot_traffic_metrics(self):
        """Plot traffic-related metrics such as bandwidth and packet rates"""
        fig = px.line(
            self.network_data.tail(500),
            y=['bandwidth', 'packets_rate', 'throughput'],
            title='Network Traffic Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_performance_metrics(self):
        """Plot performance-related metrics such as delay, jitter, and loss rate"""
        fig = px.line(
            self.network_data.tail(500),
            y=['delay', 'jitter', 'loss_rate'],
            title='Network Performance Metrics'
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_anomaly_details(self):
        """Display a table containing details about the latest anomalies"""
        anomalies = self.results[self.results['is_anomaly'] == True].tail(10)
        if not anomalies.empty:
            st.dataframe(anomalies)
        else:
            st.info("No anomalies detected in the selected time window")

def main():
    # Create an instance of the dashboard and run it
    dashboard = NetworkMonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
