import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NetworkMonitoringDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Slice Monitoring Dashboard")
        self.root.geometry("1200x800")
        
        # Load data (and process timestamps)
        self.load_data()
        
        # Set alert thresholds
        self.alert_thresholds = {
            'critical': 20.0,  # Red alert
            'warning': 10.0,   # Yellow alert
            'normal': 5.0      # Green status
        }
        
        # Build the UI
        self.setup_ui()

    def load_data(self):
        try:
            self.network_data = pd.read_csv('Dataset/network_data.csv')
            self.results = pd.read_csv('results/anomaly_detection_results.csv')
            self.analysis = pd.read_csv('results/anomaly_analysis_report.csv')
            # Process timestamps (simulate as in Streamlit version)
            self.network_data['timestamp'] = pd.to_datetime('now') - pd.to_timedelta(
                self.network_data.index * 5, unit='minutes')
        except Exception as e:
            print("Error loading data:", e)

    def setup_ui(self):
        # Create a sidebar frame for controls
        self.sidebar = tk.Frame(self.root, width=250, bg='#f0f0f0')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create the main content frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = tk.Label(self.main_frame, text="Network Slice Monitoring Dashboard", font=("Arial", 20))
        title_label.pack(pady=10)
        
        # Setup sidebar controls
        self.setup_sidebar_controls()
        
        # Key Metrics Section
        self.setup_key_metrics_section()
        
        # Anomaly Monitor Section
        self.setup_anomaly_monitor_section()
        
        # Network Analysis Section
        self.setup_network_analysis_section()
        
        # Detailed Metrics (using Notebook tabs)
        self.setup_detailed_metrics_section()
        
    def setup_sidebar_controls(self):
        lbl = tk.Label(self.sidebar, text="Dashboard Controls", bg='#f0f0f0', font=("Arial", 14))
        lbl.pack(pady=10)
        
        # Time Window slider
        time_lbl = tk.Label(self.sidebar, text="Time Window (hours)", bg='#f0f0f0')
        time_lbl.pack(pady=5)
        self.time_window_var = tk.IntVar(value=6)
        self.time_slider = tk.Scale(self.sidebar, from_=1, to=24, orient=tk.HORIZONTAL, variable=self.time_window_var)
        self.time_slider.pack(pady=5)
        
        # Multi-select for metrics (using a Listbox in MULTIPLE mode)
        metrics_lbl = tk.Label(self.sidebar, text="Select Metrics", bg='#f0f0f0')
        metrics_lbl.pack(pady=5)
        self.metrics_options = ['bandwidth', 'packets_rate', 'delay', 'jitter', 'loss_rate']
        self.metrics_listbox = tk.Listbox(self.sidebar, selectmode=tk.MULTIPLE, height=5)
        for metric in self.metrics_options:
            self.metrics_listbox.insert(tk.END, metric)
        # Default selections: bandwidth, packets_rate
        self.metrics_listbox.selection_set(0, 1)
        self.metrics_listbox.pack(pady=5)
        
        # Auto Refresh checkbox
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_cb = tk.Checkbutton(self.sidebar, text="Auto Refresh", bg='#f0f0f0', variable=self.auto_refresh_var)
        auto_refresh_cb.pack(pady=5)
        
        # Refresh button to update the dashboard manually
        refresh_btn = tk.Button(self.sidebar, text="Refresh Dashboard", command=self.refresh_dashboard)
        refresh_btn.pack(pady=10)

    def setup_key_metrics_section(self):
        # Frame to display key performance metrics
        self.key_metrics_frame = tk.Frame(self.main_frame)
        self.key_metrics_frame.pack(fill=tk.X, pady=10)
        
        # Four metric display labels arranged in a grid
        self.network_health_label = tk.Label(self.key_metrics_frame, text="Network Health: ", font=("Arial", 12))
        self.network_health_label.grid(row=0, column=0, padx=10)
        
        self.anomaly_count_label = tk.Label(self.key_metrics_frame, text="Detected Anomalies: ", font=("Arial", 12))
        self.anomaly_count_label.grid(row=0, column=1, padx=10)
        
        self.avg_bandwidth_label = tk.Label(self.key_metrics_frame, text="Avg Bandwidth: ", font=("Arial", 12))
        self.avg_bandwidth_label.grid(row=0, column=2, padx=10)
        
        self.packet_loss_label = tk.Label(self.key_metrics_frame, text="Packet Loss: ", font=("Arial", 12))
        self.packet_loss_label.grid(row=0, column=3, padx=10)
        
        self.update_key_metrics()
        
    def update_key_metrics(self):
        try:
            current_error = self.results['reconstruction_error'].iloc[-1]
            status = self.get_health_status(current_error)
            self.network_health_label.config(text=f"Network Health: {status}")
            
            anomaly_count = self.results['is_anomaly'].sum() if 'is_anomaly' in self.results.columns else 0
            self.anomaly_count_label.config(text=f"Detected Anomalies: {anomaly_count}")
            
            avg_bandwidth = self.network_data['bandwidth'].mean()
            last_bw = self.network_data['bandwidth'].iloc[-1]
            self.avg_bandwidth_label.config(text=f"Avg Bandwidth: {avg_bandwidth:.2f} Mbps (Δ {last_bw - avg_bandwidth:.2f})")
            
            avg_loss = self.network_data['loss_rate'].mean()
            last_loss = self.network_data['loss_rate'].iloc[-1]
            self.packet_loss_label.config(text=f"Packet Loss: {avg_loss:.2%} (Δ {last_loss - avg_loss:.2%})")
        except Exception as e:
            print("Error updating key metrics:", e)
     
    def setup_anomaly_monitor_section(self):
        # Section for Anomaly Detection Monitor
        self.anomaly_frame = tk.Frame(self.main_frame)
        self.anomaly_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        lbl = tk.Label(self.anomaly_frame, text="Anomaly Detection Monitor", font=("Arial", 14))
        lbl.pack()
        
        # Create Matplotlib figure for reconstruction error timeline
        self.fig_anomaly, self.ax_anomaly = plt.subplots(figsize=(6, 4))
        self.plot_anomaly_monitor()
        
        self.canvas_anomaly = FigureCanvasTkAgg(self.fig_anomaly, master=self.anomaly_frame)
        self.canvas_anomaly.draw()
        self.canvas_anomaly.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def plot_anomaly_monitor(self):
        try:
            self.ax_anomaly.clear()
            # Plot reconstruction error versus sample index
            self.ax_anomaly.plot(self.results['reconstruction_error'], label='Reconstruction Error')
            self.ax_anomaly.axhline(y=self.alert_thresholds['critical'], color='red', linestyle='--', label='Critical Threshold')
            self.ax_anomaly.axhline(y=self.alert_thresholds['warning'], color='yellow', linestyle='--', label='Warning Threshold')
            self.ax_anomaly.set_title("Reconstruction Error Timeline")
            self.ax_anomaly.set_xlabel("Sample Index")
            self.ax_anomaly.set_ylabel("Reconstruction Error")
            self.ax_anomaly.legend()
            self.canvas_anomaly.draw_idle()
        except Exception as e:
            print("Error plotting anomaly monitor:", e)
            
    def setup_network_analysis_section(self):
        # Section for Network Analysis
        self.network_analysis_frame = tk.Frame(self.main_frame)
        self.network_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Two side-by-side frames: left for Traffic Distribution, right for Feature Correlations
        left_frame = tk.Frame(self.network_analysis_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right_frame = tk.Frame(self.network_analysis_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        left_lbl = tk.Label(left_frame, text="Traffic Distribution", font=("Arial", 14))
        left_lbl.pack()
        self.fig_traffic, self.ax_traffic = plt.subplots(figsize=(4, 3))
        self.plot_traffic_distribution()
        self.canvas_traffic = FigureCanvasTkAgg(self.fig_traffic, master=left_frame)
        self.canvas_traffic.draw()
        self.canvas_traffic.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        right_lbl = tk.Label(right_frame, text="Feature Correlation Matrix", font=("Arial", 14))
        right_lbl.pack()
        self.fig_corr, self.ax_corr = plt.subplots(figsize=(4, 3))
        self.plot_feature_correlations()
        self.canvas_corr = FigureCanvasTkAgg(self.fig_corr, master=right_frame)
        self.canvas_corr.draw()
        self.canvas_corr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def plot_traffic_distribution(self):
        try:
            self.ax_traffic.clear()
            # Scatter plot using src_node (x) and dst_node (y)
            self.ax_traffic.scatter(self.network_data['src_node'], self.network_data['dst_node'], alpha=0.5)
            self.ax_traffic.set_title("Traffic Distribution")
            self.ax_traffic.set_xlabel("src_node")
            self.ax_traffic.set_ylabel("dst_node")
            self.canvas_traffic.draw_idle()
        except Exception as e:
            print("Error plotting traffic distribution:", e)
    
    def plot_feature_correlations(self):
        try:
            self.ax_corr.clear()
            # Get the selected features from the listbox; default to bandwidth and packets_rate if none selected
            selected_indices = self.metrics_listbox.curselection()
            if not selected_indices:
                selected_features = ['bandwidth', 'packets_rate']
            else:
                selected_features = [self.metrics_listbox.get(i) for i in selected_indices]
            corr = self.network_data[selected_features].corr()
            cax = self.ax_corr.matshow(corr, cmap='coolwarm')
            self.fig_corr.colorbar(cax, ax=self.ax_corr)
            self.ax_corr.set_xticks(range(len(selected_features)))
            self.ax_corr.set_xticklabels(selected_features, rotation=45)
            self.ax_corr.set_yticks(range(len(selected_features)))
            self.ax_corr.set_yticklabels(selected_features)
            self.ax_corr.set_title("Feature Correlation Matrix", pad=20)
            self.canvas_corr.draw_idle()
        except Exception as e:
            print("Error plotting feature correlations:", e)
            
    def setup_detailed_metrics_section(self):
        # Section for Detailed Metrics using Notebook tabs
        self.detailed_frame = tk.Frame(self.main_frame)
        self.detailed_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        notebook = ttk.Notebook(self.detailed_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Traffic Metrics
        tab1 = tk.Frame(notebook)
        notebook.add(tab1, text="Traffic Metrics")
        self.fig_traffic_metrics, self.ax_traffic_metrics = plt.subplots(figsize=(5, 3))
        self.plot_traffic_metrics()
        canvas_tab1 = FigureCanvasTkAgg(self.fig_traffic_metrics, master=tab1)
        canvas_tab1.draw()
        canvas_tab1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Performance Metrics
        tab2 = tk.Frame(notebook)
        notebook.add(tab2, text="Performance Metrics")
        self.fig_performance_metrics, self.ax_performance_metrics = plt.subplots(figsize=(5, 3))
        self.plot_performance_metrics()
        canvas_tab2 = FigureCanvasTkAgg(self.fig_performance_metrics, master=tab2)
        canvas_tab2.draw()
        canvas_tab2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Anomaly Details (using a Treeview table)
        tab3 = tk.Frame(notebook)
        notebook.add(tab3, text="Anomaly Details")
        self.setup_anomaly_details_table(tab3)
        
    def plot_traffic_metrics(self):
        try:
            self.ax_traffic_metrics.clear()
            self.ax_traffic_metrics.plot(self.network_data['bandwidth'], label='bandwidth')
            self.ax_traffic_metrics.plot(self.network_data['packets_rate'], label='packets_rate')
            self.ax_traffic_metrics.plot(self.network_data['throughput'], label='throughput')
            self.ax_traffic_metrics.set_title("Traffic Metrics")
            self.ax_traffic_metrics.set_xlabel("Sample Index")
            self.ax_traffic_metrics.set_ylabel("Value")
            self.ax_traffic_metrics.legend()
        except Exception as e:
            print("Error plotting traffic metrics:", e)
    
    def plot_performance_metrics(self):
        try:
            self.ax_performance_metrics.clear()
            self.ax_performance_metrics.plot(self.network_data['delay'], label='delay')
            self.ax_performance_metrics.plot(self.network_data['jitter'], label='jitter')
            self.ax_performance_metrics.plot(self.network_data['loss_rate'], label='loss_rate')
            self.ax_performance_metrics.set_title("Performance Metrics")
            self.ax_performance_metrics.set_xlabel("Sample Index")
            self.ax_performance_metrics.set_ylabel("Value")
            self.ax_performance_metrics.legend()
        except Exception as e:
            print("Error plotting performance metrics:", e)
    
    def setup_anomaly_details_table(self, parent):
        try:
            columns = list(self.results.columns)
            tree = ttk.Treeview(parent, columns=columns, show='headings')
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            anomalies = self.results[self.results['is_anomaly'] == True].tail(10)
            for _, row in anomalies.iterrows():
                tree.insert("", tk.END, values=list(row))
            tree.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print("Error setting up anomaly details table:", e)
    
    def get_health_status(self, error):
        if error > self.alert_thresholds['critical']:
            return "🔴 Critical"
        elif error > self.alert_thresholds['warning']:
            return "🟡 Warning"
        return "🟢 Healthy"
    
    def refresh_dashboard(self):
        # Reload data and update all views
        self.load_data()
        self.update_key_metrics()
        self.plot_anomaly_monitor()
        self.plot_traffic_distribution()
        self.plot_feature_correlations()
        self.plot_traffic_metrics()
        self.plot_performance_metrics()
        # For anomaly details, rebuild the detailed metrics section
        self.detailed_frame.destroy()
        self.setup_detailed_metrics_section()

def main():
    root = tk.Tk()
    app = NetworkMonitoringDashboardApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 