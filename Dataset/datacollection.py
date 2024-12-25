from datanetAPI import DatanetAPI
import numpy as np
import pandas as pd

class NetworkDataCollector:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.collected_data = []
        self.previous_values = {}  # Store previous values for rate calculations

    def collect_data(self):
        print(f"Loading data from: {self.dataset_path}")
        tool = DatanetAPI(self.dataset_path, shuffle=False)
        
        try:
            for sample_id, s in enumerate(iter(tool)):
                print(f"Processing sample {sample_id}")
                
                # Get necessary matrices
                T = s.get_traffic_matrix()      
                D = s.get_performance_matrix()   
                P = s.get_port_stats()          

                # Get matrix dimensions
                rows, cols = T.shape
                
                # Process each source-destination pair
                for src in range(rows):
                    for dst in range(cols):
                        if src == dst:
                            continue
                            
                        try:
                            # Only process if there's actual traffic
                            if isinstance(T[src,dst], dict) and 'AggInfo' in T[src,dst]:
                                traffic_info = T[src,dst]['AggInfo']
                                
                                if traffic_info.get('AvgBw', 0) > 0 or traffic_info.get('PktsGen', 0) > 0:
                                    # Get Performance Features
                                    perf_info = D[src,dst]['AggInfo'] if isinstance(D[src,dst], dict) and 'AggInfo' in D[src,dst] else {}
                                    
                                    # Calculate derived metrics
                                    flow_key = f"{src}-{dst}"
                                    current_bw = float(traffic_info.get('AvgBw', 0))
                                    packets_gen = float(traffic_info.get('PktsGen', 0))
                                    packets_drop = float(perf_info.get('PktsDrop', 0))
                                    
                                    # Calculate rate changes
                                    prev_bw = self.previous_values.get(f"{flow_key}_bw", current_bw)
                                    bw_change = (current_bw - prev_bw) / prev_bw if prev_bw > 0 else 0
                                    
                                    # Update previous values
                                    self.previous_values[f"{flow_key}_bw"] = current_bw

                                    # Collect features
                                    data_point = {
                                        'sample_id': sample_id,
                                        'time_window': sample_id,  # For temporal analysis
                                        
                                        # Primary Features
                                        'bandwidth': current_bw,
                                        'packets_rate': packets_gen,
                                        'delay': float(perf_info.get('AvgDelay', 0)),
                                        'jitter': float(perf_info.get('Jitter', 0)),
                                        
                                        # Derived Features
                                        'loss_rate': packets_drop / packets_gen if packets_gen > 0 else 0,
                                        'bandwidth_change': bw_change,
                                        'throughput': current_bw / (1 + packets_drop),
                                        
                                        # Flow Information
                                        'src_node': src,
                                        'dst_node': dst
                                    }

                                    self.collected_data.append(data_point)
                            
                        except Exception as e:
                            print(f"Error processing flow {src}->{dst}: {str(e)}")
                            continue

                if sample_id % 10 == 0:
                    print(f"Processed {sample_id} samples, collected {len(self.collected_data)} records")

                # Process more samples for better analysis
                if sample_id >= 200:  # Increased to 50 samples
                    break

        except Exception as e:
            print(f"Error during data collection: {str(e)}")
            import traceback
            traceback.print_exc()

        if not self.collected_data:
            print("Warning: No data was collected!")
            return pd.DataFrame()

        return pd.DataFrame(self.collected_data)

def main():
    # Initialize collector
    collector = NetworkDataCollector("C:/Projects/Network/Simulations")
    
    # Collect data
    df = collector.collect_data()
    
    # Check if data was collected
    if df.empty:
        print("No data was collected. Please check the dataset path and data structure.")
        return
    
    # Print data info before saving
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few records:")
    print(df.head())
    print("\nValue ranges:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv("network_data.csv", index=False)
    
    print("\nData Collection Summary:")
    print(f"Total samples: {df['sample_id'].nunique()}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    main()