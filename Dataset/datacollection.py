from datanetAPI import DatanetAPI
import numpy as np
import pandas as pd
from tqdm import tqdm

class NetworkDataCollector:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.collected_data = []
        self.previous_values = {}

    def collect_data(self):
        print(f"Loading data from: {self.dataset_path}")
        tool = DatanetAPI(self.dataset_path, shuffle=False)
        
        try:
            for sample_id, s in tqdm(enumerate(iter(tool)), desc="Processing samples", unit="sample"):
                T = s.get_traffic_matrix()
                D = s.get_performance_matrix()
                rows, cols = T.shape

                # Step 1: Precompute valid (src, dst) pairs with traffic
                valid_pairs = [
                    (src, dst) 
                    for src in range(rows) 
                    for dst in range(cols) 
                    if src != dst 
                    and isinstance(T[src, dst], dict) 
                    and 'AggInfo' in T[src, dst]
                    and (T[src, dst]['AggInfo'].get('AvgBw', 0) > 0 
                         or T[src, dst]['AggInfo'].get('PktsGen', 0) > 0)
                ]

                # Step 2: Batch-process all valid pairs using list comprehensions
                data_batch = []
                for src, dst in valid_pairs:
                    try:
                        traffic_info = T[src, dst]['AggInfo']
                        perf_info = D[src, dst]['AggInfo'] if isinstance(D[src, dst], dict) else {}
                        
                        # Extract metrics in bulk
                        current_bw = float(traffic_info.get('AvgBw', 0))
                        packets_gen = float(traffic_info.get('PktsGen', 0))
                        packets_drop = float(perf_info.get('PktsDrop', 0))
                        
                        # Calculate derived metrics
                        flow_key = f"{src}-{dst}"
                        prev_bw = self.previous_values.get(f"{flow_key}_bw", current_bw)
                        bw_change = (current_bw - prev_bw) / prev_bw if prev_bw > 0 else 0
                        self.previous_values[f"{flow_key}_bw"] = current_bw
                        
                        loss_rate = packets_drop / packets_gen if packets_gen > 0 else 0
                        throughput = current_bw / (1 + packets_drop) if packets_drop != -1 else current_bw

                        # Append data point
                        data_batch.append({
                            'sample_id': sample_id,
                            'time_window': sample_id,
                            'bandwidth': current_bw,
                            'packets_rate': packets_gen,
                            'delay': float(perf_info.get('AvgDelay', 0)),
                            'jitter': float(perf_info.get('Jitter', 0)),
                            'loss_rate': loss_rate,
                            'bandwidth_change': bw_change,
                            'throughput': throughput,
                            'src_node': src,
                            'dst_node': dst
                        })
                    except Exception as e:
                        print(f"Error processing flow {src}->{dst}: {str(e)}")
                        continue

                # Step 3: Extend collected_data in bulk
                self.collected_data.extend(data_batch)

                if sample_id >= 7899:
                    break

        except Exception as e:
            print(f"Error during data collection: {str(e)}")
            import traceback
            traceback.print_exc()

        return pd.DataFrame(self.collected_data)

def main():
    collector = NetworkDataCollector("C:/Projects/Network/Simulations")
    df = collector.collect_data()
    
    if df.empty:
        print("No data collected!")
        return
    
    print("\nDataset Info:")
    print(df.info())
    df.to_csv("Dataset/network_data.csv", index=False)
    print(f"\nSaved {len(df)} records")

if __name__ == "__main__":
    main()