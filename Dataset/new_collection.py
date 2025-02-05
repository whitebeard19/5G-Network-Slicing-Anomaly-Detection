from datanetAPI import DatanetAPI
import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
import os

# Attempt to import tqdm for progress bar; fallback if not available.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class NetworkDataCollector:
    def __init__(self, dataset_path: str, parallel: bool = True, max_workers: int = None, sample_limit: int = 1000, verbose: bool = False):
        """
        :param dataset_path: Path to the dataset.
        :param parallel: If True, process flows in parallel.
        :param max_workers: Number of threads to use for parallel processing.
        :param sample_limit: Limit on number of samples to process.
        :param verbose: If True, print detailed log messages.
        """
        self.dataset_path = dataset_path
        self.collected_data = []
        self.previous_values = {}  # Store previous values for rate calculations
        self.parallel = parallel
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() or 4
        self.sample_limit = sample_limit
        self.verbose = verbose
        self.lock = threading.Lock()

    def _process_flow(self, sample_id: int, src: int, dst: int, T, D):
        """
        Processes an individual flow between src and dst nodes.
        Returns a dictionary with the computed features, or None if the flow is skipped.
        """
        try:
            cellT = T[src, dst]
            if not (isinstance(cellT, dict) and 'AggInfo' in cellT):
                return None

            traffic_info = cellT['AggInfo']
            if traffic_info.get('AvgBw', 0) <= 0 and traffic_info.get('PktsGen', 0) <= 0:
                return None

            cellD = D[src, dst] if (isinstance(D[src, dst], dict) and 'AggInfo' in D[src, dst]) else {}
            perf_info = cellD

            flow_key = f"{src}-{dst}"
            current_bw = float(traffic_info.get('AvgBw', 0))
            packets_gen = float(traffic_info.get('PktsGen', 0))
            packets_drop = float(perf_info.get('PktsDrop', 0))
            
            with self.lock:
                prev_bw = self.previous_values.get(f"{flow_key}_bw", current_bw)
                bw_change = (current_bw - prev_bw) / prev_bw if prev_bw > 0 else 0
                self.previous_values[f"{flow_key}_bw"] = current_bw

            data_point = {
                'sample_id': sample_id,
                'time_window': sample_id,  # For temporal analysis
                'bandwidth': current_bw,
                'packets_rate': packets_gen,
                'delay': float(perf_info.get('AvgDelay', 0)),
                'jitter': float(perf_info.get('Jitter', 0)),
                'loss_rate': packets_drop / packets_gen if packets_gen > 0 else 0,
                'bandwidth_change': bw_change,
                'throughput': current_bw / (1 + packets_drop),
                'src_node': src,
                'dst_node': dst
            }
            return data_point
        except Exception as e:
            if self.verbose:
                print(f"Error processing flow {src}->{dst} in sample {sample_id}: {str(e)}")
            return None

    def collect_data(self) -> pd.DataFrame:
        if self.verbose:
            print(f"Loading data from: {self.dataset_path}")
        tool = DatanetAPI(self.dataset_path, shuffle=False)
        
        # Wrap the tool iterator with tqdm for a progress bar.
        for sample_id, s in enumerate(tqdm(tool, total=self.sample_limit, desc="Processing samples")):
            # Retrieve matrices (port stats not used in current processing)
            T = s.get_traffic_matrix()      
            D = s.get_performance_matrix()   

            rows, cols = T.shape
            # Build list of source-destination pairs (skip diagonal)
            flow_indices = [(src, dst) for src in range(rows) for dst in range(cols) if src != dst]

            if self.parallel:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Process flows in parallel.
                    results = executor.map(lambda idx: self._process_flow(sample_id, idx[0], idx[1], T, D), flow_indices)
                for res in results:
                    if res is not None:
                        self.collected_data.append(res)
            else:
                for src, dst in flow_indices:
                    dp = self._process_flow(sample_id, src, dst, T, D)
                    if dp is not None:
                        self.collected_data.append(dp)

            if self.verbose and sample_id % 10 == 0:
                print(f"Collected records so far: {len(self.collected_data)}")

            if sample_id >= self.sample_limit - 1:
                break

        if not self.collected_data:
            print("Warning: No data was collected!")
            return pd.DataFrame()

        return pd.DataFrame(self.collected_data)

def main():
    # Adjust parameters as needed.
    collector = NetworkDataCollector(
        "C:/Projects/Network/Simulations",
        parallel=True,       # Enable parallel flow processing.
        verbose=False,       # Set to True for more logging.
        sample_limit=1000    # Adjust sample limit as needed (can be increased, e.g., up to 7899).
    )
    
    df = collector.collect_data()
    
    if df.empty:
        print("No data was collected. Please check the dataset path and data structure.")
        return
    
    # Display dataset information.
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few records:")
    print(df.head())
    print("\nValue ranges:")
    print(df.describe())
    
    # Save the collected data to CSV.
    output_path = "Dataset/network_data.csv"
    df.to_csv(output_path, index=False)
    
    print("\nData Collection Summary:")
    print(f"Total samples: {df['sample_id'].nunique()}")
    print(f"Total records: {len(df)}")
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
