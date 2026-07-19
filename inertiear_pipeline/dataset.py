import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from inertiear_pipeline.preprocessing import apply_wiener_filter, segment_coherence
from inertiear_pipeline.features import extract_features

CLASS_MAP = {
    'weather': 0,
    'navigation': 1,
    'reminder': 2,
    'time': 3,
    'sun': 4,
    'stock': 5,
    'air': 6
}

def parse_action(val):
    if not isinstance(val, str):
        return None
    try:
        normalized = val.replace('|', ',')
        parsed = ast.literal_eval(normalized)
        return parsed.get('action')
    except Exception:
        # Fallback keyword match
        for act in CLASS_MAP.keys():
            if act in val.lower():
                return act
        return None

class InertiEARDataset(Dataset):
    def __init__(self, csv_file, data_dir, cache_dir=None, transform=None, use_segmentation=True, preload_ram=False):
        """
        csv_file: Path to the metadata CSV file
        data_dir: Base directory containing data/
        cache_dir: Directory to cache computed spectrograms
        transform: PyTorch transforms
        use_segmentation: Whether to apply coherence-based segmentation
        preload_ram: Whether to pre-load all cached spectrograms in RAM
        """
        self.df = pd.read_csv(csv_file, header=None)
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.use_segmentation = use_segmentation
        self.preload_ram = preload_ram
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Parse actions and keep only valid rows
        self.valid_indices = []
        self.labels = []
        self.uuids = []
        self.paths = []
        
        for idx, row in self.df.iterrows():
            action = parse_action(row[3])
            if action in CLASS_MAP:
                self.valid_indices.append(idx)
                self.labels.append(CLASS_MAP[action])
                self.uuids.append(row[0])
                self.paths.append(row[2])
                
        print(f"Loaded dataset metadata: {len(self.valid_indices)} valid VUI samples.")
        
        # Pre-load in RAM if requested
        self.preloaded_specs = {}
        if self.preload_ram and self.cache_dir:
            print("Pre-loading cached spectrograms into RAM in parallel...")
            from concurrent.futures import ThreadPoolExecutor
            
            def load_one(uuid):
                cache_path = os.path.join(self.cache_dir, f"{uuid}.npy")
                if os.path.exists(cache_path):
                    try:
                        return uuid, np.load(cache_path)
                    except Exception:
                        pass
                return None
                
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(load_one, self.uuids))
                
            for res in results:
                if res is not None:
                    uuid, spec = res
                    self.preloaded_specs[uuid] = spec
            print(f"Successfully pre-loaded {len(self.preloaded_specs)} spectrograms into RAM.")

    def __len__(self):
        return len(self.valid_indices)

    def _get_max_energy_axis(self, data):
        # data: shape (3, N)
        variances = np.var(data, axis=1)
        return np.argmax(variances)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        label = self.labels[idx]
        wav_path = self.paths[idx]
        
        # Check preloaded in RAM first
        if uuid in self.preloaded_specs:
            spec = self.preloaded_specs[uuid]
            spec_tensor = torch.from_numpy(spec).unsqueeze(0)
            if self.transform:
                spec_tensor = self.transform(spec_tensor)
            return spec_tensor, label
            
        # Check cache on disk
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{uuid}.npy")
            if os.path.exists(cache_path):
                try:
                    spec = np.load(cache_path)
                    spec_tensor = torch.from_numpy(spec).unsqueeze(0) # add channel dim -> (1, 244, 244)
                    if self.transform:
                        spec_tensor = self.transform(spec_tensor)
                    return spec_tensor, label
                except Exception as e:
                    # If loading fails, recompute
                    pass

        # Locate folders and file names
        folder_path = os.path.dirname(wav_path)
        # Note: folder_path might be relative to the workspace, adjust path
        actual_folder_path = os.path.join(self.data_dir, os.path.relpath(folder_path, "./"))
        basename = os.path.basename(folder_path)
        
        accnpy_path = os.path.join(actual_folder_path, f"{basename}.accnpy")
        gyronpy_path = os.path.join(actual_folder_path, f"{basename}.gyronpy")
        
        # Fallback if file doesn't exist
        if not os.path.exists(accnpy_path) or not os.path.exists(gyronpy_path):
            # Return empty / zeros as fallback
            spec = np.zeros((244, 244), dtype=np.float32)
        else:
            try:
                acc_data = np.load(accnpy_path) # shape (4, N)
                gyro_data = np.load(gyronpy_path) # shape (4, M)
                
                # Apply Wiener filter to all axis readings (rows 1, 2, 3)
                for i in range(1, 4):
                    acc_data[i, :] = apply_wiener_filter(acc_data[i, :])
                    gyro_data[i, :] = apply_wiener_filter(gyro_data[i, :])
                    
                if self.use_segmentation:
                    # Select max energy axis for coherence segmentation
                    acc_max_idx = self._get_max_energy_axis(acc_data[1:, :]) + 1
                    gyro_max_idx = self._get_max_energy_axis(gyro_data[1:, :]) + 1
                    
                    t_grid, envelope, threshold, segments = segment_coherence(
                        acc_data[0, :], acc_data[acc_max_idx, :],
                        gyro_data[0, :], gyro_data[gyro_max_idx, :]
                    )
                    
                    if len(segments) > 0:
                        # Find the longest segment
                        longest_seg = max(segments, key=lambda x: x[1] - x[0])
                        t_start, t_end = longest_seg[0], longest_seg[1]
                        
                        # Slice signals
                        acc_mask = (acc_data[0] >= t_start) & (acc_data[0] <= t_end)
                        gyro_mask = (gyro_data[0] >= t_start) & (gyro_data[0] <= t_end)
                        
                        # Only slice if we have enough samples
                        if np.sum(acc_mask) > 10 and np.sum(gyro_mask) > 10:
                            acc_data = acc_data[:, acc_mask]
                            gyro_data = gyro_data[:, gyro_mask]
                
                # Extract features (244x244 spectrogram)
                spec = extract_features(acc_data, gyro_data)
                
            except Exception as e:
                # If error happens during processing, use zero fallback
                spec = np.zeros((244, 244), dtype=np.float32)
                
        # Save to cache
        if self.cache_dir:
            try:
                np.save(os.path.join(self.cache_dir, f"{uuid}.npy"), spec)
            except Exception:
                pass
                
        spec_tensor = torch.from_numpy(spec).unsqueeze(0) # (1, 244, 244)
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
            
        return spec_tensor, label
