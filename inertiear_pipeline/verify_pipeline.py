import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from inertiear_pipeline.preprocessing import apply_wiener_filter, segment_coherence
from inertiear_pipeline.features import extract_features
from inertiear_pipeline.model import InertiEAR_DenseNet, get_piecewise_momentum_optimizer
from inertiear_pipeline.dataset import InertiEARDataset

def test_preprocessing():
    print("--- Testing Preprocessing & Segmentation ---")
    # Generate dummy signal with noise
    t = np.arange(0, 1000, 2.5) # 400 Hz sampling, 1 second duration
    clean_signal = np.sin(2 * np.pi * 50 * t / 1000.0) # 50 Hz tone
    noise = np.random.normal(0, 0.5, len(t))
    noisy_signal = clean_signal + noise
    
    filtered_signal = apply_wiener_filter(noisy_signal)
    
    # Check that Wiener filtering reduces noise (MSE should be lower than raw noisy signal)
    mse_raw = np.mean((noisy_signal - clean_signal) ** 2)
    mse_filt = np.mean((filtered_signal - clean_signal) ** 2)
    print(f"Wiener Filter: Raw Noise MSE = {mse_raw:.4f}, Filtered Noise MSE = {mse_filt:.4f}")
    assert mse_filt < mse_raw, "Wiener filter did not reduce noise MSE!"
    print("[PASS] Wiener Filter test passed!")
    
    # Test Coherence-based segmentation
    acc_axis = np.sin(2 * np.pi * 50 * t / 1000.0)
    gyro_axis = np.sin(2 * np.pi * 50 * t / 1000.0) # fully coherent
    # Add non-coherent noise
    acc_axis += np.random.normal(0, 0.1, len(t))
    gyro_axis += np.random.normal(0, 0.1, len(t))
    
    t_grid, envelope, threshold, segments = segment_coherence(t, acc_axis, t, gyro_axis)
    print(f"Segmentation: found {len(segments)} segments.")
    print("[PASS] Segmentation test passed!")

def test_feature_extraction():
    print("\n--- Testing Feature Extraction ---")
    # 4 channels: time, x, y, z
    t = np.arange(0, 1000, 2.5)
    acc_data = np.zeros((4, len(t)))
    acc_data[0, :] = t
    acc_data[1, :] = np.sin(2 * np.pi * 30 * t / 1000.0)
    acc_data[2, :] = np.cos(2 * np.pi * 30 * t / 1000.0)
    acc_data[3, :] = np.random.normal(0, 0.1, len(t))
    
    gyro_data = np.zeros((4, len(t)))
    gyro_data[0, :] = t
    gyro_data[1, :] = np.cos(2 * np.pi * 30 * t / 1000.0)
    gyro_data[2, :] = np.sin(2 * np.pi * 30 * t / 1000.0)
    gyro_data[3, :] = np.random.normal(0, 0.1, len(t))
    
    spec = extract_features(acc_data, gyro_data)
    print("Spectrogram shape:", spec.shape)
    assert spec.shape == (244, 244), f"Expected shape (244, 244), got {spec.shape}"
    print(f"Spectrogram range: [{spec.min():.4f}, {spec.max():.4f}]")
    assert np.all(spec >= 0) and np.all(spec <= 1), "Spectrogram is not normalized to [0, 1]!"
    print("[PASS] Feature Extraction test passed!")

def test_model():
    print("\n--- Testing Model Architecture ---")
    # Custom light config for testing on CPU
    model = InertiEAR_DenseNet(growth_rate=32, block_config=(2, 2, 2))
    x = torch.randn(4, 1, 244, 244) # Batch size 4, 1 channel, 244x244
    out = model(x)
    print("Model Output Shape:", out.shape)
    assert out.shape == (4, 7), f"Expected output shape (4, 7), got {out.shape}"
    print("[PASS] Model Architecture test passed!")

def test_dataset_dryrun():
    print("\n--- Testing Dataset and Training Dry-run ---")
    csv_file = "StealthyIMU_dataset/metadata/stealthyIMU_dryrun.csv"
    data_dir = "StealthyIMU_dataset"
    
    # Create dataset on dryrun subset
    dataset = InertiEARDataset(csv_file, data_dir, cache_dir="./test_cache", use_segmentation=True)
    assert len(dataset) > 0, "Dataset loaded zero samples!"
    
    # Load first sample
    x, y = dataset[0]
    print("Loaded sample shape:", x.shape, "label:", y)
    assert x.shape == (1, 244, 244), f"Expected shape (1, 244, 244), got {x.shape}"
    
    # Train dry-run with a single batch step
    model = InertiEAR_DenseNet(growth_rate=32, block_config=(2, 2, 2))
    optimizer = get_piecewise_momentum_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    
    # Batch inputs
    batch_x = x.unsqueeze(0) # add batch dim -> (1, 1, 244, 244)
    batch_y = torch.tensor([y], dtype=torch.long)
    
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    print(f"Dry-run forward and backward pass succeeded! Single-step Loss: {loss.item():.4f}")
    print("[PASS] Dataset and Training Dry-run test passed!")

if __name__ == "__main__":
    test_preprocessing()
    test_feature_extraction()
    test_model()
    test_dataset_dryrun()
    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
