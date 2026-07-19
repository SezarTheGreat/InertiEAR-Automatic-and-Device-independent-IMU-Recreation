import numpy as np
import scipy.signal as signal
from scipy.ndimage import zoom
from inertiear_pipeline.preprocessing import apply_filter

def reduce_dimensions(sensor_data):
    """
    Applies dimension reduction to 3-axis sensor data.
    sensor_data: numpy array of shape (3, N)
    Returns: 1D reduced signal of shape (N,)
    Formula: S_dagger(t) = sign(s_max(t)) * ||S||(t)
    """
    # 1. Compute L2 norm at each time step
    norm = np.linalg.norm(sensor_data, axis=0) # shape (N,)
    
    # 2. Find the axis with the maximum energy (variance)
    variances = np.var(sensor_data, axis=1)
    max_axis_idx = np.argmax(variances)
    s_max = sensor_data[max_axis_idx, :] # shape (N,)
    
    # 3. Apply formula
    s_dagger = np.sign(s_max) * norm
    return s_dagger

def normalize_signal(sig):
    """
    Min-max normalizes a signal to [0, 1].
    """
    min_val = np.min(sig)
    max_val = np.max(sig)
    diff = max_val - min_val
    if diff == 0:
        return np.zeros_like(sig)
    return (sig - min_val) / diff

def chronological_concatenate(t_acc, acc_sig, t_gyro, gyro_sig):
    """
    Interleaves/concatenates accelerometer and gyroscope signals chronologically based on timestamps.
    t_acc, t_gyro: arrays of timestamps (in ms)
    acc_sig, gyro_sig: 1D normalized signals
    """
    # Combine timestamps and signal values
    combined = []
    for t, val in zip(t_acc, acc_sig):
        combined.append((t, val))
    for t, val in zip(t_gyro, gyro_sig):
        combined.append((t, val))
        
    # Sort by timestamp
    combined.sort(key=lambda x: x[0])
    
    t_comb = np.array([x[0] for x in combined])
    sig_comb = np.array([x[1] for x in combined])
    return t_comb, sig_comb

def suppress_high_frequency(t_comb, sig_comb, target_fs=390.0):
    """
    Applies HPF (cutoff 80 Hz) and random downsampling to target_fs (390 Hz) to introduce jitter.
    """
    # 1. Estimate sampling rate of the combined signal
    mean_diff = np.mean(np.diff(t_comb)) # in ms
    if mean_diff == 0:
        mean_diff = 1.0 # fallback
    fs_est = 1000.0 / mean_diff
    
    # 2. High-pass filter with 80 Hz cutoff
    # Ensure cutoff is less than Nyquist frequency
    nyquist = 0.5 * fs_est
    cutoff = min(80.0, nyquist - 1.0)
    
    if cutoff > 0.1:
        sig_hpf = apply_filter(sig_comb, cutoff=cutoff, fs=fs_est, btype='high')
    else:
        sig_hpf = sig_comb
        
    # 3. Random downsampling to 390 Hz
    duration_sec = (t_comb[-1] - t_comb[0]) / 1000.0
    target_len = int(duration_sec * target_fs)
    
    if target_len > 0 and target_len < len(sig_hpf):
        # Randomly choose target_len indices, sort them to preserve chronology
        indices = np.sort(np.random.choice(len(sig_hpf), size=target_len, replace=False))
        sig_down = sig_hpf[indices]
        t_down = t_comb[indices]
    else:
        sig_down = sig_hpf
        t_down = t_comb
        
    return t_down, sig_down

def generate_spectrogram(sig, n_fft=256, hop_length=4):
    """
    Computes the spectrogram of a signal and resizes it to 244 x 244.
    Returns: 2D numpy array of shape (244, 244) normalized to [0, 1].
    """
    # 1. Compute Short-Time Fourier Transform (STFT)
    # Using Hanning window
    win = signal.windows.hann(n_fft)
    f, t_spec, Zxx = signal.stft(sig, fs=390.0, window=win, nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Amplitude spectrogram
    spec = np.abs(Zxx)
    
    # Log scale spectrogram
    spec_log = np.log(spec + 1e-10)
    
    # 2. Resize to 244 x 244 using zoom
    h, w = spec_log.shape
    if h == 0 or w == 0:
        return np.zeros((244, 244), dtype=np.float32)
        
    zoom_h = 244.0 / h
    zoom_w = 244.0 / w
    
    spec_resized = zoom(spec_log, (zoom_h, zoom_w), order=1) # Bilinear interpolation
    
    # Ensure exact 244x244 shape in case of rounding errors
    if spec_resized.shape != (244, 244):
        # Pad or crop
        padded = np.zeros((244, 244), dtype=np.float32)
        h_lim = min(244, spec_resized.shape[0])
        w_lim = min(244, spec_resized.shape[1])
        padded[:h_lim, :w_lim] = spec_resized[:h_lim, :w_lim]
        spec_resized = padded
        
    # Min-max normalize spectrogram to [0, 1]
    spec_min = np.min(spec_resized)
    spec_max = np.max(spec_resized)
    diff = spec_max - spec_min
    if diff > 0:
        spec_resized = (spec_resized - spec_min) / diff
    else:
        spec_resized = np.zeros_like(spec_resized)
        
    return spec_resized.astype(np.float32)

def extract_features(acc_data, gyro_data):
    """
    Full pipeline to extract a 244x244 spectrogram from raw accelerometer and gyroscope data.
    acc_data: shape (4, N) where row 0 is time (ms) and rows 1-3 are x, y, z
    gyro_data: shape (4, M) where row 0 is time (ms) and rows 1-3 are x, y, z
    """
    # 1. Dimension reduction
    a_dagger = reduce_dimensions(acc_data[1:, :])
    g_dagger = reduce_dimensions(gyro_data[1:, :])
    
    # 2. Normalization
    a_norm = normalize_signal(a_dagger)
    g_norm = normalize_signal(g_dagger)
    
    # 3. Chronological concatenation
    t_comb, sig_comb = chronological_concatenate(acc_data[0, :], a_norm, gyro_data[0, :], g_norm)
    
    # 4. High-frequency suppression and random downsampling
    t_down, sig_down = suppress_high_frequency(t_comb, sig_comb)
    
    # 5. Spectrogram generation
    spectrogram = generate_spectrogram(sig_down)
    return spectrogram
