import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d

def apply_wiener_filter(sig, noise_power=None):
    """
    Applies a Wiener filter to a 1D signal.
    If noise_power is not provided, uses scipy.signal.wiener.
    """
    if noise_power is None:
        # Standard local-variance based Wiener filter
        return signal.wiener(sig)
    else:
        # Frequency domain Wiener filter using known noise power
        S = np.fft.fft(sig)
        psd = np.abs(S) ** 2
        # Wiener gain: G = psd / (psd + noise_power)
        G = psd / (psd + noise_power + 1e-12)
        filtered_S = S * G
        return np.real(np.fft.ifft(filtered_S))

def design_butter_filter(cutoff, fs, btype='low', order=5):
    """
    Designs a Butterworth filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def apply_filter(sig, cutoff, fs, btype='low', order=5):
    """
    Applies a Butterworth filter using zero-phase filtfilt.
    """
    b, a = design_butter_filter(cutoff, fs, btype=btype, order=order)
    return signal.filtfilt(b, a, sig)

def upsample_signal(t, sig, target_fs=1000.0):
    """
    Upsamples a signal with timestamps t (in ms) to target_fs (default 1000 Hz).
    Returns new timestamps (in ms) and the interpolated signal.
    """
    # Create target time grid in ms
    t_new = np.arange(t[0], t[-1], 1000.0 / target_fs)
    f = interp1d(t, sig, kind='linear', fill_value="extrapolate")
    return t_new, f(t_new)

def otsu_threshold_1d(sig, num_bins=256):
    """
    Computes Otsu's threshold for a 1D signal array.
    """
    min_val, max_val = np.min(sig), np.max(sig)
    if min_val == max_val:
        return min_val
        
    counts, bin_edges = np.histogram(sig, bins=num_bins, range=(min_val, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    total = len(sig)
    sum_total = np.sum(sig)
    
    sum_back = 0.0
    w_back = 0.0
    
    max_variance = 0.0
    threshold = bin_centers[0]
    
    for i in range(num_bins):
        w_back += counts[i]
        if w_back == 0:
            continue
        w_fore = total - w_back
        if w_fore == 0:
            break
            
        sum_back += bin_centers[i] * counts[i]
        mean_back = sum_back / w_back
        mean_fore = (sum_total - sum_back) / w_fore
        
        # Inter-class variance
        var_between = w_back * w_fore * (mean_back - mean_fore) ** 2
        
        if var_between > max_variance:
            max_variance = var_between
            threshold = bin_centers[i]
            
    return threshold

def segment_coherence(t_acc, acc_axis, t_gyro, gyro_axis, fs=1000.0):
    """
    Leverages coherence between accelerometer and gyroscope responses to segment speech.
    """
    # 1. Upsample both signals to target fs (1000 Hz) to align timestamps
    t_new_acc, acc_up = upsample_signal(t_acc, acc_axis, target_fs=fs)
    t_new_gyro, gyro_up = upsample_signal(t_gyro, gyro_axis, target_fs=fs)
    
    # Align lengths to the minimum overlapping range
    t_min = max(t_new_acc[0], t_new_gyro[0])
    t_max = min(t_new_acc[-1], t_new_gyro[-1])
    
    t_grid = np.arange(t_min, t_max, 1000.0 / fs)
    acc_aligned = interp1d(t_new_acc, acc_up, kind='linear')(t_grid)
    gyro_aligned = interp1d(t_new_gyro, gyro_up, kind='linear')(t_grid)
    
    # 2. Apply 20 Hz LPF to remove DC bias and low-frequency motion
    acc_filtered = apply_filter(acc_aligned, cutoff=20.0, fs=fs, btype='low')
    gyro_filtered = apply_filter(gyro_aligned, cutoff=20.0, fs=fs, btype='low')
    
    # 3. Multiply signals
    multiplier_output = acc_filtered * gyro_filtered
    
    # 4. Apply another LPF (e.g., 5 Hz) to extract the DC bias component
    dc_bias_envelope = apply_filter(multiplier_output, cutoff=5.0, fs=fs, btype='low')
    dc_bias_envelope = np.abs(dc_bias_envelope) # Ensure envelope is positive
    
    # 5. Otsu thresholding to find boundaries
    threshold = otsu_threshold_1d(dc_bias_envelope)
    
    # Determine active regions
    active = dc_bias_envelope > threshold
    
    # Find active region changes
    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    
    # Handle edge conditions
    if active[0]:
        starts = np.insert(starts, 0, 0)
    if active[-1]:
        ends = np.append(ends, len(active) - 1)
        
    segments = []
    pad_samples = int(fs / 5) # Fs / 5 is 200 samples
    
    for start, end in zip(starts, ends):
        # Move boundaries by Fs / 5 samples forward and backward
        start_padded = max(0, start - pad_samples)
        end_padded = min(len(active) - 1, end + pad_samples)
        
        # We only keep segments that span at least some minimal duration
        if (end_padded - start_padded) > int(fs * 0.1): # at least 100 ms
            segments.append((t_grid[start_padded], t_grid[end_padded], start_padded, end_padded))
            
    return t_grid, dc_bias_envelope, threshold, segments
