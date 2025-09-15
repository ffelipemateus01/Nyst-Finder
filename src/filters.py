from scipy.signal import butter, filtfilt, medfilt
import numpy as np
from scipy.ndimage import grey_opening

def highpass_filter(signal, cutoff_hz = 1.0, fs = 120):
    nyq = 0.5 * fs
    b, a = butter(2, cutoff_hz / nyq, btype='high')
    return filtfilt(b, a, signal)

def med_baseline_subtract(signal, fs=120.0, baseline_window_s=1.0):
    window_samples = int(round(baseline_window_s * fs))
    if window_samples % 2 == 0:
        window_samples += 1
    if window_samples < 3:
        window_samples = 3
    baseline = medfilt(signal, kernel_size=window_samples)
    clipped = signal - baseline
    clipped = np.where(clipped > 0, clipped, 0.0)
    return clipped, baseline

def lowpass_baseline_subtract(signal, fs=120.0, baseline_cutoff_hz=1.0, order=3):
    nyq = 0.5 * fs
    wn = baseline_cutoff_hz / nyq
    b, a = butter(order, wn, btype='low')
    baseline = filtfilt(b, a, signal)
    clipped = signal - baseline
    clipped = np.where(clipped > 0, clipped, 0.0)
    return clipped, baseline

def morph_baseline_subtract(signal, fs=120.0, footprint_s=0.6):
    footprint = int(round(footprint_s * fs))
    if footprint < 1:
        footprint = 1
    baseline = grey_opening(signal, size=footprint)
    clipped = signal - baseline
    clipped = np.where(clipped > 0, clipped, 0.0)
    return clipped, baseline