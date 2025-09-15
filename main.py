from src.plotter import plot_pupil_with_nystagmus
from src.finder import NystFinder
import os
import pandas as pd
import numpy as np
from src.filters import med_baseline_subtract, lowpass_baseline_subtract, morph_baseline_subtract, highpass_filter
from scipy.signal import savgol_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
signal_file_1 = os.path.join(BASE_DIR, "data", "pupil_data1.csv")
#signal_file_1 = os.path.join(BASE_DIR, "data", "pupil_data2.csv")

df = pd.read_csv(signal_file_1, parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
#df['timestamp'] = pd.to_datetime(df['timestamp'])
start_time = df['timestamp'].iloc[0]
df['time_sec'] = (df['timestamp'] - start_time).dt.total_seconds()

df['pos_x_smooth'] = savgol_filter(df['pos_x'], window_length=11, polyorder=4)
df['pos_y_smooth'] = savgol_filter(df['pos_y'], window_length=11, polyorder=4)
# df['pos_x_smooth'] = df['pos_x'].rolling(7, center=True).mean()
# df['pos_y_smooth'] = df['pos_y'].rolling(7, center=True).mean()
#df['pos_x_hp'] = highpass(df['pos_x_smooth'])
#df['pos_y_hp'] = highpass(df['pos_y_smooth'])
df['vel_x_hp'] = np.gradient(df['pos_x_smooth'], df['time_sec'])
df['vel_y_hp'] = np.gradient(df['pos_y_smooth'], df['time_sec'])
df['speed'] = np.sqrt(df['vel_x_hp']**2 + df['vel_y_hp']**2)
speed_clipped, baseline = med_baseline_subtract(df['speed'].to_numpy())
df['speed_clipped'] = speed_clipped
df['speed_baseline'] = baseline

finder = NystFinder()
peaks, df, intervals = finder.detect_peaks(df)
nystagmus = finder.build_nystagmus(df, peaks, intervals)

if nystagmus:
    print(nystagmus)
plot_pupil_with_nystagmus(df, nystagmus)
