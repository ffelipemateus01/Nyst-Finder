import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pupil_with_nystagmus(df: pd.DataFrame, nystagmus= None):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time_sec'], df['pos_x_smooth'], label='pos_x')
    plt.plot(df['time_sec'], df['pos_y_smooth'], label='pos_y')
    #plt.plot(df['time_sec'], df['speed'], label='speed', linestyle='--', alpha=0.4, color='red')
    plt.plot(df['time_sec'], df['speed_baseline'], label='baseline')
    plt.plot(df['time_sec'], df['speed_clipped'], '--', label='speed_clipped', alpha=0.35, color='red')
    start_time = df['timestamp'].iloc[0]

    if nystagmus:
        for start, end in nystagmus.beat_points:
            start_sec = (start - start_time).total_seconds()
            end_sec = (end - start_time).total_seconds()
            start_pos_x = np.interp(start_sec, df['time_sec'], df['pos_x'])
            end_pos_x = np.interp(end_sec, df['time_sec'], df['pos_x'])
            plt.scatter(start_sec, start_pos_x, color='green', s=50, zorder=5)
            plt.scatter(end_sec, end_pos_x, color='red', s=50, zorder=5)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Position / Speed')
    plt.title('Pupil Signal with Nystagmus Beats')
    plt.legend()
    plt.grid(True)
    plt.show()