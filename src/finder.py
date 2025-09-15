from src.contexts.nystagmus import Nystagmus
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
import math


class NystFinder():
    # def to_dataframe(self, signal: list[Pupil]) -> pd.DataFrame:
    #     df = pd.DataFrame([{
    #         'timestamp': p.timestamp,
    #         'pos_x': p.pos_x,
    #         'pos_y': p.pos_y,
    #         'vel_x': p.vel_x,
    #         'vel_y': p.vel_y,
    #         'radius': p.radius
    #     } for p in signal])
    #     return df
    
    def detect_peaks(self, df:pd.DataFrame, fs=120.0, min_dist_s=0.1, prominence_factor=1.5):
        signal = df['speed_clipped'].to_numpy()
        if signal.size == 0:
            df['is_fast'] = False
            return None, df, []
        noise_level = np.std(signal)
        min_prominence = prominence_factor * noise_level
        min_distance = int(round(min_dist_s * fs))
        peaks, _ = find_peaks(signal, prominence=min_prominence, distance=min_distance)

        if peaks.size == 0:
            df['is_fast'] = False
            return None, df, []

        _, _, left_ips, right_ips = peak_widths(signal, peaks, rel_height=0.5)
        left_ix = np.round(left_ips).astype(int)
        right_ix = np.round(right_ips).astype(int)
        left_ix = np.clip(left_ix, 0, len(signal) - 1)
        right_ix = np.clip(right_ix, 0, len(signal) - 1)

        intervals = [(int(l), int(r)) for l, r in zip(left_ix, right_ix)]

        df['is_fast'] = False
        col_pos = df.columns.get_loc('is_fast')
        for l, r in intervals:
            df.iloc[l:r+1, col_pos] = True

        return peaks, df, intervals
    
    def classify_direction(self, angle: float) -> str:
        if -45 <= angle <= 45:
            return "right"
        if 135 <= angle or angle <= -135:
            return "left"
        if 45 < angle < 135:
            return "up"
        if -135 < angle < -45:
            return "down"
        return "oblique"

    def build_nystagmus(self, df: pd.DataFrame, peaks: np.ndarray, intervals) -> Nystagmus:
        start_time = df['timestamp'].iloc[0].timestamp()
        end_time = df['timestamp'].iloc[-1].timestamp()
        duration = end_time - start_time
        if peaks is None or len(peaks) == 0:
            return None
        beats = len(peaks)
        frequency = beats / duration if duration > 0 else 0

        slow_phase = df['speed_clipped'].drop(df.index[peaks])
        fast_phase = df['speed_clipped'].iloc[peaks]

        fast_vx = df["vel_x"].iloc[peaks].mean()
        fast_vy = df["vel_y"].iloc[peaks].mean()
        angle = math.degrees(math.atan2(fast_vy, fast_vx))

        beat_points = []
        signal = df['speed_clipped'].to_numpy()
        baseline = df['speed_baseline'].to_numpy()

        for peak in peaks:
            start_idx = peak
            while start_idx > 0 and signal[start_idx] >= baseline[start_idx]:
                start_idx -= 1
            end_idx = peak
            while end_idx < len(signal)-1 and signal[end_idx] >= baseline[end_idx]:
                end_idx += 1
            beat_points.append((df['timestamp'].iloc[start_idx], df['timestamp'].iloc[end_idx]))

        return Nystagmus(
            direction=self.classify_direction(angle),
            slow_phase_velocity=slow_phase.tolist(),
            fast_phase_velocity=fast_phase.tolist(),
            amplitude=(df['pos_x'].iloc[peaks].diff().abs().dropna().tolist()),
            avg_slow_phase_velocity=float(slow_phase.mean()) if not slow_phase.empty else 0,
            max_slow_phase_velocity=float(slow_phase.max()) if not slow_phase.empty else 0,
            frequency=frequency,
            duration=duration,
            beats=beats,
            beat_points=beat_points
        )