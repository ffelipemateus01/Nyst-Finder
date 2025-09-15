from dataclasses import dataclass

@dataclass(frozen=True)
class Nystagmus:
    direction: str
    slow_phase_velocity: list[float]
    fast_phase_velocity: list[float]
    amplitude: list[float]
    avg_slow_phase_velocity: float
    max_slow_phase_velocity: float
    frequency: float
    duration: float
    beats: int
    beat_points: list[tuple[float, float]]