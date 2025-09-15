from datetime import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class Pupil:
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    radius: float
    timestamp: datetime