from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PointCloud:
    """
    Унифицированное представление облака точек.

    Attributes:
        xyz: ndarray формы (N, 3)
        intensity: ndarray формы (N,) или None
        velocity: ndarray формы (N,) или None
    """

    xyz: np.ndarray
    intensity: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None