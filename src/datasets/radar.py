from src.core.pointcloud import PointCloud
from src.io.radar_reader import read_radar_bin
import numpy as np
from typing import Optional

def load_radar_frame(bin_path: str) -> PointCloud:
    xyz, extra = read_radar_bin(bin_path)

    intensity: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None

    if extra is not None and extra.shape[1] >= 1:
        velocity = extra[:, 0].astype(np.float32)

    if extra is not None and extra.shape[1] >= 3:
        intensity = extra[:, 2].astype(np.float32)

    return PointCloud(xyz=xyz, intensity=intensity, velocity=velocity)
