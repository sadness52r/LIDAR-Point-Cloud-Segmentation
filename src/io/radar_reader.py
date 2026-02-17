import struct
import numpy as np
from typing import Tuple, Optional
import os

def read_radar_bin(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Continental radar: record = 29 bytes:
    <fffff (x,y,z,Vr,range) + <B (RCS) + <ff (az,el)
    Returns:
      xyz: (N,3)
      extra: (N,5) columns [Vr, range, RCS, azimuth, elevation]
    """
    rec = 29
    pts = []
    extra = []

    with open(path, "rb") as f:
        while True:
            chunk = f.read(rec)
            if len(chunk) < rec:
                break
            x, y, z, vr, r = struct.unpack("<fffff", chunk[:20])
            rcs = struct.unpack("<B", chunk[20:21])[0]
            az, el = struct.unpack("<ff", chunk[21:29])

            pts.append([x, y, z])
            extra.append([vr, r, float(rcs), az, el])

    xyz = np.asarray(pts, dtype=np.float32)
    extra_arr = np.asarray(extra, dtype=np.float32) if extra else None
    return xyz, extra_arr
