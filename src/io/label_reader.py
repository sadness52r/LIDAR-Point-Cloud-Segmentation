import numpy as np
from pathlib import Path
from typing import Tuple


def read_label(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Читает .label файл HeLiMOS / KITTI формата.

    Returns:
        semantic_labels: ndarray (N,)
        instance_ids: ndarray (N,)
    """
    path = Path(path)

    if not path.exists():
      raise FileNotFoundError(f"Файл не найден: {path}")

    labels = np.fromfile(path, dtype=np.uint32)

    semantic = labels & 0xFFFF
    instance = labels >> 16

    return semantic, instance