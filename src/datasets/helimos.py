import os
from typing import List, Optional, Tuple

import numpy as np

from src.core.pointcloud import PointCloud
from src.io.bin_reader import read_kitti_bin
from src.io.label_reader import read_label


def load_helimos_frame(bin_path: str) -> PointCloud:
    """
    Загружает один кадр HeLiMOS.
    """
    return read_kitti_bin(bin_path)


def load_helimos_sequence(
    data_root: str,
    sensor: str = "Velodyne",
    frame_ids: Optional[List[int]] = None,
    split: str = "train",
    max_frames: Optional[int] = None,
    load_labels: bool = True,
) -> Tuple[List[PointCloud], Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Загружает последовательность кадров HeLiMOS.

    Parameters:
        data_root   : корень датасета Deskewed_LiDAR
        sensor      : 'Velodyne', 'Ouster', 'Avia' или 'Aeva'
        frame_ids   : конкретные ID кадров; если None — берётся из split-файла
        split       : 'train', 'val' или 'test' (используется если frame_ids=None)
        max_frames  : ограничение числа кадров
        load_labels : загружать ли метки (если файл существует)

    Returns:
        frames      : список PointCloud
        labels_list : список массивов semantic-меток (uint16), или None
        poses       : (N, 12) массив поз из poses.txt, или None
    """
    if frame_ids is None:
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            frame_ids = [int(ln.strip()) for ln in f if ln.strip()]

    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    sensor_dir = os.path.join(data_root, sensor)
    vel_dir = os.path.join(sensor_dir, "velodyne")
    lbl_dir = os.path.join(sensor_dir, "labels")
    poses_path = os.path.join(sensor_dir, "poses.txt")

    # Load poses if available
    poses: Optional[np.ndarray] = None
    if os.path.exists(poses_path):
        all_poses = np.loadtxt(poses_path)  # shape (total_frames, 12)
        poses = np.array([all_poses[fid] for fid in frame_ids if fid < len(all_poses)])

    frames: List[PointCloud] = []
    labels_list: List[np.ndarray] = []
    valid_frame_ids: List[int] = []

    for fid in frame_ids:
        bin_path = os.path.join(vel_dir, f"{fid:06d}.bin")
        if not os.path.exists(bin_path):
            continue

        pc = read_kitti_bin(bin_path)
        frames.append(pc)
        valid_frame_ids.append(fid)

        if load_labels:
            lbl_path = os.path.join(lbl_dir, f"{fid:06d}.label")
            if os.path.exists(lbl_path):
                semantic, _ = read_label(lbl_path)
                labels_list.append(semantic.astype(np.uint16))
            else:
                labels_list.append(None)

    # Re-filter poses to only valid frames
    if poses is not None and len(valid_frame_ids) != len(frame_ids):
        all_poses_full = np.loadtxt(poses_path)
        poses = np.array(
            [all_poses_full[fid] for fid in valid_frame_ids if fid < len(all_poses_full)]
        )

    return frames, (labels_list if load_labels else None), poses