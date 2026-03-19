import open3d as o3d
import numpy as np
from typing import List, Optional
from src.core.pointcloud import PointCloud


def visualize_point_cloud(pc: PointCloud) -> None:
    """
    Визуализация облака точек без семантики.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.xyz)

    if pc.intensity is not None:
        i = pc.intensity
        i_norm = (i - i.min()) / (i.max() - i.min() + 1e-6)
        colors = np.zeros((len(i_norm), 3))
        colors[:, 0] = i_norm
        colors[:, 2] = 1.0 - i_norm
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


# Palette for up to 20 moving-object clusters (RGB 0–1)
_CLUSTER_PALETTE = np.array([
    [1.00, 0.20, 0.20],  # red
    [1.00, 0.60, 0.00],  # orange
    [1.00, 1.00, 0.00],  # yellow
    [0.00, 0.90, 0.20],  # green
    [0.00, 0.80, 1.00],  # cyan
    [0.60, 0.20, 1.00],  # violet
    [1.00, 0.40, 0.80],  # pink
    [0.80, 1.00, 0.40],  # lime
    [0.40, 0.80, 1.00],  # sky
    [1.00, 0.80, 0.40],  # peach
    [0.40, 1.00, 0.80],  # mint
    [0.80, 0.40, 1.00],  # purple
    [1.00, 0.60, 0.60],  # salmon
    [0.60, 1.00, 0.60],  # light green
    [0.60, 0.60, 1.00],  # lavender
    [1.00, 0.80, 0.00],  # gold
    [0.00, 1.00, 0.80],  # teal
    [0.80, 0.00, 1.00],  # magenta
    [0.00, 0.60, 1.00],  # blue
    [1.00, 0.00, 0.60],  # rose
], dtype=np.float32)

_STATIC_COLOR  = np.array([0.55, 0.55, 0.55], dtype=np.float32)   # grey
_NOISE_COLOR   = np.array([1.00, 0.30, 0.30], dtype=np.float32)   # dim red


def visualize_mos(
    pcs: List[PointCloud],
    is_moving_list: List[np.ndarray],
    cluster_ids_list: Optional[List[np.ndarray]] = None,
    window_name: str = "Motion Object Segmentation",
) -> None:
    """
    Visualise moving / static classification for one or more frames.

    Colour scheme:
    Static points       -> grey
    Moving noise  (-1)  -> dim red  (only when cluster_ids supplied)
    Moving clusters     -> distinct colours per object
    Moving (no cluster) -> bright red

    Parameters:
        pcs              : list of PointCloud objects (one or more frames)
        is_moving_list   : list of bool arrays, True = moving (one per frame)
        cluster_ids_list : optional list of int32 arrays from cluster_moving_objects(). 
                        Values:
                        -2 -> static, -1 -> noise, ≥0 -> cluster
        window_name      : Open3D window title
    """
    geometries = []
    frame_gap = 0.0   # will be set from first frame bounding box

    for fi, (pc, is_moving) in enumerate(zip(pcs, is_moving_list)):
        n = len(pc.xyz)
        colors = np.tile(_STATIC_COLOR, (n, 1))

        if cluster_ids_list is not None:
            cids = cluster_ids_list[fi]
            moving_noise = (cids == -1)
            colors[moving_noise] = _NOISE_COLOR

            unique_clusters = np.unique(cids[cids >= 0])
            for cid in unique_clusters:
                mask = cids == cid
                c = _CLUSTER_PALETTE[int(cid) % len(_CLUSTER_PALETTE)]
                colors[mask] = c
        else:
            # No cluster info: moving → red, static → grey
            colors[is_moving] = [1.0, 0.15, 0.15]

        pcd = o3d.geometry.PointCloud()
        # Offset successive frames along Y so they don't overlap
        if fi == 0:
            y_range = float(pc.xyz[:, 1].max() - pc.xyz[:, 1].min())
            frame_gap = y_range * 1.2 if y_range > 0 else 60.0
        shifted_xyz = pc.xyz.copy()
        shifted_xyz[:, 1] += fi * frame_gap
        pcd.points = o3d.utility.Vector3dVector(shifted_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    legend_lines = _build_legend_lines(has_clusters=cluster_ids_list is not None)
    geometries.extend(legend_lines)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720,
    )


def _build_legend_lines(has_clusters: bool) -> list:
    """Return a list of Open3D LineSet objects used as a colour legend."""
    entries = [
        ("Static",  _STATIC_COLOR),
        ("Moving",  np.array([1.0, 0.15, 0.15], dtype=np.float32)),
    ]
    if has_clusters:
        entries.append(("Noise", _NOISE_COLOR))
        for ci in range(3):
            entries.append((f"Cluster {ci}", _CLUSTER_PALETTE[ci]))

    line_sets = []
    for i, (_, color) in enumerate(entries):
        z = -5.0 + i * 2.0
        pts = np.array([[80.0, 0.0, z], [85.0, 0.0, z]], dtype=np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        ls.colors = o3d.utility.Vector3dVector([color.tolist()])
        line_sets.append(ls)
    return line_sets