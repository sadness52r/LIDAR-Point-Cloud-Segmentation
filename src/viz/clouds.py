import open3d as o3d
import numpy as np
from typing import Optional
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
    pc: PointCloud,
    is_moving: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    obbs: Optional[list] = None,
    window_name: str = "Motion Object Segmentation",
) -> None:
    """
    Визуализация moving / static классификации для одного кадра.

    Цветовая схема:
    Static точки          -> серый
    Moving шум    (-1)    -> красный
    Moving кластеры       -> на каждый объект свой цвет
    Moving (без кластера) -> ярко красный

    Parameters:
        pc          : облако точек
        is_moving   : bool массив, True = moving
        cluster_ids : опциональный массив из cluster_moving_objects().
                      Values: -2 -> static, -1 -> шум, >=0 -> кластер
        obbs        : опциональный список OBB из compute_cluster_obbs() — каждый
                      рисуется проволочной рамкой в цвет своего кластера
        window_name : заголовок окна Open3D
    """
    n = len(pc.xyz)
    colors = np.tile(_STATIC_COLOR, (n, 1))

    if cluster_ids is not None:
        moving_noise = (cluster_ids == -1)
        colors[moving_noise] = _NOISE_COLOR

        unique_clusters = np.unique(cluster_ids[cluster_ids >= 0])
        for cid in unique_clusters:
            mask = cluster_ids == cid
            c = _CLUSTER_PALETTE[int(cid) % len(_CLUSTER_PALETTE)]
            colors[mask] = c
    else:
        colors[is_moving] = [1.0, 0.15, 0.15]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    if obbs:
        for obb in obbs:
            o3d_obb = o3d.geometry.OrientedBoundingBox(
                obb.center, obb.R, obb.extent,
            )
            o3d_obb.color = _CLUSTER_PALETTE[obb.cluster_id % len(_CLUSTER_PALETTE)].tolist()
            geometries.append(o3d_obb)

    geometries.extend(_build_legend_lines(has_clusters=cluster_ids is not None))

    # Используем Visualizer, чтобы поднять размер точки и затемнить фон
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()


def _build_legend_lines(has_clusters: bool) -> list:
    """Возвращает список Open3D LineSet объектов использующиеся как цветовая легенда."""
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