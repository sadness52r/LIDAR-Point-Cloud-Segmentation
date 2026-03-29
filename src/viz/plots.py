import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd
from scipy.ndimage import uniform_filter1d
from src.core.pointcloud import PointCloud
from src.config import MAP_ZOOM_LEVEL, MAP_TILES, POLYLINE_COLOR, POLYLINE_WEIGHT, START_MARKER_COLOR, END_MARKER_COLOR


def plot_velocity_vs_azimuth(pc: PointCloud) -> None:
    """
    График радиальной скорости от азимута.
    """
    if pc.velocity is None:
        raise ValueError("Velocity channel is missing")

    x, y = pc.xyz[:, 0], pc.xyz[:, 1]
    azimuth = np.degrees(np.arctan2(y, x))

    plt.figure(figsize=(8, 4))
    plt.scatter(azimuth, pc.velocity, s=1)
    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Radial velocity [m/s]")
    plt.grid(True)
    plt.show()


def plot_gps_on_map(gps_df, output_file='gps_map.html'):
    """
    Визуализация GPS траектории на интерактивной карте Google Maps (через Folium).
    
    Parameters:
        gps_df: DataFrame с GPS данными (должны быть колонки 'lat' и 'lon')
        output_file: путь для сохранения HTML файла карты
    """
    if 'lat' not in gps_df.columns or 'lon' not in gps_df.columns:
        raise ValueError("GPS DataFrame должен содержать колонки 'lat' и 'lon'")
    
    # Вычисляем центр карты
    center_lat = gps_df['lat'].mean()
    center_lon = gps_df['lon'].mean()
    
    # Создаем карту
    map_gps = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=MAP_ZOOM_LEVEL,
        tiles=MAP_TILES
    )
    
    # Добавляем линию траектории
    coords = list(zip(gps_df['lat'], gps_df['lon']))
    folium.PolyLine(
        coords,
        color=POLYLINE_COLOR,
        weight=POLYLINE_WEIGHT,
        opacity=0.7,
        popup='GPS траектория'
    ).add_to(map_gps)
    
    # Добавляем стартовую точку
    folium.CircleMarker(
        location=[gps_df['lat'].iloc[0], gps_df['lon'].iloc[0]],
        radius=8,
        popup='Старт',
        color=START_MARKER_COLOR,
        fill=True,
        fillColor=START_MARKER_COLOR,
        fillOpacity=0.8
    ).add_to(map_gps)
    
    # Добавляем финальную точку
    folium.CircleMarker(
        location=[gps_df['lat'].iloc[-1], gps_df['lon'].iloc[-1]],
        radius=8,
        popup='Конец',
        color=END_MARKER_COLOR,
        fill=True,
        fillColor=END_MARKER_COLOR,
        fillOpacity=0.8
    ).add_to(map_gps)
    
    # Сохраняем карту
    map_gps.save(output_file)
    print(f"Карта сохранена в {output_file}")
    
    return map_gps

def plot_velocity_comparison(gps_path: str, ins_path: str, output_path: str) -> None:
    """
    Графики модуля скорости и угловой скорости рыскания.

    Левый график — скорость, м/с:
        GPS  — красные точки (из конечных разностей ECEF-координат через GPS_to_V)
        ИНС  — чёрная линия (из NED-компонент скорости через INS_to_V)

    Правый график — угловая скорость рыскания, °/с:
        ИНС  — чёрные точки (производная азимута из INSPVA)

    Parameters:
        gps_path    : путь к GPS CSV (колонки: timestamp_нс, lat, lon, height, ...)
        ins_path    : путь к INSPVA CSV (колонки: timestamp_нс, lat, lon, height,
                      Vn, Ve, Vu, roll, pitch, azimuth, status)
        output_path : путь для сохранения PNG
    """
    from src.odometry import GPS_to_V, INS_to_V

    # Загрузка
    gps_raw = pd.read_csv(gps_path, header=None,
                          names=["timestamp", "lat", "lon", "height",
                                 "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12"])
    gps_raw = gps_raw.reset_index(drop=True)

    ins_raw = pd.read_csv(ins_path, header=None,
                          names=["timestamp", "latitude", "longitude", "height",
                                 "north_velocity", "east_velocity", "up_velocity",
                                 "roll", "pitch", "azimuth", "status"])
    ins_raw = ins_raw.reset_index(drop=True)

    azimuth_raw = ins_raw["azimuth"].values

    # Вычисление скоростей
    Vx_gps, Vy_gps, ts_gps = GPS_to_V(gps_raw[["timestamp", "lat", "lon", "height"]].copy())
    speed_gps = np.sqrt(Vx_gps**2 + Vy_gps**2)

    Vx_ins, Vy_ins, ts_ins = INS_to_V(ins_raw[["timestamp", "latitude", "longitude",
                                                 "north_velocity", "east_velocity",
                                                 "up_velocity"]].copy())
    speed_ins = np.sqrt(Vx_ins**2 + Vy_ins**2)

    # Угловая скорость рыскания из азимута INSPVA
    az_unwrap = np.unwrap(np.radians(azimuth_raw)) * 180.0 / np.pi
    yaw_rate = np.gradient(az_unwrap, ts_ins)

    # Временная ось: начало с первого движения
    t0 = ts_ins[0]
    ts_ins_rel = ts_ins - t0
    ts_gps_rel = ts_gps - t0

    moving = speed_ins > 0.5
    t_start = ts_ins_rel[np.argmax(moving)] if moving.any() else 0.0

    gps_mask = (ts_gps_rel >= t_start) & (speed_gps < 12)
    ts_gps_plot    = ts_gps_rel[gps_mask] - t_start
    speed_gps_plot = speed_gps[gps_mask]

    ins_mask = ts_ins_rel >= t_start
    ts_ins_plot    = ts_ins_rel[ins_mask] - t_start
    speed_ins_plot = uniform_filter1d(speed_ins[ins_mask], size=5)
    yaw_rate_plot  = yaw_rate[ins_mask]

    # Построение
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35, bottom=0.13)

    ax1 = axes[0]
    ax1.scatter(ts_gps_plot, speed_gps_plot, s=4, c="red", alpha=0.6, zorder=2, label="GPS")
    ax1.plot(ts_ins_plot, speed_ins_plot, "k-", linewidth=1.5, zorder=3, label="ИНС")
    ax1.set_xlabel("время, с", fontsize=11)
    ax1.set_ylabel("скорость, м/с", fontsize=11)
    ax1.set_xlim(0, ts_ins_plot[-1])
    ax1.set_ylim(0, 10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_title("Скорость (м/с) от времени (с) (GPS + INS)", loc="left", fontsize=11, fontweight="bold")

    ax2 = axes[1]
    ax2.plot(ts_ins_plot, yaw_rate_plot, "k.", markersize=1.5, zorder=2)
    ax2.set_xlabel("время, с", fontsize=11)
    ax2.set_ylabel("угол, град/с", fontsize=11)
    ax2.set_xlim(0, ts_ins_plot[-1])
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title("Угол (град) от времени (с)", loc="left", fontsize=11, fontweight="bold")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {output_path}")
    plt.show()


def plot_ins_track(ins_df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(ins_df["longitude"], ins_df["latitude"])
    plt.grid(True)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("INS track")
    plt.show()

def plot_imu_accel(imu_df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(imu_df["timestamp"], imu_df["acc_x"], label="acc_x")
    plt.plot(imu_df["timestamp"], imu_df["acc_y"], label="acc_y")
    plt.plot(imu_df["timestamp"], imu_df["acc_z"], label="acc_z")
    plt.grid(True)
    plt.xlabel("timestamp")
    plt.ylabel("acc (m/s^2)")
    plt.title("IMU acceleration")
    plt.legend()
    plt.show()


# MOS 2-D plots
# Palette matching the Open3D one in clouds.py
_MOS_PALETTE = [
    "#ff3333",  # red
    "#ff9900",  # orange
    "#ffff00",  # yellow
    "#00e633",  # green
    "#00ccff",  # cyan
    "#9933ff",  # violet
    "#ff66cc",  # pink
    "#ccff66",  # lime
    "#66ccff",  # sky
    "#ffcc66",  # peach
]


def plot_mos(
    pc: PointCloud,
    is_moving: np.ndarray,
    ego_params: "np.ndarray | None" = None,
    title: str = "",
    camera_img: "np.ndarray | None" = None,
) -> None:
    """
    2D-графики MOS-результата (static vs moving) с опциональным кадром камеры.

    Панели (без камеры):
      Левый  — radial velocity vs azimuth (только при наличии velocity)
      Правый — bird's-eye view (x, y)

    Панели (с камерой):
      Верхний ряд — velocity vs azimuth + bird's-eye view
      Нижний ряд  — изображение с камеры (на всю ширину)

    Parameters:
        pc          : PointCloud (xyz обязательно, velocity — опционально)
        is_moving   : bool mask
        ego_params  : [Vx, Vy] для отрисовки RANSAC-кривой (опционально)
        title       : заголовок окна
        camera_img  : RGB-массив (H, W, 3) с кадром камеры (опционально)
    """
    x, y, z = pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]
    azimuth_deg = np.degrees(np.arctan2(y, x))
    has_velocity = pc.velocity is not None

    static_mask = ~is_moving

    # Build figure layout
    if camera_img is not None:
        if has_velocity:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.65], hspace=0.35, wspace=0.3)
            ax_vel = fig.add_subplot(gs[0, 0])
            ax_bev = fig.add_subplot(gs[0, 1])
            ax_cam = fig.add_subplot(gs[1, :])
        else:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.65], hspace=0.35)
            ax_bev = fig.add_subplot(gs[0])
            ax_cam = fig.add_subplot(gs[1])
    else:
        if has_velocity:
            fig, (ax_vel, ax_bev) = plt.subplots(1, 2, figsize=(16, 7))
        else:
            fig, ax_bev = plt.subplots(1, 1, figsize=(9, 7))

    if title:
        fig.suptitle(title, fontsize=13)

    # Velocity vs Azimuth
    if has_velocity:
        v = pc.velocity
        ax_vel.scatter(
            azimuth_deg[static_mask], v[static_mask],
            s=0.4, c="#999999", alpha=0.5, label="static", rasterized=True,
        )
        ax_vel.scatter(
            azimuth_deg[is_moving], v[is_moving],
            s=3, c="#e63333", alpha=0.7, label="moving", rasterized=True,
        )
        if ego_params is not None:
            alpha_sweep = np.linspace(azimuth_deg.min(), azimuth_deg.max(), 500)
            alpha_rad = np.radians(alpha_sweep)
            vr_model = -ego_params[0] * np.cos(alpha_rad) - ego_params[1] * np.sin(alpha_rad)
            ax_vel.plot(alpha_sweep, vr_model, "k-", lw=1.8, label="ego-motion model")

        ax_vel.set_xlabel("Azimuth [deg]")
        ax_vel.set_ylabel("Radial velocity [m/s]")
        ax_vel.set_title("Radial velocity vs Azimuth")
        ax_vel.legend(loc="lower left", fontsize=8, markerscale=3)
        ax_vel.grid(True, alpha=0.3)

    # Bird's-eye view (x, y)
    ax_bev.scatter(
        x[static_mask], y[static_mask],
        s=0.3, c="#999999", alpha=0.4, label="static", rasterized=True,
    )
    ax_bev.scatter(
        x[is_moving], y[is_moving],
        s=3, c="#e63333", alpha=0.7, label="moving", rasterized=True,
    )
    ax_bev.set_xlabel("x, m")
    ax_bev.set_ylabel("y, m")
    ax_bev.set_title("Bird's-eye view")
    ax_bev.set_aspect("equal")
    ax_bev.legend(loc="upper right", fontsize=8, markerscale=3)
    ax_bev.grid(True, alpha=0.3)

    # Camera image
    if camera_img is not None:
        ax_cam.imshow(camera_img)
        ax_cam.set_title("Stereo Left Camera", fontsize=10)
        ax_cam.axis("off")

    plt.tight_layout()
    plt.show()
