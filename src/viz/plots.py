import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd
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
    
    Args:
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
