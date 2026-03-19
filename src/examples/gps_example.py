"""
Пример использования функциональности для чтения и визуализации GPS данных
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from src.io.csv_reader import read_GPS
from src.viz.plots import plot_gps_on_map
from src.config import GPS_DATA_FILE, GPS_TRAJECTORY_MAP_FILE

print("=" * 60)
print("Чтение GPS данных...")
print("=" * 60)

# Читаем GPS данные
GPS = read_GPS(GPS_DATA_FILE)
gps_df = GPS

# Выводим информацию о загруженных данных
print(f"\nПервые 5 строк GPS данных:")
print(gps_df.head())
print(f"\nСтатистика координат:")
print(f"Широта (lat): min={gps_df['lat'].min()}, max={gps_df['lat'].max()}, mean={gps_df['lat'].mean()}")
print(f"Долгота (lon): min={gps_df['lon'].min()}, max={gps_df['lon'].max()}, mean={gps_df['lon'].mean()}")
print(f"Высота (height): min={gps_df['height'].min()}, max={gps_df['height'].max()}, mean={gps_df['height'].mean()}")

# === Визуализация на Google картах ===
print("\n" + "=" * 60)
print("Визуализация GPS траектории на карте...")
print("=" * 60)

# Создаем и сохраняем интерактивную карту
plot_gps_on_map(gps_df, output_file=GPS_TRAJECTORY_MAP_FILE)

print("\n" + "=" * 60)
print("Готово!")
print("=" * 60)
print("\nГенерированные файлы:")
print(f"  - {GPS_TRAJECTORY_MAP_FILE} (полная траектория)")
print("\nОткройте эти файлы в браузере для просмотра интерактивной карты.")
