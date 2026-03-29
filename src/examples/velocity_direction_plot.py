"""
Построение графиков модуля скорости и направления движения.

Субграфик б): Модуль скорости
    - GPS  — красные точки (только модуль, через GPS_to_V)
    - INS  — чёрная линия (модуль, через INS_to_V)

Субграфик в): Направление (угловая скорость рыскания из INS)
    - INS  — чёрная линия

Использование:
    python -m src.examples.velocity_direction_plot
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from src.config import GPS_DATA_FILE, INSPVA_DATA_FILE
from src.odometry import GPS_to_V, INS_to_V

# ── Загрузка и подготовка данных ───────────────────────────────────────────────

print("Загрузка GPS...")
gps_raw = pd.read_csv(GPS_DATA_FILE, header=None,
                      names=["timestamp", "lat", "lon", "height",
                             "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12"])
gps_raw = gps_raw.reset_index(drop=True)
# GPS_to_V модифицирует timestamp in-place (нс → с)
Vx_gps, Vy_gps, ts_gps = GPS_to_V(gps_raw[["timestamp", "lat", "lon", "height"]].copy())
speed_gps = np.sqrt(Vx_gps**2 + Vy_gps**2)

print("Загрузка INS (INSPVA)...")
ins_raw = pd.read_csv(INSPVA_DATA_FILE, header=None,
                      names=["timestamp", "latitude", "longitude", "height",
                             "north_velocity", "east_velocity", "up_velocity",
                             "roll", "pitch", "azimuth", "status"])
ins_raw = ins_raw.reset_index(drop=True)
# Сохраняем азимут до вызова INS_to_V (тот меняет timestamp in-place)
azimuth_raw = ins_raw["azimuth"].values
ts_ns_raw   = ins_raw["timestamp"].values.astype(np.float64)

# INS_to_V модифицирует timestamp in-place (нс → с)
Vx_ins, Vy_ins, ts_ins = INS_to_V(ins_raw[["timestamp", "latitude", "longitude",
                                             "north_velocity", "east_velocity",
                                             "up_velocity"]].copy())
speed_ins = np.sqrt(Vx_ins**2 + Vy_ins**2)

# Направление: угловая скорость рыскания из азимута INSPVA
az_unwrap = np.unwrap(np.radians(azimuth_raw)) * 180.0 / np.pi
yaw_rate  = np.gradient(az_unwrap, ts_ins)

# ── Приведение к единой временной оси (начало движения) ───────────────────────
t0_ins = ts_ins[0]
ts_ins_rel = ts_ins - t0_ins
ts_gps_rel = ts_gps - t0_ins

# Отфильтровать GPS-выбросы и точки до начала движения
speed_threshold = 0.5
moving_ins = speed_ins > speed_threshold
if moving_ins.any():
    t_start = ts_ins_rel[np.argmax(moving_ins)]
else:
    t_start = 0.0

gps_mask = (ts_gps_rel >= t_start) & (speed_gps < 12)
ts_gps_plot  = ts_gps_rel[gps_mask] - t_start
speed_gps_plot = speed_gps[gps_mask]

ins_mask = ts_ins_rel >= t_start
ts_ins_plot  = ts_ins_rel[ins_mask] - t_start
speed_ins_plot = uniform_filter1d(speed_ins[ins_mask], size=5)
yaw_rate_plot  = yaw_rate[ins_mask]

# ── Построение графиков ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.subplots_adjust(wspace=0.35, bottom=0.13)

# Субграфик б) — Скорость
ax1 = axes[0]
ax1.scatter(ts_gps_plot, speed_gps_plot, s=4, c="red", alpha=0.6, zorder=2, label="GPS")
ax1.plot(ts_ins_plot, speed_ins_plot, "k-", linewidth=1.5, zorder=3, label="ИНС")
ax1.set_xlabel("время, с", fontsize=11)
ax1.set_ylabel("скорость, м/с", fontsize=11)
ax1.set_xlim(0, ts_ins_plot[-1])
ax1.set_ylim(0, 10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_title("б)", loc="left", fontsize=11, fontweight="bold")

# Субграфик в) — Направление (угловая скорость рыскания)
ax2 = axes[1]
ax2.plot(ts_ins_plot, yaw_rate_plot, "k.", markersize=1.5, zorder=2)
ax2.set_xlabel("время, с", fontsize=11)
ax2.set_ylabel("угол, град/с", fontsize=11)
ax2.set_xlim(0, ts_ins_plot[-1])
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_title("в)", loc="left", fontsize=11, fontweight="bold")

# ── Сохранение ─────────────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", "velocity_direction_plot.png",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Сохранено: {out_path}")
plt.show()
