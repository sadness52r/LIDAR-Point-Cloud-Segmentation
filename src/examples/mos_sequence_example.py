"""
Покадровая MOS-визуализация Hercules Aeva + стерео-камера → PNG.

Сетка 2×2 с фиксированными осями (как в LiDAR.avi):
    камера      │ BEV (полный)
    ────────────┼──────────────
    velocity    │ BEV (зум)

Оси вычисляются один раз по первым N кадрам и дальше не меняются,
поэтому графики не скачут между кадрами.

Использование:
    python -m src.examples.mos_sequence_example \
        --aeva   03_Day/Aeva \
        --camera 03_Day/stereo_left \
        --model  models/mos_rf.pkl \
        --output output/mos_frames

    # Собрать GIF (ImageMagick):
    magick convert -delay 10 -loop 0 output/mos_frames/*.png mos.gif

    # Собрать видео (ffmpeg):
    ffmpeg -framerate 10 -i output/mos_frames/%06d.png -c:v libx264 -pix_fmt yuv420p mos.mp4
"""

import sys, os, argparse, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.datasets.hercules import load_hercules_aeva
from src.motion_segmentation import MotionSegmenter, ransac_ego_motion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_camera_index(camera_dir: str):
    """Return sorted list of (timestamp_ns, filepath) for all .png files."""
    entries = []
    for f in os.listdir(camera_dir):
        if not f.lower().endswith(".png"):
            continue
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            entries.append((int(stem), os.path.join(camera_dir, f)))
    entries.sort()
    return entries


def _find_closest(cam_index, lidar_ts: int):
    """Binary search for the closest camera timestamp."""
    if not cam_index:
        return None
    lo, hi = 0, len(cam_index) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cam_index[mid][0] < lidar_ts:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(cam_index[lo - 1][0] - lidar_ts) < abs(cam_index[lo][0] - lidar_ts):
        best = lo - 1
    return cam_index[best][1]


def _compute_fixed_limits(bin_files, n_probe=20):
    """
    Scan first n_probe frames to determine stable axis limits.
    Returns dict with azimuth, velocity, x, y ranges (with padding).
    """
    az_min, az_max = np.inf, -np.inf
    v_min, v_max = np.inf, -np.inf
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    step = max(1, len(bin_files) // n_probe)
    probes = bin_files[::step][:n_probe]

    for path in probes:
        pc = load_hercules_aeva(path)
        x, y = pc.xyz[:, 0], pc.xyz[:, 1]
        az = np.degrees(np.arctan2(y, x))

        az_min = min(az_min, np.percentile(az, 1))
        az_max = max(az_max, np.percentile(az, 99))
        x_min = min(x_min, np.percentile(x, 1))
        x_max = max(x_max, np.percentile(x, 99))
        y_min = min(y_min, np.percentile(y, 1))
        y_max = max(y_max, np.percentile(y, 99))

        if pc.velocity is not None:
            v_min = min(v_min, np.percentile(pc.velocity, 1))
            v_max = max(v_max, np.percentile(pc.velocity, 99))

    # Add 10% padding
    def pad(lo, hi):
        margin = (hi - lo) * 0.1
        return lo - margin, hi + margin

    lim = {
        "az": pad(az_min, az_max),
        "v": pad(v_min, v_max) if v_min < np.inf else (-10, 10),
        "x": pad(x_min, x_max),
        "y": pad(y_min, y_max),
    }

    # Zoom BEV: ±30 m around ego (sensor at origin)
    zoom_r = 30
    lim["x_zoom"] = (-zoom_r, zoom_r)
    lim["y_zoom"] = (-zoom_r, zoom_r)

    return lim


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_frame(fig, axes, pc, is_moving, ego_params, camera_img,
                frame_idx, n_frames, lim):
    """Clear axes and redraw a single MOS frame with fixed axis limits."""
    ax_cam, ax_bev, ax_vel, ax_zoom = axes

    for ax in axes:
        ax.clear()

    x, y = pc.xyz[:, 0], pc.xyz[:, 1]
    azimuth_deg = np.degrees(np.arctan2(y, x))
    static = ~is_moving

    # ── Top-left: Camera ──────────────────────────────────────────────────
    if camera_img is not None:
        ax_cam.imshow(camera_img)
    ax_cam.set_title("Stereo Left Camera", fontsize=9)
    ax_cam.axis("off")

    # ── Top-right: BEV full ───────────────────────────────────────────────
    ax_bev.scatter(x[static], y[static],
                   s=0.3, c="#999999", alpha=0.4, rasterized=True)
    ax_bev.scatter(x[is_moving], y[is_moving],
                   s=3, c="#e63333", alpha=0.7, rasterized=True)
    ax_bev.set_xlim(lim["x"])
    ax_bev.set_ylim(lim["y"])
    ax_bev.set_xlabel("x, m")
    ax_bev.set_ylabel("y, m")
    ax_bev.set_title("Bird's-eye view", fontsize=9)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)

    # ── Bottom-left: Velocity vs Azimuth ──────────────────────────────────
    if pc.velocity is not None:
        v = pc.velocity
        ax_vel.scatter(azimuth_deg[static], v[static],
                       s=0.4, c="#999999", alpha=0.5, rasterized=True)
        ax_vel.scatter(azimuth_deg[is_moving], v[is_moving],
                       s=3, c="#e63333", alpha=0.7, rasterized=True)
        if ego_params is not None:
            a_sweep = np.linspace(lim["az"][0], lim["az"][1], 500)
            vr = -ego_params[0] * np.cos(np.radians(a_sweep)) \
                 - ego_params[1] * np.sin(np.radians(a_sweep))
            ax_vel.plot(a_sweep, vr, "k-", lw=1.5)
    ax_vel.set_xlim(lim["az"])
    ax_vel.set_ylim(lim["v"])
    ax_vel.set_xlabel("azimuth, deg")
    ax_vel.set_ylabel("radial velocity, m/s")
    ax_vel.set_title("Radial velocity vs Azimuth", fontsize=9)
    ax_vel.grid(True, alpha=0.3)

    # ── Bottom-right: BEV zoom ────────────────────────────────────────────
    ax_zoom.scatter(x[static], y[static],
                    s=0.5, c="#999999", alpha=0.4, rasterized=True)
    ax_zoom.scatter(x[is_moving], y[is_moving],
                    s=4, c="#e63333", alpha=0.7, rasterized=True)
    ax_zoom.set_xlim(lim["x_zoom"])
    ax_zoom.set_ylim(lim["y_zoom"])
    ax_zoom.set_xlabel("x, m")
    ax_zoom.set_ylabel("y, m")
    ax_zoom.set_title("BEV (near range)", fontsize=9)
    ax_zoom.set_aspect("equal")
    ax_zoom.grid(True, alpha=0.3)

    n_mov = int(is_moving.sum())
    n_tot = len(is_moving)
    fig.suptitle(
        f"Frame {frame_idx + 1}/{n_frames}  |  "
        f"moving {n_mov}/{n_tot} ({100 * n_mov / n_tot:.1f}%)",
        fontsize=12, y=0.98,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOS покадровая визуализация Hercules → PNG")
    parser.add_argument("--aeva", type=str, required=True,
                        help="Папка с .bin кадрами Aeva (например 03_Day/Aeva)")
    parser.add_argument("--camera", type=str, required=False, default=None,
                        help="Папка stereo_left с .png кадрами камеры")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к MOS модели (нужна только для сенсоров без Допплера)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Порог P(moving) для RF (по умолчанию: 0.85)")
    parser.add_argument("--inlier-threshold", type=float, default=0.5,
                        help="RANSAC inlier threshold [m/s] для Doppler MOS (по умолчанию: 0.5)")
    parser.add_argument("--output", type=str, default="output/mos_frames",
                        help="Папка для сохранения PNG кадров (по умолчанию: output/mos_frames)")
    parser.add_argument("--start", type=int, default=0,
                        help="Начальный индекс кадра (по умолчанию: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Максимальное число кадров (по умолчанию: все)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="DPI сохраняемых изображений (по умолчанию: 120)")
    args = parser.parse_args()

    # ── Collect frames ────────────────────────────────────────────────────
    bin_files = sorted(glob.glob(os.path.join(args.aeva, "*.bin")))
    if not bin_files:
        print(f"Ошибка: .bin файлов не найдено в {args.aeva}")
        return

    bin_files = bin_files[args.start:]
    if args.max_frames is not None:
        bin_files = bin_files[:args.max_frames]
    print(f"Кадров для обработки: {len(bin_files)}")

    # ── Camera index ──────────────────────────────────────────────────────
    cam_index = []
    if args.camera:
        cam_index = _build_camera_index(args.camera)
        print(f"Найдено кадров камеры: {len(cam_index)}")

    # ── Output dir ────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── Load MOS model (only needed for non-Doppler sensors) ─────────────
    seg = MotionSegmenter(threshold=args.threshold, inlier_threshold=args.inlier_threshold)
    if args.model:
        seg.load(args.model)

    # ── Compute fixed axis limits from sample of frames ───────────────────
    print("Вычисляю диапазоны осей...")
    lim = _compute_fixed_limits(bin_files)
    print(f"  azimuth: [{lim['az'][0]:.0f}, {lim['az'][1]:.0f}] deg")
    print(f"  velocity: [{lim['v'][0]:.1f}, {lim['v'][1]:.1f}] m/s")
    print(f"  BEV x: [{lim['x'][0]:.0f}, {lim['x'][1]:.0f}] m")
    print(f"  BEV y: [{lim['y'][0]:.0f}, {lim['y'][1]:.0f}] m")

    # ── Setup figure 2×2 (fixed size, no tight bbox) ─────────────────────
    fig, ((ax_cam, ax_bev), (ax_vel, ax_zoom)) = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )
    axes = (ax_cam, ax_bev, ax_vel, ax_zoom)

    # ── Process all frames ────────────────────────────────────────────────
    n = len(bin_files)
    for i, bin_path in enumerate(bin_files):
        pc = load_hercules_aeva(bin_path)
        [is_moving] = seg.segment_frames([pc])

        ego_params = None
        if pc.velocity is not None:
            ego_params, _ = ransac_ego_motion(pc, inlier_threshold=args.inlier_threshold)

        camera_img = None
        if cam_index:
            stem = os.path.splitext(os.path.basename(bin_path))[0]
            if stem.isdigit():
                img_path = _find_closest(cam_index, int(stem))
                if img_path:
                    camera_img = mpimg.imread(img_path)

        _draw_frame(fig, axes, pc, is_moving, ego_params, camera_img, i, n, lim)

        out_path = os.path.join(args.output, f"{i:06d}.png")
        fig.savefig(out_path, dpi=args.dpi)

        print(f"\r[{i + 1}/{n}] saved {out_path}", end="", flush=True)

    plt.close(fig)
    print(f"\n\nГотово! {n} кадров сохранено в {args.output}/")
    print(f"\nСобрать GIF:   magick convert -delay 10 -loop 0 {args.output}/*.png mos.gif")
    print(f"Собрать видео: ffmpeg -framerate 10 -i {args.output}/%06d.png -c:v libx264 -pix_fmt yuv420p mos.mp4")


if __name__ == "__main__":
    main()
