import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPS_MAP_FILE


def _find_closest_camera_image(camera_dir: str, lidar_ts: int):
    """
    Finds the stereo camera image whose filename timestamp is closest
    to lidar_ts (both in nanoseconds).  Returns the full path or None.
    """
    try:
        files = [f for f in os.listdir(camera_dir) if f.lower().endswith(".png")]
        candidates = []
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem.isdigit():
                candidates.append((int(stem), f))
        if not candidates:
            return None
        closest = min(candidates, key=lambda x: abs(x[0] - lidar_ts))
        return os.path.join(camera_dir, closest[1])
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR processing CLI")
    parser.add_argument(
    "--dataset",
    choices=["helimos", "helipr", "hercules"],
    required=True)
    parser.add_argument("--bin", type=str, required=False)
    parser.add_argument("--ins", type=str, required=False, help="Путь к INS CSV файлу")
    parser.add_argument("--imu", type=str, required=False, help="Путь к IMU CSV файлу")
    parser.add_argument("--gps", type=str, required=False, help="Путь к GPS CSV файлу")
    parser.add_argument("--radar", type=str, required=False, help="Путь к Radar BIN файлу")
    parser.add_argument("--action",
                        choices=["cloud", "velocity", "ego-velocity", "map", "track", "accel",
                                 "mos", "mos-train", "mos-sequence"],
                        required=True)
    parser.add_argument("--output", type=str, required=False, default=GPS_MAP_FILE,
                        help=f"Путь для сохранения HTML карты GPS (по умолчанию: {GPS_MAP_FILE})")

    # MOS-specific arguments
    parser.add_argument("--model", type=str, required=False, default="models/mos_rf.pkl",
                        help="Путь к файлу модели MOS (по умолчанию: models/mos_rf.pkl)")
    parser.add_argument("--sensor", type=str, required=False, default="Velodyne",
                        choices=["Velodyne", "Ouster", "Avia", "Aeva"],
                        help="Тип сенсора для HeLiMOS (по умолчанию: Velodyne)")
    parser.add_argument("--split", type=str, required=False, default="train",
                        choices=["train", "val", "test"],
                        help="Раздел датасета для обучения (по умолчанию: train)")
    parser.add_argument("--max-frames", type=int, required=False, default=None,
                        help="Максимальное число кадров для обучения/инференса")
    parser.add_argument("--n-frames", type=int, required=False, default=5,
                        help="Число кадров для mos-sequence (по умолчанию: 5)")
    parser.add_argument("--sequence", type=str, required=False,
                        help="Путь к корню датасета Deskewed_LiDAR для mos-sequence")
    parser.add_argument("--n-context", type=int, required=False, default=3,
                        help="Размер временного окна для temporal MOS (по умолчанию: 3)")
    parser.add_argument("--threshold", type=float, required=False, default=0.85,
                        help="Порог P(moving) для RF классификатора (по умолчанию: 0.85)")
    parser.add_argument("--inlier-threshold", type=float, required=False, default=0.5,
                        help="RANSAC inlier threshold [m/s] для Doppler MOS (по умолчанию: 0.5)")
    parser.add_argument("--camera", type=str, required=False,
                        help="Папка со снимками стерео-камеры (stereo_left). "
                             "Ближайший по временной метке кадр добавляется к MOS-графику.")
    parser.add_argument("--gpu", action="store_true",
                        help="Использовать GPU (XGBoost CUDA) вместо CPU (Random Forest) для обучения MOS")

    args = parser.parse_args()

    # ── Motion Object Segmentation (независимо от --dataset) ──────────────────

    if args.action == "mos-train":
        from src.motion_segmentation import MotionSegmenter

        data_root = args.sequence or "data/Deskewed_LiDAR"
        seg = MotionSegmenter(threshold=args.threshold, inlier_threshold=args.inlier_threshold, use_gpu=args.gpu)
        seg.train_on_helimos(
            data_root=data_root,
            sensor=args.sensor,
            split=args.split,
            max_frames=args.max_frames,
        )
        seg.save(args.model)
        return

    if args.action == "mos":
        from src.motion_segmentation import MotionSegmenter, ransac_ego_motion
        from src.viz.plots import plot_mos

        # Загрузка кадра через loader, соответствующий --dataset
        pc = None
        if args.dataset == "helipr":
            from src.datasets.helipr import load_helipr_aeva
            if not args.bin:
                print("Ошибка: для action=mos с helipr требуется --bin")
                return
            pc = load_helipr_aeva(args.bin)
        elif args.dataset == "hercules":
            if args.radar:
                from src.datasets.radar import load_radar_frame
                pc = load_radar_frame(args.radar)
            elif args.bin:
                from src.datasets.hercules import load_hercules_aeva
                pc = load_hercules_aeva(args.bin)
            else:
                print("Ошибка: для action=mos с hercules требуется --bin или --radar")
                return
        else:
            # helimos — KITTI формат
            from src.datasets.helimos import load_helimos_frame
            if not args.bin:
                print("Ошибка: для action=mos требуется --bin <путь_к_файлу.bin>")
                return
            pc = load_helimos_frame(args.bin)

        seg = MotionSegmenter(threshold=args.threshold, inlier_threshold=args.inlier_threshold, use_gpu=args.gpu)

        # Модель нужна только для сенсоров без Doppler-скорости.
        # Для Aeva/Radar (velocity != None) используется RANSAC — модель не требуется.
        if pc.velocity is None:
            if not os.path.exists(args.model):
                print(f"Ошибка: модель не найдена ({args.model}). "
                      "Для сенсоров без Doppler-скорости необходимо обучить модель (--action mos-train).")
                return
            seg.load(args.model)
        else:
            print("[MOS] Doppler velocity detected → using RANSAC (model not needed)")

        [is_moving] = seg.segment_frames([pc])

        # RANSAC кривая собственной скорости (если скорость есть)
        ego_params = None
        if pc.velocity is not None:
            ego_params, _ = ransac_ego_motion(pc, inlier_threshold=args.inlier_threshold)

        n_moving = int(is_moving.sum())
        n_total = len(is_moving)
        print(f"Moving: {n_moving}/{n_total} points ({100*n_moving/n_total:.1f}%)")

        camera_img = None
        if args.camera:
            import matplotlib.image as mpimg
            bin_stem = os.path.splitext(os.path.basename(args.bin))[0]
            if bin_stem.isdigit():
                lidar_ts = int(bin_stem)
                img_path = _find_closest_camera_image(args.camera, lidar_ts)
                if img_path:
                    camera_img = mpimg.imread(img_path)
                    print(f"Camera image: {os.path.basename(img_path)}")
                else:
                    print("Предупреждение: камерных изображений в указанной папке не найдено.")
            else:
                print("Предупреждение: имя .bin-файла не является временной меткой — камера проигнорирована.")

        plot_mos(pc, is_moving, ego_params=ego_params, camera_img=camera_img)
        return

    if args.action == "mos-sequence":
        from src.motion_segmentation import MotionSegmenter, cluster_moving_objects
        from src.datasets.helimos import load_helimos_sequence
        from src.viz.clouds import visualize_mos

        data_root = args.sequence or "data/Deskewed_LiDAR"
        seg = MotionSegmenter(threshold=args.threshold, inlier_threshold=args.inlier_threshold, use_gpu=args.gpu)
        seg.load(args.model)

        frames, _, poses = load_helimos_sequence(
            data_root=data_root,
            sensor=args.sensor,
            split=args.split,
            max_frames=args.n_frames,
            load_labels=False,
        )
        if not frames:
            print("Ошибка: кадры не найдены. Проверьте --sequence и --sensor.")
            return

        if poses is not None and len(poses) == len(frames):
            print(f"[MOS] Using temporal consistency (n_context={args.n_context})")
            is_moving_list = seg.segment_sequence(frames, poses, n_context=args.n_context)
        else:
            print("[MOS] No poses available — using per-frame segmentation")
            is_moving_list = seg.segment_frames(frames)

        cluster_ids_list = [
            cluster_moving_objects(pc, im) for pc, im in zip(frames, is_moving_list)
        ]

        total_moving = sum(int(m.sum()) for m in is_moving_list)
        total_pts = sum(len(m) for m in is_moving_list)
        print(f"Frames: {len(frames)} | Moving: {total_moving}/{total_pts} "
              f"({100*total_moving/total_pts:.1f}%)")

        # 3D визуализация сегментации кадров: без данных с камеры особо нет смысла
        # visualize_mos(frames, is_moving_list, cluster_ids_list,
        #               window_name=f"MOS — {len(frames)} frames")
        return

    # ── Стандартные действия по датасетам ────────────────────────────────────

    if args.dataset == "helimos":
        from src.datasets.helimos import load_helimos_frame
        from src.viz.clouds import visualize_point_cloud

        if not args.bin:
            print("Ошибка: для helimos требуется указать --bin <путь_к_файлу>")
            return
        pc = load_helimos_frame(args.bin)
        visualize_point_cloud(pc)

    elif args.dataset == "helipr":
        from src.datasets.helipr import load_helipr_aeva
        from src.viz.clouds import visualize_point_cloud
        from src.viz.plots import plot_velocity_vs_azimuth

        if not args.bin:
            print("Ошибка: для helipr требуется указать --bin <путь_к_файлу>")
            return
        pc = load_helipr_aeva(args.bin)
        if args.action == "velocity":
            plot_velocity_vs_azimuth(pc)
        else:
            visualize_point_cloud(pc)

    elif args.dataset == "hercules":
        # GPS
        if args.gps:
            from src.io.csv_reader import read_GPS
            from src.viz.plots import plot_gps_on_map

            if args.action == "map":
                gps_df = read_GPS(args.gps)
                if gps_df is not None:
                    plot_gps_on_map(gps_df, args.output)
            elif args.action == "ego-velocity":
                from src.odometry import GPS_to_V
                from src.viz.plots import plot_ego_velocity
                gps_df = read_GPS(args.gps)
                if gps_df is not None:
                    Vx, Vy, ts = GPS_to_V(gps_df)
                    plot_ego_velocity(Vx, Vy, ts, source="GPS")
            else:
                print("Ошибка: для --gps поддерживаются action='map' и action='ego-velocity'")

        # INS
        elif args.ins:
            from src.datasets.ins import load_ins
            from src.viz.plots import plot_ins_track

            if args.action == "track":
                ins_df = load_ins(args.ins)
                plot_ins_track(ins_df)
            elif args.action == "ego-velocity":
                from src.odometry import INS_to_V
                from src.viz.plots import plot_ego_velocity
                ins_df = load_ins(args.ins)
                Vx, Vy, ts = INS_to_V(ins_df)
                plot_ego_velocity(Vx, Vy, ts, source="INS")
            else:
                print("Ошибка: для --ins поддерживаются action='track' и action='ego-velocity'")

        # IMU
        elif args.imu:
            from src.datasets.imu import load_imu
            from src.viz.plots import plot_imu_accel

            if args.action == "accel":
                imu_df = load_imu(args.imu)
                plot_imu_accel(imu_df)
            else:
                print("Ошибка: для --imu поддерживается только action='accel'")

        # Radar
        elif args.radar:
            from src.datasets.radar import load_radar_frame
            from src.viz.clouds import visualize_point_cloud
            from src.viz.plots import plot_velocity_vs_azimuth

            pc = load_radar_frame(args.radar)
            if args.action == "velocity":
                plot_velocity_vs_azimuth(pc)
            else:
                visualize_point_cloud(pc)

        # Aeva LiDAR (по умолчанию --bin)
        elif args.bin:
            from src.datasets.hercules import load_hercules_aeva
            from src.viz.clouds import visualize_point_cloud
            from src.viz.plots import plot_velocity_vs_azimuth

            pc = load_hercules_aeva(args.bin)
            if args.action == "velocity":
                plot_velocity_vs_azimuth(pc)
            else:
                visualize_point_cloud(pc)

        else:
            print("Ошибка: для hercules укажите один из: --bin, --radar, --gps, --ins, --imu")


if __name__ == "__main__":
    main()