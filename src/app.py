import sys
import os
import argparse

# Добавляем родительскую директорию в sys.path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPS_MAP_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR processing CLI")
    parser.add_argument(
    "--dataset",
    choices=["helimos", "helipr", "hercules", "gps", "ins", "imu", "radar"],
    required=True)
    parser.add_argument("--bin", type=str, required=False)
    parser.add_argument("--ins", type=str, required=False, help="Путь к INS CSV файлу")
    parser.add_argument("--imu", type=str, required=False, help="Путь к IMU CSV файлу")
    parser.add_argument("--gps", type=str, required=False, help="Путь к GPS CSV файлу")
    parser.add_argument("--radar", type=str, required=False, help="Путь к Radar BIN файлу")
    parser.add_argument("--action", choices=["cloud", "velocity", "map", "track", "accel"], required=True)
    parser.add_argument("--output", type=str, required=False, default=GPS_MAP_FILE, 
                       help=f"Путь для сохранения HTML карты GPS (по умолчанию: {GPS_MAP_FILE})")

    args = parser.parse_args()

    if args.dataset == "gps":
        # GPS обработка - импортируем только необходимые модули
        from src.io.csv_reader import read_GPS
        from src.viz.plots import plot_gps_on_map
        
        if not args.gps:
            print("Ошибка: для GPS требуется указать --gps <путь_к_файлу>")
            return
        
        if args.action == "map":
            gps_df = read_GPS(args.gps)
            if gps_df is not None:
                plot_gps_on_map(gps_df, args.output)
        else:
            print(f"Ошибка: для GPS поддерживается только action='map'")
    
    elif args.dataset == "helimos":
        # Импортируем модули для helimos только если нужны
        from src.datasets.helimos import load_helimos_frame
        from src.viz.clouds import visualize_point_cloud
        
        if not args.bin:
            print("Ошибка: для helimos требуется указать --bin <путь_к_файлу>")
            return
        pc = load_helimos_frame(args.bin)
        visualize_point_cloud(pc)
    
    elif args.dataset == "helipr":
        # Импортируем модули для helipr только если нужны
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
        # Импортируем только если нужно
        from src.datasets.hercules import load_hercules_aeva
        from src.viz.clouds import visualize_point_cloud
        from src.viz.plots import plot_velocity_vs_azimuth

        if not args.bin:
            print("Ошибка: для hercules требуется указать --bin <путь_к_файлу>")
            return

        pc = load_hercules_aeva(args.bin)

        if args.action == "velocity":
            plot_velocity_vs_azimuth(pc)
        else:
            visualize_point_cloud(pc)
    
    elif args.dataset == "radar":
        from src.datasets.radar import load_radar_frame
        from src.viz.clouds import visualize_point_cloud
        from src.viz.plots import plot_velocity_vs_azimuth

        if not args.radar:
            print("Ошибка: для radar требуется указать --radar <путь_к_файлу.bin>")
            return

        pc = load_radar_frame(args.radar)
        if args.action == "velocity":
            plot_velocity_vs_azimuth(pc)
        else:
            visualize_point_cloud(pc)


    elif args.dataset == "ins":
        from src.datasets.ins import load_ins
        from src.viz.plots import plot_ins_track

        if not args.ins:
            print("Ошибка: для INS требуется указать --ins <путь_к_файлу.csv>")
            return

        ins_df = load_ins(args.ins)
        if args.action == "track":
            plot_ins_track(ins_df)
        else:
            print("Ошибка: для INS поддерживается только action='track'")

    elif args.dataset == "imu":
        from src.datasets.imu import load_imu
        from src.viz.plots import plot_imu_accel

        if not args.imu:
            print("Ошибка: для IMU требуется указать --imu <путь_к_файлу.csv>")
            return

        imu_df = load_imu(args.imu)
        if args.action == "accel":
            plot_imu_accel(imu_df)
        else:
            print("Ошибка: для IMU поддерживается только action='accel'")



if __name__ == "__main__":
    main()