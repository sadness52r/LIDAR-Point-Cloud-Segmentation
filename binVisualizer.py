import open3d as o3d
import numpy as np
import struct
import os

def read_kitti_bin_file(filename):
    """Чтение KITTI формата - 16 байт на точку: x, y, z, intensity"""
    points = []
    with open(filename, "rb") as file:
        while True:
            data = file.read(16)
            if len(data) < 16:
                break
            
            x, y, z, intensity = struct.unpack('ffff', data)
            points.append([x, y, z])
    return points

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    points_array = np.array(points)
    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    if len(points_array) > 0:
        z_coords = points_array[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        if z_max > z_min:
            # Нормализуем высоту от 0 до 1
            normalized_z = (z_coords - z_min) / (z_max - z_min)
            # Создаем цветовую карту (синий -> зеленый -> красный)
            colors = np.zeros((len(points_array), 3))
            colors[:, 0] = normalized_z  # Красный
            colors[:, 1] = 1 - normalized_z  # Зеленый  
            colors[:, 2] = 0.5  # Синий
            pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Визуализируем {len(points_array)} точек")
    o3d.visualization.draw_geometries([pcd])


filename = input("Введите путь к файлу: ") # example: "/Users/kirillbukin/Projects/LIDAR-Point-Cloud-Segmentation/000020.bin"
pointcloud = read_kitti_bin_file(filename)
print(f"Прочитано точек: {len(pointcloud)}")
visualize_point_cloud(pointcloud)
