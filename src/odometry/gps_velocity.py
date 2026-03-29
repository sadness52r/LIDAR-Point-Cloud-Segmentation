import numpy as np
import pyproj
 
 
def geodetic_to_cartesian(lat: float, lon: float, alt: float):
    """
    Преобразование геодезических координат в декартовы (ECEF).
 
    Parameters:
        lat : широта (градусы)
        lon : долгота (градусы)
        alt : высота (метры)
 
    Returns:
        (x, y, z) в метрах (ECEF)
    """
    geodetic_crs = pyproj.CRS("EPSG:4326")
    ecef_crs = pyproj.CRS("EPSG:4978")
    transformer = pyproj.Transformer.from_crs(geodetic_crs, ecef_crs)
    x, y, z = transformer.transform(lat, lon, alt)
    return x, y, z
 
 
def GPS_to_V(GPS):
    """
    Вычисление скорости по GPS-координатам методом конечных разностей.
 
    Преобразует пары последовательных GPS-точек в ECEF и вычисляет
    Vx = dx/dt, Vy = dy/dt. Пропускает точки с неизменной широтой
    (дубликаты / статические данные).
 
    Parameters:
        GPS : dict-like (DataFrame) с ключами:
              'timestamp' (нс), 'lat', 'lon', 'height'
 
    Returns:
        Vx : np.ndarray — скорость по оси X (ECEF), м/с
        Vy : np.ndarray — скорость по оси Y (ECEF), м/с
        ts : np.ndarray — временные метки (секунды)
    """
    Vx = []
    Vy = []
    ts = []
 
    geodetic_crs = pyproj.CRS("EPSG:4326")
    ecef_crs = pyproj.CRS("EPSG:4978")
    transformer = pyproj.Transformer.from_crs(geodetic_crs, ecef_crs)
 
    GPS['timestamp'] = GPS['timestamp'] / 1e9
 
    for i in range(1, len(GPS)):
        if GPS['lat'][i - 1] != GPS['lat'][i]:
            x1, y1, z1 = transformer.transform(
                GPS['lat'][i - 1], GPS['lon'][i - 1], GPS['height'][i - 1]
            )
            x2, y2, z2 = transformer.transform(
                GPS['lat'][i], GPS['lon'][i], GPS['height'][i]
            )
            dt = GPS['timestamp'][i] - GPS['timestamp'][i - 1]
            Vx.append((x2 - x1) / dt)
            Vy.append((y2 - y1) / dt)
            ts.append(GPS['timestamp'][i])
 
    return np.array(Vx), np.array(Vy), np.array(ts)
 