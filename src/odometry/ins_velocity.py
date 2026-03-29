import numpy as np
 
 
def INS_to_V(INS):
    """
    Преобразование INS-скоростей (NED) в ECEF-скорости (Vx, Vy).
 
    Для каждой точки строится матрица поворота R из локальной NED-системы
    в ECEF по текущей широте и долготе, затем Vxyz = R @ [Ve, Vn, Vu].
 
    Parameters:
        INS : dict-like (DataFrame) с ключами:
              'timestamp' (нс), 'latitude', 'longitude',
              'north_velocity', 'east_velocity', 'up_velocity'
 
    Returns:
        Vx : np.ndarray — скорость по оси X (ECEF), м/с
        Vy : np.ndarray — скорость по оси Y (ECEF), м/с
        ts : np.ndarray — временные метки (секунды)
    """
    INS['timestamp'] = INS['timestamp'] / 1e9
 
    Vn = INS['north_velocity']
    Ve = INS['east_velocity']
    Vu = INS['up_velocity']
 
    Vx = []
    Vy = []
    ts = []
 
    for i in range(len(INS)):
        phi = INS['latitude'][i] * np.pi / 180
        lam = INS['longitude'][i] * np.pi / 180
 
        # Матрица поворота NED → ECEF
        R = np.array([
            [-np.sin(lam), -np.sin(phi) * np.cos(lam), np.cos(phi) * np.cos(lam)],
            [ np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi) * np.sin(lam)],
            [           0,  np.cos(phi),                np.sin(phi)],
        ])
 
        Vgps = np.array([[Ve[i]], [Vn[i]], [Vu[i]]])
        Vxyz = R @ Vgps
 
        Vx.append(Vxyz[0, 0])
        Vy.append(Vxyz[1, 0])
        ts.append(INS['timestamp'][i])
 
    return np.array(Vx), np.array(Vy), np.array(ts)
 