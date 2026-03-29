"""
Восстановление полного вектора скорости из радиальных скоростей.
 
По набору радиальных скоростей (Vr) и соответствующих азимутов решает
переопределённую систему методом псевдообратной матрицы для оценки
полного вектора скорости (Vx, Vy).
"""
 
import numpy as np
 
 
def az_Vr_to_full_V(azimuth: np.ndarray, Vr: np.ndarray):
    """
    Восстановление полного вектора скорости из радиальных компонент.
 
    Решает задачу: Vr_i = Vx * sin(az_i) - Vy * cos(az_i)
    через псевдообратную матрицу (метод наименьших квадратов).
 
    Parameters:
        azimuth : np.ndarray, shape (N,) — азимуты точек (радианы)
        Vr      : np.ndarray, shape (N,) — радиальные скорости (м/с)
 
    Returns:
        V     : float — абсолютное значение скорости (м/с)
        angle : float — направление скорости (радианы)
        Vx    : float — компонента скорости по X (м/с)
        Vy    : float — компонента скорости по Y (м/с)
    """
    X = np.array([np.sin(azimuth), -np.cos(azimuth)]).T
    b = np.reshape(Vr, [len(Vr), 1])
    V_est = np.linalg.pinv(X) @ b
 
    Vx = V_est[0, 0]
    Vy = V_est[1, 0]
    angle = np.arctan(Vx / Vy)
    V = np.sqrt(Vx**2 + Vy**2)
 
    return V, angle, Vx, Vy
 