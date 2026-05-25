import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.core.pointcloud import PointCloud
from src.io.bin_reader import read_kitti_bin
from src.io.label_reader import read_label

# HeLiMOS label constants (binary MOS variant)
STATIC_LABEL = 9    # static environment
MOVING_LABEL = 251  # moving object


def compute_segmentation_metrics(is_moving_pred: np.ndarray,
                                 semantic_labels: np.ndarray) -> dict:
    """
    Метрики бинарной point-wise сегментации static vs moving на HeLiMOS labels.

    Положительный класс = moving. Точки с labels, отличными от STATIC_LABEL
    и MOVING_LABEL (например, неразмеченные / outside-FOV), исключаются.

    Returns: dict с n, tp/fp/fn/tn, accuracy, precision, recall, f1, iou.
    """
    valid = (semantic_labels == STATIC_LABEL) | (semantic_labels == MOVING_LABEL)
    if not valid.any():
        return {}
    y_pred = is_moving_pred[valid].astype(bool)
    y_true = (semantic_labels[valid] == MOVING_LABEL)

    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    n = tp + fp + fn + tn

    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return dict(n=n, n_moving_gt=int(y_true.sum()), n_static_gt=int((~y_true).sum()),
                tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=acc, precision=prec, recall=rec, f1=f1, iou=iou)


def print_segmentation_metrics(m: dict, title: str = "") -> None:
    """Печать метрик сегментации в человекочитаемом виде."""
    if not m:
        print("[MOS-eval] нет валидных labels (static=9 или moving=251)")
        return
    print()
    print("=" * 70)
    header = "Метрики сегментации (static vs moving)"
    if title:
        header += f" — {title}"
    print(header)
    print("=" * 70)
    print(f"  N валидных точек: {m['n']}  "
          f"(ground truth: moving={m['n_moving_gt']}, static={m['n_static_gt']})")
    print(f"  TP={m['tp']:>8}  FP={m['fp']:>8}  FN={m['fn']:>8}  TN={m['tn']:>8}")
    print(f"  Accuracy  = {m['accuracy']:.4f}")
    print(f"  Precision = {m['precision']:.4f}   (класс moving)")
    print(f"  Recall    = {m['recall']:.4f}   (класс moving)")
    print(f"  F1        = {m['f1']:.4f}   (класс moving)")
    print(f"  IoU       = {m['iou']:.4f}   (класс moving)")
    print("=" * 70)
    print()


def _find_helimos_label(bin_path: str) -> "str | None":
    """
    Из пути к .bin кадру HeLiMOS получить путь к соответствующему .label.
    Возвращает None если файл отсутствует.
    Структура HeLiMOS: <sensor>/velodyne/000123.bin → <sensor>/labels/000123.label
    """
    import os as _os
    parts = bin_path.replace("\\", "/").split("/")
    try:
        i = len(parts) - 1 - parts[::-1].index("velodyne")
    except ValueError:
        return None
    parts[i] = "labels"
    label_path = _os.path.join(*parts)
    stem, _ = _os.path.splitext(label_path)
    label_path = stem + ".label"
    return label_path if _os.path.exists(label_path) else None

# Feature extraction
def _extract_features(pc: PointCloud) -> np.ndarray:
    """
    Извлечь матрицу признаков для каждой точки из облака точек.

    Feature columns
    ---------------
    0: x   1: y   2: z
    3: range (3D расстояние от сенсора)
    4: azimuth   [rad]
    5: elevation [rad]
    6: intensity  (0 если нет)
    7: radial velocity (только если pc.velocity не None)

    Returns
    -------
    ndarray (N, 7) или (N, 8)
    """
    x, y, z = pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]
    xy = np.sqrt(x ** 2 + y ** 2)
    rng = np.sqrt(xy ** 2 + z ** 2) # расстояние от сенсора до точки
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, xy + 1e-9) # вертикальный угол

    intensity = pc.intensity if pc.intensity is not None else np.zeros_like(x)
    feats = [x, y, z, rng, azimuth, elevation, intensity]

    if pc.velocity is not None: # Допплеровская скорость
        feats.append(pc.velocity)

    return np.stack(feats, axis=1).astype(np.float32)

# RANSAC ego-motion (for Doppler-capable sensors)
def ransac_ego_motion(
    pc: PointCloud,
    n_iterations: int = 300,
    inlier_threshold: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Оценивает ego-velocity с помощью радиальных скоростей объектов и RANSAC.

    Parameters:
        pc               : PointCloud с не-None радиальными скоростями
        n_iterations     : кол-во RANSAC итераций
        inlier_threshold : порог для отнесения точки к стационарной
        seed             : RNG seed для воспроизведения

    Returns:
        params     : np.ndarray, shape (2,) — оцененные [V_x, V_y]
        is_static  : bool ndarray, shape (N,) — True для стационарных точек
    """
    assert pc.velocity is not None, "RANSAC requires per-point Doppler velocity."

    x, y = pc.xyz[:, 0], pc.xyz[:, 1]
    alpha = np.arctan2(y, x)
    v_r = pc.velocity.astype(np.float64)

    A = np.column_stack([-np.cos(alpha), -np.sin(alpha)])

    rng = np.random.default_rng(seed)
    n_pts = len(v_r)
    best_inliers: Optional[np.ndarray] = None
    best_count = 0

    for _ in range(n_iterations):
        idx = rng.choice(n_pts, 2, replace=False)
        try:
            params, *_ = np.linalg.lstsq(A[idx], v_r[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(A @ params - v_r)
        inliers = residuals < inlier_threshold
        n_in = int(inliers.sum())
        if n_in > best_count:
            best_count = n_in
            best_inliers = inliers

    if best_inliers is not None and best_inliers.sum() >= 2:
        params, *_ = np.linalg.lstsq(A[best_inliers], v_r[best_inliers], rcond=None)
        residuals = np.abs(A @ params - v_r)
        is_static = residuals < inlier_threshold
    else:
        params = np.zeros(2)
        is_static = np.ones(n_pts, dtype=bool)

    return params.astype(np.float32), is_static

# Temporal consistency (pose-based)
def _pose_to_se3(row: np.ndarray) -> np.ndarray:
    """Convert a 12-element 3×4 pose row to a 4×4 SE(3) matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = row.reshape(3, 4)
    return T

def temporal_consistency_segment(
    frames: List[PointCloud],
    poses: np.ndarray,
    n_context: int = 3,
    voxel_size: float = 0.5,
    moving_threshold: float = 0.35,
) -> List[np.ndarray]:
    """
    Classify points using temporal voxel-occupancy consistency.

    Algorithm:
    For each frame i, transform all points from the ±n_context window into
    frame i's coordinate system. Build a voxel occupancy map and record
    which frames contributed points to each voxel.

    A voxel occupied by many frames -> static environment.
    A voxel occupied by only 1–2 frames -> moving object (it was elsewhere
    in the other frames).

    Parameters:
        frames           : N consecutive PointCloud objects
        poses            : (N, 12) array of 3×4 SE(3) poses (rows from poses.txt)
        n_context        : half-width of the temporal window
        voxel_size       : spatial resolution [m]
        moving_threshold : temporal-occupancy fraction below which a point is considered moving

    Returns:
        List of bool arrays (True = moving), one per input frame.
    """
    n = len(frames)
    assert len(poses) >= n, "Need one pose per frame."

    Ts = [_pose_to_se3(poses[i]) for i in range(n)]
    results: List[np.ndarray] = []

    for i in range(n):
        T_inv_i = np.linalg.inv(Ts[i])
        lo = max(0, i - n_context)
        hi = min(n, i + n_context + 1)
        n_window = hi - lo

        # Map from voxel key -> set of frame indices that put a point there
        voxel_frames: dict = defaultdict(set)

        for j in range(lo, hi):
            T_rel = T_inv_i @ Ts[j]
            pts = frames[j].xyz.astype(np.float64)
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # добавляем столбец единиц для будущего нормального перемножения матриц
            pts_t = (T_rel @ pts_h.T)[:3].T  # переводим точку кадра j в систему координат кадра i

            # Encode voxel index as a single int64 for fast hashing
            vi = np.floor(pts_t / voxel_size).astype(np.int64)
            # Cantor-style linear key (assumes ±10 000 voxels per axis)
            STRIDE = 20001
            keys = vi[:, 0] * STRIDE * STRIDE + vi[:, 1] * STRIDE + vi[:, 2] # кодируем координаты вокселя в одно число
            for k in keys:
                voxel_frames[int(k)].add(j) # для каждого вокселя храним все различные номера кадров, точки которых попадают в этот воксель

        # Score current-frame points
        pts_i = frames[i].xyz.astype(np.float64)
        vi_i = np.floor(pts_i / voxel_size).astype(np.int64)
        STRIDE = 20001
        keys_i = vi_i[:, 0] * STRIDE * STRIDE + vi_i[:, 1] * STRIDE + vi_i[:, 2]

        occupancy = np.array(
            [len(voxel_frames.get(int(k), set())) / n_window for k in keys_i],  # для вокселя точек кадра i смотрим сколько точек кадров j туда попало
            dtype=np.float32,
        )
        results.append(occupancy < moving_threshold) # чем больше попадает - тем выше шанс, что объект стационарный

    return results

# DBSCAN clustering of moving objects
def cluster_moving_objects(
    pc: PointCloud,
    is_moving: np.ndarray,
    eps_xyz: float = 1.0,
    eps_vr: float = 1.0,
    min_samples: int = 4,
    two_stage: bool = True,
) -> np.ndarray:
    """
    DBSCAN-кластеризация движущихся точек.

    Returns:
        cluster_ids : int32 ndarray (N,)
            -2 -> static, -1 -> noise среди moving, >=0 -> id кластера
    """
    cluster_ids = np.full(len(is_moving), -2, dtype=np.int32)
    moving_idx = np.where(is_moving)[0]

    if len(moving_idx) < min_samples:
        return cluster_ids

    # Сначала помечаем все moving точки как noise (-1)
    cluster_ids[moving_idx] = -1

    xyz_mov = pc.xyz[moving_idx].astype(np.float64)
    has_vel = pc.velocity is not None
    vel_mov = pc.velocity[moving_idx].astype(np.float64) if has_vel else None

    if two_stage and has_vel:
        labels = _two_stage_dbscan(
            xyz_mov, vel_mov, eps_xyz, min_samples, eps_vr,
        )
    else:
        if has_vel:
            vr_scale = eps_xyz / max(eps_vr, 1e-6)
            features = np.column_stack([xyz_mov, vel_mov * vr_scale])
        else:
            features = xyz_mov
        db = DBSCAN(eps=eps_xyz, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(features)

    cluster_ids[moving_idx] = labels.astype(np.int32)
    return cluster_ids


def _two_stage_dbscan(
    xyz: np.ndarray,
    vel: np.ndarray,
    eps_xyz: float,
    min_samples: int,
    eps_vr: float,
) -> np.ndarray:
    """
    Двухэтапный DBSCAN в физических единицах.

    Этап 1: кластеризация по Vr [m/s] с eps=eps_vr.
    Этап 2: внутри каждой группы Vr — пространственная кластеризация по xyz [m]
            с eps=eps_xyz.
    """
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)

    # Этап 1: кластеризация по Vr (в м/с)
    db_vel = DBSCAN(eps=eps_vr, min_samples=min_samples, n_jobs=-1)
    vel_labels = db_vel.fit_predict(vel.reshape(-1, 1))

    next_cluster = 0
    for vl in sorted(set(vel_labels)):
        if vl == -1:
            continue
        idx = np.where(vel_labels == vl)[0]
        if len(idx) < min_samples:
            continue

        # Этап 2: пространственная кластеризация (в метрах)
        db_xyz = DBSCAN(eps=eps_xyz, min_samples=min_samples, n_jobs=-1)
        sub_labels = db_xyz.fit_predict(xyz[idx])

        for sl in sorted(set(sub_labels)):
            if sl == -1:
                continue
            sub_mask = sub_labels == sl
            labels[idx[sub_mask]] = next_cluster
            next_cluster += 1

    return labels


# Oriented bounding box for detected clusters
@dataclass
class OBB:
    """Ориентированный 3D bounding box для одного кластера."""
    center: np.ndarray # центр в системе координат сенсора
    extent: np.ndarray # длины сторон вдоль главных осей
    R: np.ndarray      # поворот: главные оси -> мир
    cluster_id: int
    n_points: int

    def corners(self) -> np.ndarray:
        """8 вершин бокса в мировых координатах, форма (8, 3)."""
        ax, ay, az = self.extent / 2.0
        local = np.array([
            [-ax, -ay, -az], [ ax, -ay, -az], [ ax,  ay, -az], [-ax,  ay, -az],
            [-ax, -ay,  az], [ ax, -ay,  az], [ ax,  ay,  az], [-ax,  ay,  az],
        ], dtype=np.float64)
        return (self.R @ local.T).T + self.center


def compute_cluster_obbs(pc: PointCloud, cluster_ids: np.ndarray) -> List[OBB]:
    """
    Для каждого DBSCAN-кластера (id >= 0) построить ориентированный bounding box
    через PCA Open3D. Кластеры, для которых OBB не строится (вырожденная геометрия),
    пропускаются.
    """
    import open3d as o3d

    MIN_CLUSTER_POINTS = 15

    obbs: List[OBB] = []
    for cid in np.unique(cluster_ids[cluster_ids >= 0]):
        if int((cluster_ids == cid).sum()) < MIN_CLUSTER_POINTS:
            continue
        mask = cluster_ids == cid
        pts = pc.xyz[mask].astype(np.float64)

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            obb = pcd.get_oriented_bounding_box()
        except RuntimeError:
            # Open3D кидает RuntimeError если точки коллинеарны/копланарны
            continue

        obbs.append(OBB(
            center=np.asarray(obb.center, dtype=np.float64),
            extent=np.asarray(obb.extent, dtype=np.float64),
            R=np.asarray(obb.R, dtype=np.float64),
            cluster_id=int(cid),
            n_points=int(mask.sum()),
        ))
    return obbs


# MotionSegmenter – main class
class MotionSegmenter:
    """
    Moving Object Segmentation для облака точек лидара.

    - RANSAC, если все кадры в данных имеют радиальную скорость.
    - Random Forest, иначе (требуется предварительный вызов train_on_helimos / load).
    - XGBoost, если надо GPU
    """

    def __init__(self, threshold: float = 0.85, inlier_threshold: float = 0.5,
                 use_gpu: bool = False) -> None: # значения хардкодом вынести в константы конфига
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold = threshold  # P(moving) выше этого -> moving
        self.inlier_threshold = inlier_threshold  # порог для RANSAC [m/s]
        self.use_gpu = use_gpu

    # Training
    def train_on_helimos(
        self,
        data_root: str,
        sensor: str = "Velodyne",
        split: str = "train",
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Обучение Random Forest (или XGBoost для GPU) классификатора на HeLiMOS размеченных данных.

        Parameters:
            data_root  : путь к Deskewed_LiDAR корню (содержит train.txt и др.)
            sensor     : 'Velodyne', 'Ouster', 'Avia', or 'Aeva'
            split      : 'train', 'val', or 'test'
            max_frames : ограничение на используемые кадры (None = использовать все)
        """
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            frame_ids = [int(ln.strip()) for ln in f if ln.strip()]

        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]

        sensor_dir = os.path.join(data_root, sensor)
        vel_dir = os.path.join(sensor_dir, "velodyne") # путь к bin файлам
        lbl_dir = os.path.join(sensor_dir, "labels") # путь к меткам

        all_feats: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        skipped = 0

        print(f"[MOS] Loading {len(frame_ids)} frames  sensor={sensor}  split={split}")
        for i, fid in enumerate(frame_ids):
            bin_path = os.path.join(vel_dir, f"{fid:06d}.bin")
            lbl_path = os.path.join(lbl_dir, f"{fid:06d}.label")

            if not os.path.exists(bin_path) or not os.path.exists(lbl_path):
                skipped += 1
                continue

            pc = read_kitti_bin(bin_path) # получаем облако точек из bin файла
            semantic, _ = read_label(lbl_path) # получаем метки этих точек

            mask = (semantic == STATIC_LABEL) | (semantic == MOVING_LABEL) # убираем кадры, где все точки - мусор
            if mask.sum() == 0:
                skipped += 1
                continue

            all_feats.append(_extract_features(pc)[mask]) # берем фичи только не мусора
            all_labels.append((semantic[mask] == MOVING_LABEL).astype(np.int8)) 

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(frame_ids)} frames …")

        if not all_feats:
            raise RuntimeError(
                "No labelled frames found. Check data_root / sensor path."
            )

        X = np.vstack(all_feats)
        y = np.concatenate(all_labels)
        print(
            f"[MOS] Dataset: {len(X):,} points | "
            f"{y.mean() * 100:.2f}% moving | {skipped} frames skipped"
        )

        # Субсэмплинг: берём не более max_train_points с сохранением
        # баланса классов (все moving + случайная выборка static)
        max_train_points = 2_000_000 # возможно, стоит вынести в константу конфига
        if len(X) > max_train_points:
            moving_idx = np.where(y == 1)[0]
            static_idx = np.where(y == 0)[0]

            # Берём все moving точки (их мало)
            n_moving = len(moving_idx)
            n_static_budget = max_train_points - n_moving # сколько стационарных точек будем брать
            if n_static_budget < n_moving:
                n_static_budget = n_moving  # как минимум 1:1

            rng = np.random.default_rng(42)
            static_sample = rng.choice(
                static_idx, size=min(n_static_budget, len(static_idx)), replace=False
            )
            keep = np.concatenate([moving_idx, static_sample])
            rng.shuffle(keep)
            X = X[keep]
            y = y[keep]
            print(
                f"[MOS] Subsampled -> {len(X):,} points "
                f"({(y == 1).sum():,} moving + {(y == 0).sum():,} static)"
            )

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X)

        if self.use_gpu:
            self.classifier = self._make_xgb_classifier(X_sc, y)
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )
            print("[MOS] Training Random Forest (CPU) …")
            self.classifier.fit(X_sc, y)

        feat_names = ["x", "y", "z", "range", "azimuth", "elevation", "intensity"]
        if X.shape[1] > 7:
            feat_names.append("velocity")
        print("[MOS] Feature importances:")
        importances = self.classifier.feature_importances_
        for name, imp in zip(feat_names, importances):
            bar = "█" * int(imp * 50)
            print(f"  {name:10s} {imp:.3f}  {bar}")

    # GPU (XGBoost)
    @staticmethod
    def _make_xgb_classifier(X_sc: np.ndarray, y: np.ndarray):
        """Обучение с помощью XGBClassifier на GPU (CUDA)."""
        import xgboost as xgb

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        scale = n_neg / max(n_pos, 1)  # для баланса классов, усиливает вклад ошибок по редкому классу 

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            device="cuda",
            random_state=42,
        )
        print("[MOS] Training XGBoost (GPU/CUDA) …")
        clf.fit(X_sc, y)
        return clf

    # Inference
    def segment_frame(
        self,
        pc: PointCloud,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Классифицирует точки одного кадра как moving (True) или static (False).
        - pc имеет .velocity  ->  RANSAC ego-motion
        - иначе               ->  Random Forest (должен быть загружен) или XGBoost (GPU)

        Returns:
            is_moving  : bool ndarray, True = moving
            ego_params : (V_x, V_y) от RANSAC, либо None если использовался классификатор
        """
        if pc.velocity is not None:
            ego_params, is_static = ransac_ego_motion(
                pc, inlier_threshold=self.inlier_threshold
            )
            return ~is_static, ego_params

        if self.classifier is None:
            raise RuntimeError(
                "No model loaded and no Doppler velocity available.\n"
                "Run train_on_helimos() or load() first."
            )
        feats = _extract_features(pc)
        feats_sc = self.scaler.transform(feats)
        proba = self.classifier.predict_proba(feats_sc)[:, 1]
        return proba > self.threshold, None

    def segment_frames(
        self,
        frames: List[PointCloud],
    ) -> List[np.ndarray]:
        """
        Батч-вариант segment_frame. Возвращает только маски moving;
        ego-motion отбрасывается. Если нужны и `ego_params` - вызывайте
        `segment_frame` напрямую.
        """
        return [self.segment_frame(pc)[0] for pc in frames]

    def segment_sequence(
        self,
        frames: List[PointCloud],
        poses: np.ndarray,
        n_context: int = 3,
        voxel_size: float = 0.5,
        moving_threshold: float = 0.35,
    ) -> List[np.ndarray]:
        """
        Classify a sequence of frames using temporal voxel-occupancy.

        Combines temporal consistency with the per-frame classifier:
        a point is moving only if both temporal occupancy is low
        and the per-frame classifier (or RANSAC) says moving.

        Parameters:
            frames           : N consecutive PointCloud objects
            poses            : (N, 12) array of 3×4 SE(3) poses (from poses.txt)
            n_context        : temporal window half-width (frames)
            voxel_size       : voxel grid resolution [m]
            moving_threshold : occupancy fraction below which a point is flagged

        Returns:
            List of bool arrays (True = moving), one per frame.
        """
        temporal = temporal_consistency_segment(
            frames, poses, n_context, voxel_size, moving_threshold
        )
        per_frame = self.segment_frames(frames)

        # Intersection: moving if temporal AND per-frame classifier agree
        return [t & p for t, p in zip(temporal, per_frame)]

    # Persistence
    def save(self, path: str) -> None:
        """Сохранить обученную модель на диск."""
        if self.classifier is None:
            raise RuntimeError("Train the model before saving.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "scaler": self.scaler,
            "threshold": self.threshold,
        }, path)
        print(f"[MOS] Model saved → {path}  (threshold={self.threshold})")

    def load(self, path: str) -> None:
        """Load a previously saved model from disk."""
        data = joblib.load(path)
        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.threshold = data.get("threshold", 0.85)
        print(f"[MOS] Model loaded ← {path}  (threshold={self.threshold})")

# Dataset helpers (HeLiMOS)
def iter_helimos_labeled(
    data_root: str,
    sensor: str = "Velodyne",
    split: str = "train",
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, PointCloud, np.ndarray]]:
    """
    Iterate over labelled HeLiMOS frames.

    Yields
    ------
    (frame_id, PointCloud, semantic_labels)

    semantic_labels is a uint16 array where:
      9   -> static
      251 -> moving
      0   -> unlabeled
    """
    split_file = os.path.join(data_root, f"{split}.txt")
    with open(split_file) as f:
        frame_ids = [int(ln.strip()) for ln in f if ln.strip()]

    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    sensor_dir = os.path.join(data_root, sensor)
    vel_dir = os.path.join(sensor_dir, "velodyne")
    lbl_dir = os.path.join(sensor_dir, "labels")

    for fid in frame_ids:
        bin_path = os.path.join(vel_dir, f"{fid:06d}.bin")
        lbl_path = os.path.join(lbl_dir, f"{fid:06d}.label")

        if not os.path.exists(bin_path) or not os.path.exists(lbl_path):
            continue

        pc = read_kitti_bin(bin_path)
        semantic, _ = read_label(lbl_path)
        yield fid, pc, semantic
