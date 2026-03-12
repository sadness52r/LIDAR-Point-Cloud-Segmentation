"""
Motion Object Segmentation (MOS) for LiDAR point clouds.

Two backends:
  1. velocity-RANSAC  – ego-motion estimation from per-point Doppler velocity
                        (Aeva / HeliPR sensors). No model required.
  2. Random-Forest    – classifier trained on HeLiMOS labelled data.
                        Works on any sensor; use when Doppler is unavailable.

Typical workflow
----------------
# --- Training (once) ---
seg = MotionSegmenter()
seg.train_on_helimos("data/Deskewed_LiDAR", sensor="Velodyne", max_frames=500)
seg.save("models/mos_rf.pkl")

# --- Inference ---
seg = MotionSegmenter()
seg.load("models/mos_rf.pkl")

frames = [load_helimos_frame(p) for p in frame_paths]
is_moving_list = seg.segment_frames(frames)          # per-frame, no poses
# or
is_moving_list = seg.segment_sequence(frames, poses) # temporal consistency

# --- Cluster moving objects ---
cluster_ids = cluster_moving_objects(frames[0], is_moving_list[0])
"""

import os
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.core.pointcloud import PointCloud
from src.io.bin_reader import read_kitti_bin
from src.io.label_reader import read_label

# ──────────────────────────────────────────────────────────────────────────────
# HeLiMOS label constants (binary MOS variant)
# ──────────────────────────────────────────────────────────────────────────────
STATIC_LABEL = 9    # static environment
MOVING_LABEL = 251  # moving object


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(pc: PointCloud) -> np.ndarray:
    """
    Extract per-point feature matrix from a PointCloud.

    Feature columns
    ---------------
    0: x   1: y   2: z
    3: range (3-D distance from sensor)
    4: azimuth   [rad]
    5: elevation [rad]
    6: intensity  (0 if unavailable)
    7: radial velocity (only present when pc.velocity is not None)

    Returns
    -------
    ndarray of shape (N, 7) or (N, 8)
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


# ──────────────────────────────────────────────────────────────────────────────
# RANSAC ego-motion (for Doppler-capable sensors)
# ──────────────────────────────────────────────────────────────────────────────

def ransac_ego_motion(
    pc: PointCloud,
    n_iterations: int = 300,
    inlier_threshold: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate ego-velocity from per-point Doppler data with RANSAC.

    Model
    -----
    For a stationary object seen at azimuth α:

        v_r(α) = −V_x · cos(α) − V_y · sin(α)

    This is linear in the unknown ego-velocity [V_x, V_y].
    Moving objects violate this model and produce large residuals.

    Parameters
    ----------
    pc               : PointCloud with non-None velocity
    n_iterations     : RANSAC iterations
    inlier_threshold : residual threshold [m/s] to label a point as static
    seed             : RNG seed for reproducibility

    Returns
    -------
    ego_params : np.ndarray, shape (2,) — estimated [V_x, V_y]
    is_static  : bool ndarray, shape (N,) — True for static points
    """
    assert pc.velocity is not None, "RANSAC requires per-point Doppler velocity."

    x, y = pc.xyz[:, 0], pc.xyz[:, 1]
    alpha = np.arctan2(y, x)
    v_r = pc.velocity.astype(np.float64)

    # Design matrix:  v_r = [-cos α, -sin α] @ [Vx, Vy]
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
    else:  # терминальное условие, если RANSAC не нашел лучшей модели (все точки объявляются статическими и собственная скорость авто = 0 - рискованно)
        params = np.zeros(2)
        is_static = np.ones(n_pts, dtype=bool)

    return params.astype(np.float32), is_static


# ──────────────────────────────────────────────────────────────────────────────
# Temporal consistency (pose-based)
# ──────────────────────────────────────────────────────────────────────────────

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

    Algorithm
    ---------
    For each frame i, transform all points from the ±n_context window into
    frame i's coordinate system.  Build a voxel occupancy map and record
    which frames contributed points to each voxel.

    A voxel occupied by many frames → static environment.
    A voxel occupied by only 1–2 frames → moving object (it was elsewhere
    in the other frames).

    Parameters
    ----------
    frames           : N consecutive PointCloud objects
    poses            : (N, 12) array of 3×4 SE(3) poses (rows from poses.txt)
    n_context        : half-width of the temporal window
    voxel_size       : spatial resolution [m]
    moving_threshold : temporal-occupancy fraction below which a point is
                       considered moving

    Returns
    -------
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

        # Map from voxel key → set of frame indices that put a point there
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


# ──────────────────────────────────────────────────────────────────────────────
# DBSCAN clustering of moving objects
# ──────────────────────────────────────────────────────────────────────────────

def cluster_moving_objects(
    pc: PointCloud,
    is_moving: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 5,
    velocity_weight: float = 2.0,
) -> np.ndarray:
    """
    Cluster moving points into individual objects with DBSCAN.

    Parameters
    ----------
    pc              : source PointCloud
    is_moving       : bool mask (True = moving)
    eps             : DBSCAN neighbourhood radius [m]
    min_samples     : minimum cluster size
    velocity_weight : weight applied to the velocity channel when building
                      the feature space (higher = velocity matters more)

    Returns
    -------
    cluster_ids : int32 array, shape (N,).
                  -2  → static point
                  -1  → moving noise (no cluster)
                  ≥0  → object cluster index
    """
    cluster_ids = np.full(len(is_moving), -2, dtype=np.int32)
    moving_idx = np.where(is_moving)[0]

    if len(moving_idx) < min_samples:
        return cluster_ids

    pts = pc.xyz[moving_idx, :2].astype(np.float32)     # x, y

    if pc.velocity is not None:
        vel = pc.velocity[moving_idx].reshape(-1, 1).astype(np.float32)
        pts = np.hstack([pts, vel * velocity_weight])

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(pts)
    cluster_ids[moving_idx] = labels.astype(np.int32)

    return cluster_ids


# ──────────────────────────────────────────────────────────────────────────────
# MotionSegmenter – main class
# ──────────────────────────────────────────────────────────────────────────────

class MotionSegmenter:
    """
    Moving Object Segmentation for LiDAR point clouds.

    Two inference backends (auto-selected):
    - Doppler RANSAC  if all input frames have per-point velocity.
    - Random Forest   otherwise (requires prior call to train_on_helimos /
                      load).
    """

    def __init__(self, threshold: float = 0.85, inlier_threshold: float = 0.5,
                 use_gpu: bool = False) -> None:
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold = threshold  # P(moving) выше этого → moving
        self.inlier_threshold = inlier_threshold  # RANSAC residual threshold [m/s]
        self.use_gpu = use_gpu

    # ── Training ─────────────────────────────────────────────────────────────

    def train_on_helimos(
        self,
        data_root: str,
        sensor: str = "Velodyne",
        split: str = "train",
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Train a Random Forest classifier on HeLiMOS labelled data.

        Parameters
        ----------
        data_root  : path to Deskewed_LiDAR root (contains train.txt etc.)
        sensor     : 'Velodyne', 'Ouster', 'Avia', or 'Aeva'
        split      : 'train', 'val', or 'test'
        max_frames : limit frames used (None = use all)
        """
        split_file = os.path.join(data_root, f"{split}.txt")
        with open(split_file) as f:
            frame_ids = [int(ln.strip()) for ln in f if ln.strip()]

        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]

        sensor_dir = os.path.join(data_root, sensor)
        vel_dir = os.path.join(sensor_dir, "velodyne")
        lbl_dir = os.path.join(sensor_dir, "labels")

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

            pc = read_kitti_bin(bin_path)
            semantic, _ = read_label(lbl_path)

            mask = (semantic == STATIC_LABEL) | (semantic == MOVING_LABEL)
            if mask.sum() == 0:
                skipped += 1
                continue

            all_feats.append(_extract_features(pc)[mask])
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
        max_train_points = 2_000_000
        if len(X) > max_train_points:
            moving_idx = np.where(y == 1)[0]
            static_idx = np.where(y == 0)[0]

            # Берём все moving точки (их мало)
            n_moving = len(moving_idx)
            n_static_budget = max_train_points - n_moving
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
                f"[MOS] Subsampled → {len(X):,} points "
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

    # ── GPU (XGBoost) ────────────────────────────────────────────────────────

    @staticmethod
    def _make_xgb_classifier(X_sc: np.ndarray, y: np.ndarray):
        """Create and train an XGBClassifier on GPU (CUDA)."""
        import xgboost as xgb

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        scale = n_neg / max(n_pos, 1)

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

    # ── Inference ─────────────────────────────────────────────────────────────

    def segment_frames(
        self,
        frames: List[PointCloud],
    ) -> List[np.ndarray]:
        """
        Classify each point in each frame as moving (True) or static (False).

        Backend selection
        -----------------
        - All frames have .velocity  →  RANSAC ego-motion (no model needed)
        - Otherwise                  →  Random Forest (model must be loaded)

        Parameters
        ----------
        frames : one or more PointCloud objects

        Returns
        -------
        List of bool arrays, one per frame, True = moving.
        """
        if not frames:
            return []

        use_velocity = all(f.velocity is not None for f in frames)

        if not use_velocity and self.classifier is None:
            raise RuntimeError(
                "No model loaded and no Doppler velocity available.\n"
                "Run train_on_helimos() or load() first."
            )

        results: List[np.ndarray] = []
        for pc in frames:
            if use_velocity:
                _, is_static = ransac_ego_motion(pc, inlier_threshold=self.inlier_threshold)
                results.append(~is_static)
            else:
                feats = _extract_features(pc)
                feats_sc = self.scaler.transform(feats)
                proba = self.classifier.predict_proba(feats_sc)[:, 1]
                results.append(proba > self.threshold)

        return results

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
        a point is moving only if **both** temporal occupancy is low
        **and** the per-frame classifier (or RANSAC) says moving.

        Parameters
        ----------
        frames           : N consecutive PointCloud objects
        poses            : (N, 12) array of 3×4 SE(3) poses (from poses.txt)
        n_context        : temporal window half-width (frames)
        voxel_size       : voxel grid resolution [m]
        moving_threshold : occupancy fraction below which a point is flagged

        Returns
        -------
        List of bool arrays (True = moving), one per frame.
        """
        temporal = temporal_consistency_segment(
            frames, poses, n_context, voxel_size, moving_threshold
        )
        per_frame = self.segment_frames(frames)

        # Intersection: moving if temporal AND per-frame classifier agree
        return [t & p for t, p in zip(temporal, per_frame)]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist trained model to disk."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers (HeLiMOS)
# ──────────────────────────────────────────────────────────────────────────────

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
      9   → static
      251 → moving
      0   → unlabeled
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
