from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from data_generation.configuration import resolve_config

class TransformError(Exception):
    """Base class for transform lookup failures."""

class TransformLookupError(TransformError):
    """A requested frame is unknown to the buffer."""

class TransformConnectivityError(TransformError):
    """Two frames are known but not connected in one tree."""

class TransformExtrapolationError(TransformError):
    """The requested time lies outside the data stored in the buffer."""

@dataclass(frozen=True)
class Pose:
    """
    Rigid pose.

    :ivar position: np.ndarray - (x, y, z)
    :ivar orientation: np.ndarray - quaternion (x, y, z, w)
    """

    position: np.ndarray
    orientation: np.ndarray

@dataclass(frozen=True)
class StampedTransform:
    """
    Transform that maps points from `child` coordinates to `parent` coordinates.

    :ivar stamp: float - time in seconds
    :ivar parent: str - parent frame id
    :ivar child: str - child frame id
    :ivar translation: np.ndarray - (x, y, z)
    :ivar rotation: np.ndarray - quaternion (x, y, z, w)
    """
    stamp: float
    parent: str
    child: str
    translation: np.ndarray
    rotation: np.ndarray

@dataclass(frozen=True)
class CameraInfo:
    """
    Used fields of sensor_msgs/CameraInfo.

    :ivar frame_id: str - optical frame id
    :ivar width: int - image width in pixels
    :ivar height: int - image height in pixels
    :ivar K: np.ndarray - intrinsic matrix
    :ivar D: np.ndarray - distortion coefficients
    :ivar R: np.ndarray - rectification matrix
    :ivar P: np.ndarray - 3x4 projection matrix, row-major
    """
    frame_id: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    P: np.ndarray

def quaternion_to_rotation(quaternion, epsilon=None) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix; a near-zero quaternion gives the identity.

    :param quaternion: np.ndarray - (4,) quaternion (x, y, z, w), normalized internally
    :param epsilon: float | None - squared-norm threshold for the identity, None = `geometry.quaternion_epsilon`
    :return: np.ndarray - (3, 3) rotation matrix
    """
    if epsilon is None:
        epsilon = resolve_config()["geometry"]["quaternion_epsilon"]
    q = np.asarray(quaternion, dtype=np.float64)
    norm_sq = float(q @ q)
    if norm_sq < epsilon:
        return np.eye(3)
    x, y, z, w = q * math.sqrt(1.0 / norm_sq)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])

def pose_to_matrix(pose, epsilon=None) -> np.ndarray:
    """
    Build the 4x4 homogeneous matrix of a pose.

    :param pose: Pose - pose to convert
    :param epsilon: float | None - quaternion near-zero threshold, None = `geometry.quaternion_epsilon`
    :return: np.ndarray - (4, 4) homogeneous matrix
    """
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_rotation(pose.orientation, epsilon)
    mat[:3, 3] = pose.position
    return mat

def yaw_from_matrix(mat, epsilon=None) -> float:
    """
    Return the yaw (rotation about z) of a homogeneous matrix.

    :param mat: np.ndarray - (4, 4) homogeneous matrix
    :param epsilon: float | None - degeneracy threshold, None = `geometry.quaternion_epsilon`
    :return: float - yaw in radians, 0.0 if the x axis projects degenerately onto the xy plane
    """
    if epsilon is None:
        epsilon = resolve_config()["geometry"]["quaternion_epsilon"]
    if math.hypot(mat[0, 0], mat[1, 0]) > epsilon:
        return math.atan2(mat[1, 0], mat[0, 0])
    return 0.0

class _FrameCache:
    """Stored transforms of one child frame (a single entry if static)."""

    __slots__ = ("is_static", "times", "samples")

    def __init__(self, is_static) -> None:
        """
        Create an empty cache.

        :param is_static: bool - child frame holds a single static transform
        """
        self.is_static = is_static
        self.times = []
        self.samples = []  # (parent, translation, rotation)

class TransformBuffer:
    """Time-aware transform tree with tf2-style lookups at the latest common time."""

    def __init__(self, cache_time, norm_tolerance=None, epsilon=None) -> None:
        """
        Create an empty buffer.

        :param cache_time: float - seconds of dynamic history kept per child frame
        :param norm_tolerance: float | None - allowed quaternion squared-norm error, None = config
        :param epsilon: float | None - quaternion near-zero threshold, None = config
        """
        if norm_tolerance is None or epsilon is None:
            geometry = resolve_config()["geometry"]
            norm_tolerance = geometry["quaternion_norm_tolerance"] if norm_tolerance is None else norm_tolerance
            epsilon = geometry["quaternion_epsilon"] if epsilon is None else epsilon
        self.cache_time = float(cache_time)
        self.norm_tolerance = float(norm_tolerance)
        self.epsilon = float(epsilon)
        self.rejected = Counter()
        self._frames = {}
        self._known = set()

    @staticmethod
    def _strip(frame_id) -> str:
        """
        Remove a leading slash from a frame id.

        :param frame_id: str - frame id
        :return: str - stripped frame id
        """
        return frame_id[1:] if frame_id.startswith("/") else frame_id

    def set_transform(self, transform, static=False) -> bool:
        """
        Insert a transform, replacing static entries and time-sorting dynamic ones.

        :param transform: StampedTransform - transform to insert
        :param static: bool - static transforms never expire
        :return: bool - False if rejected; reasons are counted in `self.rejected`
        """
        parent = self._strip(transform.parent)
        child = self._strip(transform.child)
        translation = np.asarray(transform.translation, dtype=np.float64)
        rotation = np.asarray(transform.rotation, dtype=np.float64)

        if not parent or not child:
            self.rejected["empty frame id"] += 1
            return False
        if parent == child:
            self.rejected["transform from a frame to itself"] += 1
            return False
        if not (np.all(np.isfinite(translation)) and np.all(np.isfinite(rotation))):
            self.rejected["non-finite transform"] += 1
            return False
        if abs(float(rotation @ rotation) - 1.0) > self.norm_tolerance:
            self.rejected["quaternion not normalized"] += 1
            return False

        cache = self._frames.get(child)
        if cache is None:
            cache = self._frames[child] = _FrameCache(static)
        sample = (parent, translation, rotation)

        if cache.is_static:
            cache.times = [transform.stamp]
            cache.samples = [sample]
        else:
            if cache.times and cache.times[0] > transform.stamp + self.cache_time:
                self.rejected["older than the cache window"] += 1
                return False
            idx = bisect.bisect_right(cache.times, transform.stamp)
            cache.times.insert(idx, transform.stamp)
            cache.samples.insert(idx, sample)
            drop = bisect.bisect_left(cache.times, cache.times[-1] - self.cache_time)
            if drop:
                del cache.times[:drop]
                del cache.samples[:drop]

        self._known.update((parent, child))
        return True

    def _chain(self, frame) -> tuple[list[str], dict[str, float | None]]:
        """
        Walk from a frame up to the root of its tree; raise TransformLookupError on a loop.

        :param frame: str - start frame id
        :return: tuple[list[str], dict[str, float | None]] - frames from `frame` to the root, and each hop frame's newest stamp (None if static)
        """
        frames = [frame]
        latest = {}
        while True:
            cache = self._frames.get(frames[-1])
            if cache is None:
                return frames, latest
            latest[frames[-1]] = None if cache.is_static else cache.times[-1]
            parent = cache.samples[-1][0]
            if parent in frames:
                raise TransformLookupError(f"transform tree loop at frame '{parent}'")
            frames.append(parent)

    def _hop_matrix(self, frame, time) -> np.ndarray:
        """
        Build the transform of one hop at `time`; raise TransformExtrapolationError outside a dynamic frame's buffered range.

        :param frame: str - child frame id
        :param time: float | None - time in seconds, ignored for static frames
        :return: np.ndarray - (4, 4) matrix mapping `frame` coordinates to its parent
        """
        cache = self._frames[frame]
        if cache.is_static:
            parent, translation, rotation = cache.samples[-1]
        else:
            times = cache.times
            idx = bisect.bisect_left(times, time)
            if idx < len(times) and times[idx] == time:
                parent, translation, rotation = cache.samples[idx]
            elif idx == 0 or idx == len(times):
                raise TransformExtrapolationError(
                    f"time {time:.6f} for frame '{frame}' is outside the buffered range "
                    f"[{times[0]:.6f}, {times[-1]:.6f}]"
                )
            else:
                parent, t0, q0 = cache.samples[idx - 1]
                _, t1, q1 = cache.samples[idx]
                ratio = (time - times[idx - 1]) / (times[idx] - times[idx - 1])
                translation = t0 + (t1 - t0) * ratio
                rotation = Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))([ratio]).as_quat()[0]
        mat = np.eye(4)
        mat[:3, :3] = quaternion_to_rotation(rotation, self.epsilon)
        mat[:3, 3] = translation
        return mat

    def lookup_latest(self, target, source) -> np.ndarray:
        """
        Look up the transform from `source` into `target` coordinates at the latest common time.

        :param target: str - frame the points are expressed in
        :param source: str - frame the points are given in
        :return: np.ndarray - (4, 4) matrix mapping `source` to `target`
        :raises TransformLookupError: a frame is unknown
        :raises TransformConnectivityError: the frames are in different trees
        :raises TransformExtrapolationError: the hops have no common covered time
        """
        target = self._strip(target)
        source = self._strip(source)
        for frame in (target, source):
            if frame not in self._known:
                raise TransformLookupError(f"frame '{frame}' does not exist")
        if target == source:
            return np.eye(4)

        src_frames, src_latest = self._chain(source)
        tgt_frames, tgt_latest = self._chain(target)
        tgt_set = set(tgt_frames)
        src_stop = next((i for i, f in enumerate(src_frames) if f in tgt_set), None)
        if src_stop is None:
            raise TransformConnectivityError(f"'{target}' and '{source}' are not connected")
        tgt_stop = tgt_frames.index(src_frames[src_stop])

        src_hops = src_frames[:src_stop]
        tgt_hops = tgt_frames[:tgt_stop]
        stamps = [src_latest[f] for f in src_hops if src_latest[f] is not None]
        stamps += [tgt_latest[f] for f in tgt_hops if tgt_latest[f] is not None]
        time = min(stamps) if stamps else None

        def to_ancestor(hops) -> np.ndarray:
            """
            Compose hop matrices up to the common ancestor.

            :param hops: list[str] - frames from the start frame, excluding the ancestor
            :return: np.ndarray - (4, 4) matrix mapping start-frame to ancestor coordinates
            """
            acc = np.eye(4)
            for frame in hops:
                acc = self._hop_matrix(frame, time) @ acc
            return acc

        return np.linalg.inv(to_ancestor(tgt_hops)) @ to_ancestor(src_hops)

def validate_camera_distortion(model, coefficients) -> np.ndarray:
    """
    Validate a distortion model and its coefficients; raise ValueError if unsupported or malformed.

    :param model: str - "plumb_bob", "rational_polynomial" or "equidistant"
    :param coefficients: array-like - distortion coefficients
    :return: np.ndarray - 1D float64 coefficients
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    lengths = {"plumb_bob": (4, 5, 8, 12, 14), "rational_polynomial": (8, 12, 14), "equidistant": (4,)}
    if model not in lengths:
        raise ValueError(f"Unsupported camera distortion model: {model!r}")
    if coefficients.size not in lengths[model] or not np.isfinite(coefficients).all():
        raise ValueError(
            f"Invalid {model} distortion coefficients: expected lengths {lengths[model]}, "
            f"got {coefficients.size}"
        )
    return coefficients

def reproject_depth_to_rgb(depth_m, depth_K, depth_D, depth_model, rgb_K, rgb_D,
                           rgb_model, rgb_from_depth, output_shape) -> np.ndarray:
    """
    Reproject a depth image into the RGB camera, keeping the nearest depth per pixel.

    :param depth_m: array-like - (H, W) depth in meters, <= 0 or non-finite = no reading
    :param depth_K: array-like - depth camera intrinsic matrix
    :param depth_D: array-like - depth camera distortion coefficients
    :param depth_model: str - depth camera distortion model name
    :param rgb_K: array-like - RGB camera intrinsic matrix
    :param rgb_D: array-like - RGB camera distortion coefficients
    :param rgb_model: str - RGB camera distortion model name
    :param rgb_from_depth: array-like - (4, 4) transform from depth to RGB coordinates
    :param output_shape: tuple[int, int] - output (H, W)
    :return: np.ndarray - float32 depth of `output_shape`, 0 = no reading
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Depth image must be two-dimensional")
    depth_D = validate_camera_distortion(depth_model, depth_D)
    rgb_D = validate_camera_distortion(rgb_model, rgb_D)
    depth_K = np.asarray(depth_K, dtype=np.float64).reshape(3, 3)
    rgb_K = np.asarray(rgb_K, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(rgb_from_depth, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(depth_K).all() or not np.isfinite(rgb_K).all() or not np.isfinite(transform).all():
        raise ValueError("Camera calibration and transform must be finite")
    height, width = output_shape
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(v):
        return np.zeros((height, width), dtype=np.float32)
    z = depth[v, u]
    pixels = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
    if depth_model == "equidistant":
        normalized = cv2.fisheye.undistortPoints(pixels, depth_K, depth_D.reshape(4, 1)).reshape(-1, 2)
    else:
        normalized = cv2.undistortPoints(pixels, depth_K, depth_D).reshape(-1, 2)
    points = np.column_stack((normalized * z[:, None], z))
    rgb_points = points @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(rgb_points).all(axis=1) & (rgb_points[:, 2] > 0)
    rgb_points = rgb_points[valid]
    if not len(rgb_points):
        return np.zeros((height, width), dtype=np.float32)
    if rgb_model == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            rgb_points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), rgb_K, rgb_D.reshape(4, 1)
        )
    else:
        projected, _ = cv2.projectPoints(rgb_points, np.zeros(3), np.zeros(3), rgb_K, rgb_D)
    projected = projected.reshape(-1, 2)
    finite = np.isfinite(projected).all(axis=1)
    projected, rgb_points = projected[finite], rgb_points[finite]
    return zbuffer_depth(projected[:, 0], projected[:, 1], rgb_points[:, 2], output_shape)

def scale_intrinsics(K, from_shape, to_shape) -> np.ndarray:
    """
    Rescale a camera matrix for an image resized from `from_shape` to `to_shape`.

    :param K: array-like - (3, 3) camera matrix for `from_shape`
    :param from_shape: tuple[int, int] - calibrated (H, W)
    :param to_shape: tuple[int, int] - resized (H, W)
    :return: np.ndarray - (3, 3) camera matrix for `to_shape`
    """
    # cv2.resize pixel-centre convention; distortion coefficients are unaffected (normalized coordinates)
    scaled = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    sx = to_shape[1] / from_shape[1]
    sy = to_shape[0] / from_shape[0]
    scaled[0, :2] *= sx
    scaled[1, :2] *= sy
    scaled[0, 2] = (scaled[0, 2] + 0.5) * sx - 0.5
    scaled[1, 2] = (scaled[1, 2] + 0.5) * sy - 0.5
    scaled[2] = (0.0, 0.0, 1.0)
    return scaled

def zbuffer_depth(u, v, z, output_shape) -> np.ndarray:
    """
    Rasterize projected points, keeping the nearest depth per pixel.

    :param u: np.ndarray - (N,) pixel columns
    :param v: np.ndarray - (N,) pixel rows
    :param z: np.ndarray - (N,) positive depths in meters
    :param output_shape: tuple[int, int] - output (H, W)
    :return: np.ndarray - float32 depth of `output_shape`, 0 = no point
    """
    height, width = output_shape
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    flat = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(flat, py[inside] * width + px[inside], np.asarray(z, dtype=np.float64)[inside])
    flat[np.isinf(flat)] = 0
    return flat.reshape(height, width).astype(np.float32)

def project_points_to_depth(points_cam, K, output_shape) -> np.ndarray:
    """
    Render points in a rectified pinhole camera's optical frame into a depth image.

    :param points_cam: array-like - (N, 3) points in meters, z along the optical axis
    :param K: array-like - (3, 3) rectified camera matrix for `output_shape`
    :param output_shape: tuple[int, int] - output (H, W)
    :return: np.ndarray - float32 depth of `output_shape`, 0 = no point
    """
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    u = K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]
    v = K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]
    return zbuffer_depth(u, v, points[:, 2], output_shape)

class PinholeCameraModel:
    """Rectified pinhole projection through the 3x4 projection matrix P (distortion is not applied)."""

    def __init__(self) -> None:
        """Create a camera model without calibration."""
        self.tf_frame = None
        self.projection = None

    @property
    def ready(self) -> bool:
        """
        Whether a calibration has been set.

        :return: bool - True once `update` has been called
        """
        return self.projection is not None

    def update(self, info) -> None:
        """
        Store the frame id and projection matrix P of a calibration.

        :param info: CameraInfo - calibration
        """
        self.tf_frame = info.frame_id
        self.projection = np.asarray(info.P, dtype=np.float64).reshape(3, 4)

    def project3d_to_pixel(self, point) -> tuple[float, float]:
        """
        Project a point in the camera optical frame to pixels; raise RuntimeError without calibration.

        :param point: np.ndarray - (3,) point (x, y, z)
        :return: tuple[float, float] - pixel (u, v), NaN if the point projects to infinity
        """
        if self.projection is None:
            raise RuntimeError("camera model has no calibration")
        x, y, w = self.projection @ np.array([point[0], point[1], point[2], 1.0])
        if w != 0.0:
            return x / w, y / w
        return float("nan"), float("nan")
