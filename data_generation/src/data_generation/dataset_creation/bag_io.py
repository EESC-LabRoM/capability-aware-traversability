from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Connection

from data_generation.configuration import resolve_config
from data_generation.dataset_creation.geometry import CameraInfo, Pose, StampedTransform

@dataclass(frozen=True)
class Odometry:
    """Used fields of nav_msgs/Odometry: stamp (s), pose, linear_x (m/s), angular_z (rad/s)."""

    stamp: float
    pose: Pose
    linear_x: float
    angular_z: float

class BagReader:
    """Context manager iterating a ROS1 .bag's messages in recorded-time order."""

    def __init__(self, path) -> None:
        """
        Create a reader; the bag opens on `__enter__`.

        :param path: str | Path - ROS1 .bag file
        """
        self._reader = AnyReader([Path(path)])

    def __enter__(self) -> BagReader:
        """
        Open the bag.

        :return: BagReader - this reader
        """
        self._reader.open()
        return self

    def __exit__(self, *exc_info) -> None:
        """
        Close the bag without suppressing exceptions.

        :param exc_info: tuple - exception type, value and traceback
        """
        self._reader.close()

    def _connections(self, topics) -> list[Connection]:
        """
        Select connections that carry any of the topics.

        :param topics: Iterable[str] - topic names
        :return: list[Connection] - matching connections
        """
        topics = set(topics)
        return [c for c in self._reader.connections if c.topic in topics]

    def available_topics(self) -> dict[str, str]:
        """
        Map every topic in the bag to its message type.

        :return: dict[str, str] - topic name to message type
        """
        return {c.topic: c.msgtype for c in self._reader.connections}

    def message_count(self, topics) -> int:
        """
        Count the messages recorded on the given topics.

        :param topics: Iterable[str] - topic names; absent topics count as zero
        :return: int - total message count
        """
        return sum(c.msgcount for c in self._connections(topics))

    def messages(self, topics) -> Iterator[tuple[str, str, Any]]:
        """
        Yield messages of the topics in recorded-time order.

        :param topics: Iterable[str] - topic names; yields nothing if none exist in the bag
        :return: Iterator[tuple[str, str, Any]] - (topic, message type, deserialized message)
        """
        connections = self._connections(topics)
        if not connections:  # an empty selection would otherwise mean "all connections"
            return
        for connection, _, raw in self._reader.messages(connections=connections):
            yield connection.topic, connection.msgtype, self._reader.deserialize(raw, connection.msgtype)

def stamp_to_sec(stamp) -> float:
    """
    Convert a ROS time stamp to seconds.

    :param stamp: Any - builtin_interfaces/Time with `sec` and `nanosec`
    :return: float - time in seconds
    """
    return stamp.sec + stamp.nanosec * 1e-9

def odometry_from_msg(msg) -> Odometry:
    """
    Extract the used odometry fields.

    :param msg: Any - nav_msgs/Odometry message
    :return: Odometry - stamp, pose, forward velocity and yaw rate
    """
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return Odometry(
        stamp=stamp_to_sec(msg.header.stamp),
        pose=Pose(np.array([p.x, p.y, p.z], dtype=np.float64), np.array([q.x, q.y, q.z, q.w], dtype=np.float64)),
        linear_x=float(msg.twist.twist.linear.x),
        angular_z=float(msg.twist.twist.angular.z),
    )

def transforms_from_msg(msg) -> list[StampedTransform]:
    """
    Convert a TF message to stamped transforms.

    :param msg: Any - tf2_msgs/TFMessage message
    :return: list[StampedTransform] - one per entry, mapping child to parent coordinates
    """
    transforms = []
    for t in msg.transforms:
        tr = t.transform.translation
        rot = t.transform.rotation
        transforms.append(StampedTransform(
            stamp=stamp_to_sec(t.header.stamp),
            parent=t.header.frame_id,
            child=t.child_frame_id,
            translation=np.array([tr.x, tr.y, tr.z], dtype=np.float64),
            rotation=np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float64),
        ))
    return transforms

def camera_info_from_msg(msg) -> CameraInfo:
    """
    Extract the calibration from a camera info message.

    :param msg: Any - sensor_msgs/CameraInfo message
    :return: CameraInfo - frame id, image size and float64 K, D, R, P arrays
    """
    return CameraInfo(
        frame_id=msg.header.frame_id,
        width=int(msg.width),
        height=int(msg.height),
        K=np.asarray(msg.K, dtype=np.float64),
        D=np.asarray(msg.D, dtype=np.float64),
        R=np.asarray(msg.R, dtype=np.float64),
        P=np.asarray(msg.P, dtype=np.float64),
    )

def is_compressed(msgtype) -> bool:
    """
    Check for a compressed image message type.

    :param msgtype: str - ROS message type name
    :return: bool - True if the type ends with `CompressedImage`
    """
    return msgtype.endswith("CompressedImage")

def raw_image_to_array(msg, decoding=None) -> np.ndarray:
    """
    Decode a sensor_msgs/Image pixel buffer, honoring row stride and endianness.

    :param msg: Any - sensor_msgs/Image message
    :param decoding: dict | None - `image_decoding` config section, None = default
    :return: np.ndarray - (H, W) for one channel, else (H, W, C)
    :raises ValueError: If the encoding is unsupported or the buffer is too small
    """
    if decoding is None:
        decoding = resolve_config()["image_decoding"]
    raw_encodings = decoding["raw_encodings"]
    if msg.encoding not in raw_encodings:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    kind, channels = raw_encodings[msg.encoding]
    dtype = np.dtype(kind).newbyteorder(">" if msg.is_bigendian else "<")
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = width * channels * dtype.itemsize
    if step < row_bytes:
        raise ValueError(f"image step {step} is smaller than a row ({row_bytes} bytes)")
    data = np.asarray(msg.data, dtype=np.uint8)
    if data.size < step * height:
        raise ValueError(f"image buffer has {data.size} bytes, expected at least {step * height}")
    rows = np.ascontiguousarray(data[: step * height].reshape(height, step)[:, :row_bytes])
    arr = rows.view(dtype).astype(dtype.newbyteorder("="), copy=False)
    return arr.reshape(height, width) if channels == 1 else arr.reshape(height, width, channels)

def _decode_compressed(msg, flags) -> np.ndarray:
    """
    Decode a compressed image message with OpenCV.

    :param msg: Any - sensor_msgs/CompressedImage message
    :param flags: int - OpenCV imread flag
    :return: np.ndarray - decoded image
    :raises ValueError: If OpenCV cannot decode the data
    """
    img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), flags)
    if img is None:
        raise ValueError(f"could not decode compressed image (format '{msg.format}')")
    return img

def _to_uint8(arr) -> np.ndarray:
    """
    Convert a uint8 or uint16 image to uint8 (uint16 is scaled by 1/256).

    :param arr: np.ndarray - uint8 (returned unchanged) or uint16 image
    :return: np.ndarray - uint8 image
    :raises ValueError: If the dtype is neither uint8 nor uint16
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.clip(np.rint(arr / 256.0), 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert {arr.dtype} image to 8 bit")

def decode_color(msg, msgtype, decoding=None) -> np.ndarray:
    """
    Decode an RGB topic message to a BGR uint8 image.

    :param msg: Any - sensor_msgs/Image or CompressedImage message
    :param msgtype: str - ROS message type name of `msg`
    :param decoding: dict | None - `image_decoding` config section, None = default
    :return: np.ndarray - (H, W, 3) uint8 BGR image
    :raises ValueError: If the encoding cannot be converted to bgr8
    """
    if is_compressed(msgtype):
        return _decode_compressed(msg, cv2.IMREAD_COLOR)
    if decoding is None:
        decoding = resolve_config()["image_decoding"]
    arr = raw_image_to_array(msg, decoding)
    enc = msg.encoding
    if enc in decoding["bayer_to_bgr"]:
        return cv2.cvtColor(arr, getattr(cv2, decoding["bayer_to_bgr"][enc]))
    if enc in ("bgr8", "8UC3"):
        return arr
    if enc == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 2:
        return cv2.cvtColor(_to_uint8(arr), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"cannot convert encoding '{enc}' to bgr8")

def decode_mono(msg, msgtype, decoding=None) -> np.ndarray:
    """
    Decode an image topic message to a mono8 image.

    :param msg: Any - sensor_msgs/Image or CompressedImage message
    :param msgtype: str - ROS message type name of `msg`
    :param decoding: dict | None - `image_decoding` config section, None = default
    :return: np.ndarray - (H, W) uint8 image
    :raises ValueError: If the encoding cannot be converted to mono8
    """
    if is_compressed(msgtype):
        return _decode_compressed(msg, cv2.IMREAD_GRAYSCALE)
    if decoding is None:
        decoding = resolve_config()["image_decoding"]
    arr = raw_image_to_array(msg, decoding)
    enc = msg.encoding
    if enc in decoding["bayer_to_gray"]:
        return cv2.cvtColor(arr, getattr(cv2, decoding["bayer_to_gray"][enc]))
    if arr.ndim == 2:
        return _to_uint8(arr)
    if enc in ("bgr8", "8UC3"):
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if enc == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if enc == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    if enc == "bgra8":
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"cannot convert encoding '{enc}' to mono8")

def decode_depth(msg, msgtype, decoding=None) -> np.ndarray:
    """
    Decode a depth topic message to float32 metres (16UC1 is millimetres, 32FC1 is metres).

    :param msg: Any - sensor_msgs/Image or CompressedImage message
    :param msgtype: str - ROS message type name of `msg`
    :param decoding: dict | None - `image_decoding` config section, None = default
    :return: np.ndarray - (H, W) float32 depth in metres, NaN where compressedDepth 32FC1 has no reading
    :raises ValueError: If the format or encoding is unsupported or the data is malformed
    """
    if is_compressed(msgtype):
        data = np.asarray(msg.data, dtype=np.uint8)
        fmt = msg.format.lower()
        if "compresseddepth" not in fmt:
            if "16uc1" in fmt:
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img is None or img.ndim != 2 or img.dtype != np.uint16:
                    raise ValueError("compressed 16UC1 depth must be a uint16 PNG")
                return img.astype(np.float32) / 1000.0
            if "32fc1" in fmt:
                raise ValueError("compressed 32FC1 depth requires a compressedDepth header")
            raise ValueError(f"unsupported compressed depth format '{msg.format}'")

        if "png" not in fmt or not ("16uc1" in fmt or "32fc1" in fmt):
            raise ValueError(f"unsupported compressedDepth format '{msg.format}'")
        if data.size <= 12:
            raise ValueError("compressedDepth data is missing its 12-byte header or PNG payload")
        header = data[:12].tobytes()
        compression_format = int.from_bytes(header[:4], "little")
        if compression_format != 0:
            raise ValueError(f"unsupported compressedDepth compression format {compression_format}")
        encoded = data[12:]
        img = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 2 or img.dtype != np.uint16:
            raise ValueError("compressedDepth PNG payload must decode to a uint16 single-channel image")
        if "16uc1" in fmt:
            return img.astype(np.float32) / 1000.0
        # 32FC1 compressedDepth: 12-byte header holds (a, b), depth = a / (disparity - b)
        quant_a, quant_b = np.frombuffer(header[4:], dtype="<f4")
        disparity = img.astype(np.float32)
        depth = np.full(img.shape, np.nan, dtype=np.float32)
        valid = disparity != 0
        denominator = disparity[valid] - quant_b
        depth[valid] = quant_a / denominator
        return depth

    arr = raw_image_to_array(msg, decoding)
    if arr.ndim != 2:
        raise ValueError(f"depth image must have one channel, got encoding '{msg.encoding}'")
    if msg.encoding == "16UC1":
        return arr.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        return arr.astype(np.float32)
    raise ValueError(f"unsupported depth encoding '{msg.encoding}'; expected '16UC1' or '32FC1'")

def depth_to_uint16_mm(depth_metres) -> np.ndarray:
    """Convert metres to saturating uint16 millimetres; 0 for non-finite or non-positive values."""
    depth = np.asarray(depth_metres, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    millimetres = np.zeros(depth.shape, dtype=np.uint16)
    scaled = np.minimum(depth[valid] * 1000.0, np.iinfo(np.uint16).max)
    millimetres[valid] = np.rint(scaled).astype(np.uint16)
    return millimetres
