import argparse
import bisect
import contextlib
import csv
import dataclasses
import glob
import os
import signal
import sys
import threading
import traceback
from collections import Counter, deque
from collections.abc import Iterator

import cv2
import numpy as np
from rosbags.highlevel import AnyReaderError

from data_generation.configuration import load_config, resolve_config
from data_generation.utils.common_utils import require_empty_output
from data_generation.dataset_creation.bag_io import BagReader, camera_info_from_msg, decode_color, decode_depth, depth_to_uint16_mm, odometry_from_msg, stamp_to_sec, transforms_from_msg
from data_generation.dataset_creation.geometry import PinholeCameraModel, Pose, TransformBuffer, TransformError, pose_to_matrix, yaw_from_matrix

def resolve_ros_topics(bag_config, topics_override=None) -> dict:
    """
    Resolve bag topic names and check them against the enabled outputs.

    :param bag_config: dict - bag_processing config section
    :param topics_override: dict | None - replaces bag_processing.ros_topics
    :return: dict - topic names by role (rgb_image, odometry, optionally camera_info, depth_image, rgb_right_image)
    :raises ValueError: If rgb_image or odometry is missing, or trajectory masks are enabled without camera_info or depth_image
    """
    topics = topics_override if topics_override is not None else (bag_config.get("ros_topics") or {})
    if not isinstance(topics, dict):
        raise ValueError("bag_processing.ros_topics must be a mapping of roles to topic names.")
    if not topics.get("rgb_image") or not topics.get("odometry"):
        raise ValueError(
            "bag_processing.ros_topics must define at least 'rgb_image' and 'odometry' "
            "(topic names inside the .bag; 'camera_info' and 'depth_image' are needed for trajectory masks, "
            "'rgb_right_image' is optional). Example:\n"
            "  bag_processing:\n"
            "    ros_topics:\n"
            "      rgb_image: /camera/image_raw\n"
            "      camera_info: /camera/camera_info\n"
            "      depth_image: /camera/depth\n"
            "      odometry: /odom"
        )
    if bag_config.get("save_trajectory_mask", True) and not topics.get("camera_info"):
        raise ValueError(
            "Trajectory masks need a camera calibration: set bag_processing.ros_topics.camera_info "
            "or disable them with bag_processing.save_trajectory_mask: false"
        )
    if bag_config.get("save_trajectory_mask", True) and not topics.get("depth_image"):
        raise ValueError(
            "Trajectory masks need depth data: set bag_processing.ros_topics.depth_image "
            "or disable masks with bag_processing.save_trajectory_mask: false."
        )
    return topics

def _bag_path_problem(bag_file, processed_suffixes) -> str | None:
    """
    Describe why a path is not a readable ROS1 .bag file.

    :param bag_file: str - path given as the bag file
    :param processed_suffixes: Iterable[str] - lowercase suffixes of processed dataset files
    :return: str | None - actionable message, None if the path looks like a valid .bag file
    """
    if not bag_file:
        return "no bag file given: pass a .bag path or set bag_processing.bag_file in the config."
    if os.path.isdir(bag_file):
        names = os.listdir(bag_file)
        if "data.csv" in names or "rgb" in names:
            return (f"'{bag_file}' is a processed dataset directory (it contains data.csv/rgb written by this "
                    "tool), not a raw recording. Select the original .bag file instead; processed output "
                    "directories are the result of this step, not its input.")
        n_bags = len(glob.glob(os.path.join(bag_file, "**", "*.bag"), recursive=True))
        hint = (f"it contains {n_bags} .bag file(s): select them individually, or pass the directory as "
                "--dataset to process all of them" if n_bags else "it contains no .bag files")
        return f"'{bag_file}' is a directory, not a ROS1 .bag file ({hint})."
    if not os.path.exists(bag_file):
        return (f"bag file not found: '{bag_file}'. Check the path, or download/copy the recording "
                "(Download Datasets) before processing it.")
    if os.path.splitext(bag_file)[1].lower() in processed_suffixes:
        return (f"'{bag_file}' looks like a file of a processed dataset (csv/image output), not a raw "
                "recording. Select the original .bag file instead.")
    if not bag_file.endswith(".bag"):
        return f"'{bag_file}' is not a ROS1 .bag file (expected the extension .bag)."
    return None

def validate_bag_input(bag_file, config, topics_override=None) -> None:
    """
    Reject unusable bag input before creating output.

    :param bag_file: str - path to the .bag file
    :param config: dict | None - complete generation config, None = default
    :param topics_override: dict | None - replaces bag_processing.ros_topics
    :raises ValueError: Listing every problem found
    """
    config = resolve_config(config)
    validation = config["validation"]
    max_listed_topics = validation["max_listed_topics"]
    bag_config = config.get("bag_processing") or {}
    tf_topics = tuple(bag_config["tf_topics"])
    topics = resolve_ros_topics(bag_config, topics_override)

    problem = _bag_path_problem(bag_file, validation["processed_file_suffixes"])
    if problem:
        raise ValueError(f"Invalid bag input: {problem}")

    mask_needed = bool(bag_config.get("save_trajectory_mask", True))
    depth_needed = bool(
        bag_config.get("save_depth", True)
        or mask_needed
        or ((bag_config.get("processing") or {}).get("min_mean_depth") or 0) > 0
        or (bag_config.get("black_mask_depth_threshold") or 0.0) > 0
    )

    required = [("rgb_image", topics["rgb_image"]), ("odometry", topics["odometry"])]
    if mask_needed:
        required.append(("camera_info", topics["camera_info"]))
    if depth_needed and topics.get("depth_image"):
        required.append(("depth_image", topics["depth_image"]))

    try:
        with BagReader(bag_file) as bag:
            available = bag.available_topics()
            counts = {topic: bag.message_count([topic]) for _, topic in required}
            tf_count = bag.message_count(tf_topics)
    except PermissionError:
        raise ValueError(
            f"Invalid bag input: no permission to read '{bag_file}'. Fix the file permissions."
        ) from None
    except (AnyReaderError, OSError, UnicodeDecodeError) as exc:
        # invalid bag magic can raise a unicode decoding error
        raise ValueError(
            f"Invalid bag input: '{bag_file}' cannot be read as a ROS1 .bag ({exc}). The file is corrupt, "
            "truncated, unindexed or not a ROS1 bag. Re-download/copy it, or repair it with "
            "`rosbag reindex` / `rosbag fix`."
        ) from exc

    problems = []
    for role, topic in required:
        if topic not in available:
            problems.append(f"{role} topic '{topic}' is not in the bag")
        elif counts[topic] == 0:
            problems.append(f"{role} topic '{topic}' has no messages")
    if mask_needed and tf_count == 0:
        problems.append(f"trajectory masks need transforms, but the bag has no messages on {' or '.join(tf_topics)}")

    if problems:
        listed = sorted(available)[:max_listed_topics]
        more = f" (+{len(available) - len(listed)} more)" if len(available) > len(listed) else ""
        raise ValueError(
            f"Bag '{bag_file}' lacks required data: " + "; ".join(problems) + ". "
            f"Topics in the bag: {', '.join(listed) or 'none'}{more}. "
            "Fix bag_processing.ros_topics in the config to match these names, disable the outputs that "
            "need the missing data (save_trajectory_mask: false), or use a complete recording."
        )

class ValidationTimeout(Exception):
    pass

class TrajectoryMaskError(Exception):
    """A frame's trajectory mask cannot be generated; skip the frame."""

def timeout_handler(signum, frame) -> None:
    """
    Signal handler that raises ValidationTimeout.

    :param signum: int - signal number
    :param frame: types.FrameType - interrupted stack frame
    """
    raise ValidationTimeout("Processing timed out")

@contextlib.contextmanager
def frame_timeout(seconds) -> Iterator[None]:
    """
    Raise ValidationTimeout if the body runs longer than `seconds`; no-op off the main thread or without SIGALRM.

    :param seconds: int - timeout in seconds
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

class BagProcessor:
    """Extract RGB, depth and trajectory masks from a ROS bag, skipping frames closer than min_dist."""

    def __init__(self, config_path="configs/config.yaml", bag_file=None, cli_args=None, topics_override=None) -> None:
        """
        Load config, validate the bag, prepare output, read the odometry trajectory.

        :param config_path: str - YAML config file
        :param bag_file: str - .bag file
        :param cli_args: argparse.Namespace | None - command line options
        :param topics_override: dict | None - replaces bag_processing.ros_topics
        :raises ValueError: If the bag input is invalid or the output directory is not empty
        """
        self.cli_args = cli_args
        self.config = self.load_config(config_path)
        self.topics_override = topics_override
        self.setup_parameters(bag_file)
        validate_bag_input(self.params["bag_file"], self.config, self.topics_override)
        require_empty_output(self.params["output_dir"])
        self.setup_directories()

        self.cam_model = PinholeCameraModel()
        self.tf_buffer = TransformBuffer(
            self.params["tf_buffer_duration"],
            self.params["quaternion_norm_tolerance"],
            self.params["quaternion_epsilon"],
        )

        self.depth_img = None
        self.depth_history = deque(maxlen=3)
        self.mean_depth = 0.0

        self.last_saved_pose = None
        self.issue_counts = Counter()

        self.csv_path = os.path.join(self.params["output_dir"], "data.csv")
        self.init_csv()

        print(f"Loading trajectory from {self.params['bag_file']}...")
        self.traj_times, self.traj_poses = self.load_trajectory_from_bag(self.params["bag_file"])
        print(f"Trajectory loaded with {len(self.traj_times)} poses")

    def load_config(self, config_path) -> dict:
        """
        Load the config with an optional override file.

        :param config_path: str - YAML file
        :return: dict - complete config
        """
        return load_config(config_path)

    def setup_parameters(self, bag_file=None) -> None:
        """
        Set self.params from the config.

        :param bag_file: str | None - bag path
        """
        bag_config = self.config.get("bag_processing", {})

        selected_bag_file = bag_file or ""
        topics = resolve_ros_topics(bag_config, self.topics_override)

        if getattr(self.cli_args, "robot_width", None) is not None:
            robot_width = float(self.cli_args.robot_width)
        else:
            robot_width = float(bag_config["robot"]["width"])

        self.params = {
            "bag_file": selected_bag_file,
            "output_dir": bag_config.get("output_dataset_dir", "dataset_processed"),
            "min_dist": bag_config.get("min_dist", 1.0),
            "min_mean_depth": bag_config["processing"]["min_mean_depth"],

            "topic_rgb": topics.get("rgb_image"),

            "topic_info": topics.get("camera_info"),
            "topic_depth": topics.get("depth_image"),
            "topic_odom": topics.get("odometry"),

            "robot_width": robot_width,
            "horizon_sec": bag_config["trajectory"]["horizon_sec"],
            "start_offset": bag_config["trajectory"]["start_offset"],

            "tf_buffer_duration": bag_config["processing"]["tf_buffer_duration"],
            "frame_timeout_sec": bag_config["processing"]["frame_timeout_sec"],
            "max_images": bag_config["processing"].get("max_images", None),
            "invert_velocity": bag_config.get("invert_velocity", False),
            "save_trajectory_mask": bag_config.get("save_trajectory_mask", True),
            "save_positive_masks": bag_config.get("save_positive_masks", True),
            "save_depth": bag_config.get("save_depth", True),
            "black_mask_depth_threshold": bag_config.get("black_mask_depth_threshold", 0.0),

            "tf_topics": tuple(bag_config["tf_topics"]),
            "global_frame": bag_config["global_frame"],
            "fallback_camera_frame": bag_config["fallback_camera_frame"],
            "image_decoding": self.config["image_decoding"],
            "quaternion_epsilon": self.config["geometry"]["quaternion_epsilon"],
            "quaternion_norm_tolerance": self.config["geometry"]["quaternion_norm_tolerance"],
            "positive_mask_prefix": self.config["data_format"]["positive_mask_prefix"],
        }

        self.bag_name = os.path.splitext(os.path.basename(self.params["bag_file"]))[0]
        self.params["output_dir"] = os.path.join(self.params["output_dir"], "dataset_" + self.bag_name)

    def setup_directories(self) -> None:
        """Create the output directory and enabled subdirectories."""
        if not os.path.exists(self.params["output_dir"]):
            os.makedirs(self.params["output_dir"])

        subdirs = ["rgb"]
        if self.params["save_positive_masks"]:
            subdirs.append("positive_masks")
        if self.params["save_depth"]:
            subdirs.append("depth")
        if self.params["save_trajectory_mask"]:
            subdirs.append("trajectory_masks")
        for subdir in subdirs:
            subdir_path = os.path.join(self.params["output_dir"], subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)

    def init_csv(self) -> None:
        """Write the data.csv header if missing."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["rgb_img_name"]
                if self.params["save_depth"]:
                    header.append("depth_img_name")

                if self.params["save_trajectory_mask"]:
                    header.append("trajectory_img_name")

                if self.params["save_positive_masks"]:
                    header.append("positive_mask_name")

                header.extend([
                    "linear_vel", "angular_vel", "position_x", "position_y", "timestamp"
                ])
                writer.writerow(header)

    def load_trajectory_from_bag(self, path) -> tuple[list[float], list[Pose]]:
        """
        Load the odometry trajectory from the bag.

        :param path: str - .bag file
        :return: tuple[list[float], list[Pose]] - timestamps in seconds and poses, sorted by time
        """
        entries = []
        with BagReader(path) as bag:
            for _, _, msg in bag.messages([self.params["topic_odom"]]):
                odom = odometry_from_msg(msg)
                entries.append((odom.stamp, odom.pose))
        entries.sort(key=lambda entry: entry[0])
        return [e[0] for e in entries], [e[1] for e in entries]

    def process_bag(self) -> int:
        """
        Stream the bag in recorded-time order, saving a frame for each accepted RGB image.

        RGB, depth, camera info, odometry and the configured TF topics share one stream, so a frame only sees the state recorded before it.

        :return: int - number of saved frames, always positive
        :raises FileNotFoundError: If the bag file does not exist
        :raises RuntimeError: If no frame was saved; nothing is deleted and no completion marker is written
        """
        if not os.path.exists(self.params["bag_file"]):
            raise FileNotFoundError(f"Bag file not found: {self.params['bag_file']}")

        print(f"Processing bag file: {self.params['bag_file']}")

        topics = [
            self.params["topic_rgb"],
            self.params["topic_depth"],
            self.params["topic_info"],
            self.params["topic_odom"],
            *self.params["tf_topics"],
        ]
        topics = list(dict.fromkeys(t for t in topics if t))

        if self.params["save_trajectory_mask"] and not self.params["topic_depth"]:
            print("WARNING: no depth topic configured; trajectory masks will be empty.")

        count_saved = 0
        count_rgb = 0
        with BagReader(self.params["bag_file"]) as bag:
            available = bag.available_topics()
            for name in (self.params["topic_rgb"], self.params["topic_odom"], self.params["topic_info"],
                         self.params["topic_depth"]):
                if name and name not in available:
                    print(f"WARNING: topic {name} not found in {self.params['bag_file']}")

            print(f"Total messages to scan: {bag.message_count(topics)}")

            latest_odom = None

            for topic, msgtype, msg in bag.messages(topics):
                if self.params["max_images"] is not None and count_saved >= self.params["max_images"]:
                    print(f"Reached limit of {self.params['max_images']} images.")
                    break

                if topic in self.params["tf_topics"]:
                    self.handle_tf(topic, msg)

                elif topic == self.params["topic_info"]:
                    self.cam_model.update(camera_info_from_msg(msg))

                elif topic == self.params["topic_odom"]:
                    latest_odom = odometry_from_msg(msg)

                elif topic == self.params["topic_depth"]:
                    self.handle_depth(msg, msgtype)

                elif topic == self.params["topic_rgb"]:
                    count_rgb += 1
                    if self.process_frame(msg, msgtype, count_saved, latest_odom):
                        count_saved += 1

        print(f"Finished processing bag file. Saved {count_saved} frames.")
        reasons = (self.issue_counts + self.tf_buffer.rejected).most_common()
        for reason, n in reasons:
            print(f"  {n} x {reason}")
        if count_saved == 0:
            detail = "".join(f"; {n} x {reason}" for reason, n in reasons)
            raise RuntimeError(
                f"No frame was saved from {self.params['bag_file']}: none of {count_rgb} RGB messages passed the "
                f"odometry, distance, reverse-motion, depth and trajectory mask checks{detail}. "
                f"processing_completed.txt was not written; files already in {self.params['output_dir']} were left untouched."
            )
        return count_saved

    def handle_tf(self, topic, msg) -> None:
        """
        Store a TF message's transforms in the buffer.

        :param topic: str - topic name, /tf_static transforms are stored as static
        :param msg: tf2_msgs/TFMessage - transform message
        """
        static = topic.endswith("tf_static")
        for transform in transforms_from_msg(msg):
            self.tf_buffer.set_transform(transform, static=static)

    def handle_depth(self, msg, msgtype) -> None:
        """
        Decode a depth image and update the moving average of its mean depth.

        :param msg: sensor_msgs/Image | sensor_msgs/CompressedImage - depth message
        :param msgtype: str - ROS message type name of msg
        """
        try:
            self.depth_img = decode_depth(msg, msgtype, self.params["image_decoding"])

            valid_mask = np.isfinite(self.depth_img) & (self.depth_img > 0)

            if np.any(valid_mask):
                current_mean = np.mean(self.depth_img[valid_mask])
                self.depth_history.append(current_mean)
                self.mean_depth = sum(self.depth_history) / len(self.depth_history)
        except (ValueError, cv2.error) as e:
            self.issue_counts["depth decode failed"] += 1
            print(e)

    def check_distance(self, current_pose) -> bool | np.bool_:
        """
        Check the robot moved at least min_dist since the last saved frame.

        :param current_pose: Pose - current robot pose, position in meters
        :return: bool - True if nothing saved yet or planar distance >= min_dist
        """
        if self.last_saved_pose is None:
            return True

        dx = current_pose.position[0] - self.last_saved_pose.position[0]
        dy = current_pose.position[1] - self.last_saved_pose.position[1]
        dist = np.sqrt(dx*dx + dy*dy)

        return dist >= self.params["min_dist"]

    def _skip(self, count, reason) -> bool:
        """
        Count and report a skipped frame.

        :param count: int - frame index for the log message
        :param reason: str - why the frame was skipped
        :return: bool - always False
        """
        self.issue_counts[f"frame skipped: {reason}"] += 1
        print(f"Frame {count}: skipped, {reason}")
        return False

    def process_frame(self, msg, msgtype, count, odom) -> bool:
        """
        Validate one RGB frame, generate its masks and save it.

        :param msg: sensor_msgs/Image | sensor_msgs/CompressedImage - RGB image message
        :param msgtype: str - ROS message type name of msg
        :param count: int - index assigned to the frame if saved
        :param odom: bag_io.Odometry | None - latest odometry message
        :return: bool - True if the frame was saved, False if skipped
        """
        if odom is None:
            print(f"Frame {count}: No odom message")
            self.issue_counts["frame skipped: no odom message yet"] += 1
            return False

        if not self.check_distance(odom.pose):
            return False

        t_img = stamp_to_sec(msg.header.stamp)

        print(f"Processing frame {count}...", end="\r")

        try:
            with frame_timeout(self.params["frame_timeout_sec"]):
                try:
                    img = decode_color(msg, msgtype, self.params["image_decoding"])
                except (ValueError, cv2.error):
                    img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except ValidationTimeout:
            self.issue_counts["frame skipped: image decoding timed out"] += 1
            return False
        except Exception as e:
            print(f"Error decoding image {count}: {e}")
            self.issue_counts["frame skipped: image decoding error"] += 1
            return False

        if img is None:
            print(f"Frame {count}: Image is None after decoding")
            self.issue_counts["frame skipped: image could not be decoded"] += 1
            return False

        min_depth = self.params.get("min_mean_depth", 0.0)
        if min_depth > 0:
            if self.mean_depth < min_depth:
                return False

        linear_vel = odom.linear_x
        if getattr(self.cli_args, 'remove_reverse', 'true').lower() == 'true':
            if linear_vel < -0.05:
                return False

        if self.params["invert_velocity"]:
            linear_vel = -linear_vel

        # copy: the shared latest odometry message must stay unchanged
        odom = dataclasses.replace(odom, linear_x=linear_vel)

        force_black_mask = False
        if self.params["black_mask_depth_threshold"] > 0 and self.mean_depth < self.params["black_mask_depth_threshold"]:
            force_black_mask = True

        if force_black_mask:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            print(f"Frame {count}: Forced black mask due to low depth ({self.mean_depth:.2f} < {self.params['black_mask_depth_threshold']})")
        elif self.params["save_trajectory_mask"]:
            try:
                mask = self.generate_trajectory_mask(img.shape[:2], t_img, odom)
            except TrajectoryMaskError as e:
                return self._skip(count, f"trajectory mask unavailable ({e})")
        else:
            mask = None

        if hasattr(self.cli_args, 'width') and hasattr(self.cli_args, 'height'):
            img = cv2.resize(img, (int(self.cli_args.width), int(self.cli_args.height)))
            if self.depth_img is not None:
                self.depth_img = cv2.resize(self.depth_img, (int(self.cli_args.width), int(self.cli_args.height)), interpolation=cv2.INTER_NEAREST)
            if mask is not None:
                mask = cv2.resize(mask, (int(self.cli_args.width), int(self.cli_args.height)), interpolation=cv2.INTER_NEAREST)

        self.save_data(img, mask, count, odom, force_black_mask)

        self.last_saved_pose = odom.pose

        return True

    def _select_trajectory_poses(self, t_img, odom) -> list[Pose]:
        """
        Select the poses used to draw the trajectory mask.

        :param t_img: float - frame time in seconds
        :param odom: bag_io.Odometry - odometry at the frame time
        :return: list[Pose] - current pose followed by past poses (newest first) when reversing, else poses within the horizon
        """
        horizon = self.params["horizon_sec"]
        offset = self.params["start_offset"]

        if odom.linear_x < -0.05:
            lo = bisect.bisect_right(self.traj_times, t_img - horizon)
            hi = bisect.bisect_left(self.traj_times, t_img)
            poses = self.traj_poses[lo:hi][::-1]
        else:
            lo = bisect.bisect_right(self.traj_times, t_img - offset)
            hi = bisect.bisect_left(self.traj_times, t_img + horizon)
            poses = self.traj_poses[lo:hi]

        return [odom.pose, *poses]

    def generate_trajectory_mask(self, img_shape, t_img, odom) -> np.ndarray:
        """
        Project the trajectory into the camera image as a mask; drawn only if mean depth exceeds 1 m.

        :param img_shape: tuple[int, int] - mask (H, W)
        :param t_img: float - frame time in seconds
        :param odom: bag_io.Odometry - odometry at the frame time
        :return: np.ndarray - (H, W) uint8 mask, path pixels 255
        :raises TrajectoryMaskError: If no camera calibration was received or no transform to the camera frame exists
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        valid_poses = self._select_trajectory_poses(t_img, odom)

        if not self.cam_model.ready:
            raise TrajectoryMaskError("no camera_info message received yet")

        target_frame = self.cam_model.tf_frame if self.cam_model.tf_frame else self.params["fallback_camera_frame"]
        global_frame = self.params["global_frame"]
        try:
            trans_cam_base = self.tf_buffer.lookup_latest(target_frame, global_frame)
        except TransformError as e:
            raise TrajectoryMaskError(f"TF {global_frame} -> {target_frame}: {e}") from e
        rot_cam_base = trans_cam_base[:3, :3]
        pos_cam_base = trans_cam_base[:3, 3]

        mat_base_odom = np.linalg.inv(pose_to_matrix(odom.pose, self.params["quaternion_epsilon"]))

        prev_left = None
        prev_right = None
        traj_shape = getattr(self.cli_args, 'traj_shape', 'polygon')

        if self.mean_depth > 1:
            for pose in valid_poses:
                mat_p_base = mat_base_odom @ pose_to_matrix(pose, self.params["quaternion_epsilon"])

                yaw_rel = yaw_from_matrix(mat_p_base, self.params["quaternion_epsilon"])

                x_rel = mat_p_base[0, 3]
                y_rel = mat_p_base[1, 3]
                z_rel = mat_p_base[2, 3]

                num_offsets = 3
                offsets = np.linspace(-self.params["robot_width"]/2, self.params["robot_width"]/2, num_offsets)

                curr_left = None
                curr_right = None

                for line_idx, off in enumerate(offsets):
                    dx = np.cos(yaw_rel + 1.5708) * off
                    dy = np.sin(yaw_rel + 1.5708) * off

                    p_cam = rot_cam_base @ np.array([x_rel + dx, y_rel + dy, z_rel]) + pos_cam_base
                    if p_cam[2] > 0.1:
                        u, v = self.cam_model.project3d_to_pixel(p_cam)
                        if not (np.isfinite(u) and np.isfinite(v)):
                            continue

                        if line_idx == 0:
                            curr_left = (int(u), int(v))
                        elif line_idx == num_offsets - 1:
                            curr_right = (int(u), int(v))

                        if traj_shape == 'point' and 0 <= u < w and 0 <= v < h:
                            cv2.circle(mask, (int(u), int(v)), 3, 255, -1)

                if traj_shape == 'polygon':
                    if curr_left is not None and curr_right is not None and prev_left is not None and prev_right is not None:
                        poly_pts = np.array([prev_left, curr_left, curr_right, prev_right], dtype=np.int32)
                        cv2.fillPoly(mask, [poly_pts], 255)

                if curr_left is not None:
                    prev_left = curr_left
                if curr_right is not None:
                    prev_right = curr_right

        return mask

    def save_data(self, img, mask, count, odom, force_black_mask=False) -> None:
        """
        Write one frame's images and append its data.csv row.

        :param img: np.ndarray - BGR image
        :param mask: np.ndarray | None - trajectory mask, also saved as positive mask when enabled
        :param count: int - frame index used in file names
        :param odom: bag_io.Odometry - odometry providing velocity, position in meters and timestamp in seconds
        :param force_black_mask: bool - unused; True when the mask was forced to black (low mean depth)
        """
        out_dir = self.params["output_dir"]

        rgb_filename = f"rgb_{self.bag_name}_{count:06d}.jpg"
        rgb_path = os.path.join("rgb", rgb_filename)
        cv2.imwrite(os.path.join(out_dir, rgb_path), img)

        mask_path = ""
        if self.params["save_trajectory_mask"] and mask is not None:
            mask_filename = f"trajectory_{self.bag_name}_{count:06d}.jpg"
            mask_path = os.path.join("trajectory_masks", mask_filename)
            cv2.imwrite(os.path.join(out_dir, mask_path), mask)
        timestamp = odom.stamp
        linear_vel = odom.linear_x
        angular_vel = odom.angular_z
        pos_x, pos_y = odom.pose.position[:2]

        row = [rgb_path]
        if self.params["save_depth"]:
            depth_path = ""
            if self.depth_img is not None:
                depth_filename = f"depth_{self.bag_name}_{count:06d}.tif"
                depth_path = os.path.join("depth", depth_filename)
                depth_uint16 = depth_to_uint16_mm(self.depth_img)
                cv2.imwrite(os.path.join(out_dir, depth_path), depth_uint16)
            row.append(depth_path)

        if self.params["save_trajectory_mask"]:
            row.append(mask_path)

        if self.params["save_positive_masks"]:
            positive_mask_path = ""
            if mask is not None:
                positive_mask_filename = (
                    f"{self.params['positive_mask_prefix']}{os.path.splitext(rgb_filename)[0]}.tif"
                )
                positive_mask_path = os.path.join("positive_masks", positive_mask_filename)
                cv2.imwrite(os.path.join(out_dir, positive_mask_path), mask)
            row.append(positive_mask_path)

        row.extend([
            linear_vel, angular_vel, float(pos_x), float(pos_y), timestamp
        ])

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def build_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser.

    :return: argparse.ArgumentParser - parser for the config, dataset, resize and trajectory options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, required=True, help="Path to one .bag file or a directory of bags.")
    parser.add_argument("--width", type=int, default=640, help="Resize image width")
    parser.add_argument("--height", type=int, default=480, help="Resize image height")
    parser.add_argument("--remove_reverse", type=str, default="true", help="Remove reverse motion frames")
    parser.add_argument("--traj_shape", type=str, default="polygon", help="Trajectory shape: point or polygon")
    parser.add_argument("--robot_width", type=float, default=None, help="Override robot width for trajectory masks")
    return parser

def main(argv=None) -> None:
    """
    Process the selected bag files into datasets.

    :param argv: list[str] | None - command line arguments, None = sys.argv
    """
    args = build_parser().parse_args(argv)

    bags_to_process = []
    if os.path.isdir(args.dataset):
        bags_to_process = sorted(glob.glob(os.path.join(args.dataset, "**", "*.bag"), recursive=True))
    elif args.dataset.endswith(".bag"):
        bags_to_process = [args.dataset]
    else:
        raise ValueError(f"--dataset must be a .bag file or a directory containing .bag files: {args.dataset}")
    if not bags_to_process:
        raise ValueError(f"No .bag files found at --dataset path: {args.dataset}")

    print(f"Found {len(bags_to_process)} bags to process.")
    for bag_file in bags_to_process:
        try:
            print(f"\n--- Starting processing for: {bag_file} ---")
            processor = BagProcessor(config_path=args.config, bag_file=bag_file, cli_args=args)
            processor.process_bag()
            with open(os.path.join(processor.params["output_dir"], "processing_completed.txt"), "w") as f:
                f.write("done")
            print(f"--- Completed processing for: {bag_file} ---\n")
        except Exception as e:
            print(f"ERROR processing bag {bag_file}: {e}")
            traceback.print_exc()
            sys.exit(1)




if __name__ == '__main__':
    main()
