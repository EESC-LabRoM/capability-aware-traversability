import argparse
import csv
import json
import os

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

from data_generation.configuration import load_config
from data_generation.dataset_creation.geometry import (
    reproject_depth_to_rgb,
    scale_intrinsics,
    validate_camera_distortion,
)
from data_generation.utils.common_utils import require_empty_output
from data_generation.dataset_creation.bag_io import depth_to_uint16_mm

def _grandtour_static_transform(tf: dict, target: str, source: str) -> np.ndarray:
    """Return the 4x4 transform from `source` to `target` coordinates."""
    def matrix(record: dict) -> np.ndarray:
        translation, rotation = record["translation"], record["rotation"]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = R.from_quat(
            [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
        ).as_matrix()
        result[:3, 3] = [translation["x"], translation["y"], translation["z"]]
        return result

    # records are T_frame<-parent with parent `base` or `box_base`, box_base children go through the box_base record
    bridge = matrix(tf["box_base"])

    def frame_from_base(frame: str) -> np.ndarray:
        if frame == "base":
            return np.eye(4, dtype=np.float64)
        record = tf[frame]
        transform = matrix(record)
        if record["base_frame_id"] == "box_base":
            return transform @ bridge
        if record["base_frame_id"] != "base":
            raise ValueError(
                f"TF frame '{frame}' has unsupported parent '{record['base_frame_id']}'; expected base or box_base"
            )
        return transform

    return frame_from_base(target) @ np.linalg.inv(frame_from_base(source))

def validate_grandtour_input(dataset_dir, save_depth=True) -> None:
    """
    Validate required raw streams and images without creating output.

    :param dataset_dir: str - raw GrandTour mission directory with data/ and images/
    :param save_depth: bool - whether depth streams are required
    :raises ValueError: If the mission is already processed or streams, camera metadata or RGB images are missing or invalid
    """
    if os.path.isfile(os.path.join(dataset_dir, "data.csv")):
        raise ValueError(
            f"{dataset_dir} is already processed. Use Manage Local Datasets "
            "or segmentation instead of Process Raw Dataset."
        )
    problems = []
    groups = {}
    required_streams = ["anymal_state_odometry", "hdr_front"]
    if save_depth:
        required_streams.append("depth_camera_front_upper")
    for stream in required_streams:
        path = os.path.join(dataset_dir, "data", stream)
        try:
            group = zarr.open_group(path, mode="r")
            times = group["timestamp"][:]
            if (times.ndim != 1 or times.size == 0 or not np.isfinite(times).all()
                    or np.any(np.diff(times) < 0)):
                raise ValueError("timestamps must be nonempty, finite and ordered")
            groups[stream] = group
            if stream == "anymal_state_odometry":
                for key, width in (("pose_pos", 3), ("pose_orien", 4),
                                   ("twist_lin", 3), ("twist_ang", 3)):
                    array = group[key]
                    if array.shape != (len(times), width):
                        raise ValueError(f"{key} shape {array.shape} does not match timestamps")
                    if not np.isfinite(array[[0, len(times) - 1]]).all():
                        raise ValueError(f"{key} contains invalid values")
            else:
                attrs = group.attrs
                camera = attrs["camera_info"]
                if (np.asarray(camera["K"]).shape != (9,)
                        or not np.isfinite(camera["K"]).all()
                        or camera["K"][0] <= 0 or camera["K"][4] <= 0):
                    raise ValueError("camera_info.K must contain valid camera intrinsics")
                validate_camera_distortion(camera["distortion_model"], camera["D"])
                transform = attrs["transform"]
                for field, axes in (("translation", "xyz"), ("rotation", "xyzw")):
                    if not all(np.isfinite(transform[field][axis]) for axis in axes):
                        raise ValueError(f"invalid camera transform {field}")
                if sum(transform["rotation"][axis] ** 2 for axis in "xyzw") == 0:
                    raise ValueError("camera transform has a zero quaternion")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            problems.append(f"data/{stream}: {exc}")
    try:
        with open(os.path.join(dataset_dir, "data", "tf", ".zattrs"), encoding="utf-8") as source:
            tf = json.load(source)["tf"]
        if (
            tf.get("box_base", {}).get("base_frame_id") != "base"
            or tf.get("box_base", {}).get("child_frame_id") != "box_base"
        ):
            raise ValueError("TF metadata is missing the base-to-box_base transform")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"data/tf: {exc}")
    if "hdr_front" in groups:
        images_dir = os.path.join(dataset_dir, "images", "hdr_front")
        first_image = None
        missing = 0
        for index in range(len(groups["hdr_front"]["timestamp"])):
            candidates = (f"{index:06d}.jpeg", f"{index:06d}.png",
                          f"{index}.png", f"{index}.jpeg")
            image = next((os.path.join(images_dir, name) for name in candidates
                          if os.path.isfile(os.path.join(images_dir, name))), None)
            if image is None:
                missing += 1
            elif first_image is None:
                first_image = image
        if missing:
            problems.append(f"images/hdr_front: {missing} referenced RGB frames are missing")
        if first_image:
            rgb_sample = cv2.imread(first_image)
            camera = groups["hdr_front"].attrs["camera_info"]
            if rgb_sample is None:
                problems.append(f"Unreadable RGB image: {first_image}")
            elif rgb_sample.shape[:2] != (camera["height"], camera["width"]):
                problems.append(f"RGB image dimensions do not match hdr_front calibration: {first_image}")
    if save_depth and "depth_camera_front_upper" in groups:
        depth_dir = os.path.join(dataset_dir, "images", "depth_camera_front_upper")
        timestamps = groups["depth_camera_front_upper"]["timestamp"]
        for index in range(len(timestamps)):
            candidates = (f"{index:06d}.png", f"{index}.png")
            image = next(
                (os.path.join(depth_dir, name) for name in candidates
                 if os.path.isfile(os.path.join(depth_dir, name))),
                None,
            )
            if image is None:
                problems.append(f"images/depth_camera_front_upper: missing frame {index}")
                break
            if index == 0:
                sample = cv2.imread(image, cv2.IMREAD_UNCHANGED)
                camera = groups["depth_camera_front_upper"].attrs["camera_info"]
                if sample is None or sample.dtype != np.uint16 or sample.ndim != 2:
                    problems.append(
                        "images/depth_camera_front_upper: expected 16-bit millimeter depth PNGs"
                    )
                elif sample.shape != (camera["height"], camera["width"]):
                    problems.append(
                        "Depth image dimensions do not match depth_camera_front_upper calibration"
                    )
    if problems:
        raise ValueError(
            f"Incomplete or invalid GrandTour recording: {dataset_dir}. "
            + "; ".join(problems)
            + ". Download this Grandtour mission again from Download Datasets "
            "or restore its required streams and images before processing."
        )

class GrandTourProcessor:
    def __init__(self, config_path="configs/config.yaml", dataset_dir=None, cli_args=None) -> None:
        """
        Load config and data, validate input, prepare the output directory.

        :param config_path: str - YAML config file
        :param dataset_dir: str | None - raw mission directory, None = bag_processing.dataset_dir
        :param cli_args: argparse.Namespace | None - optional config overrides
        :raises ValueError: If the input is invalid or the output directory is not empty
        """
        self.cli_args = cli_args
        self.config = self.load_config(config_path)
        self.dataset_dir = dataset_dir if dataset_dir else self.config.get("bag_processing", {}).get("dataset_dir", "datasets/grand_tour")

        self.dataset_dir = os.path.abspath(self.dataset_dir)
        self.bag_name = os.path.basename(self.dataset_dir)

        self.setup_parameters()
        validate_grandtour_input(self.dataset_dir, save_depth=self.params["save_depth"])
        require_empty_output(self.params["output_dir"])
        self.load_data()
        self.setup_directories()

        self.csv_path = os.path.join(self.params["output_dir"], "data.csv")
        self.init_csv()

        self.last_saved_pos = None

    def load_config(self, config_path) -> dict:
        """
        Load the config with an optional override file.

        :param config_path: str - YAML file
        :return: dict - complete config
        """
        return load_config(config_path)

    def setup_parameters(self) -> None:
        """Set self.params from the config."""
        bag_config = self.config.get("bag_processing", {})

        self.params = {
            "output_dir": bag_config.get("output_dataset_dir", "dataset_processed"),
            "min_dist": bag_config.get("min_dist", 1.0),

            "robot_width": float(self.cli_args.robot_width if getattr(self.cli_args, "robot_width", None) is not None else bag_config.get("robot", {}).get("width", 0.6)),
            "horizon_sec": bag_config.get("trajectory", {}).get("horizon_sec", 2.0),
            "start_offset": bag_config.get("trajectory", {}).get("start_offset", 0.0),

            "max_images": bag_config.get("processing", {}).get("max_images", None),
            "invert_velocity": bag_config.get("invert_velocity", False),
            "save_trajectory_mask": bag_config.get("save_trajectory_mask", True),
            "save_depth": bag_config.get("save_depth", True),
            "odom_time_tolerance": self.config["validation"]["odom_time_tolerance"],
        }

        self.params["output_dir"] = os.path.join(self.params["output_dir"], "dataset_grandtour_" + self.bag_name)

    def setup_directories(self) -> None:
        """Create the output directory and subdirectories."""
        if not os.path.exists(self.params["output_dir"]):
            os.makedirs(self.params["output_dir"])

        subdirs = ["rgb"]
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

                header.extend([
                    "linear_vel", "angular_vel", "position_x", "position_y", "timestamp"
                ])
                writer.writerow(header)

    def load_data(self) -> None:
        """Open the Zarr streams, cache odometry, read camera intrinsics and camera-to-base extrinsics."""
        print(f"Loading data from {self.dataset_dir}...")
        odom_path = os.path.join(self.dataset_dir, "data/anymal_state_odometry")
        hdr_path = os.path.join(self.dataset_dir, "data/hdr_front")

        self.odom_z = zarr.open(odom_path, mode='r')
        self.hdr_z = zarr.open(hdr_path, mode='r')

        self.odom_times = self.odom_z['timestamp'][:]
        self.hdr_times = self.hdr_z['timestamp'][:]

        print("Caching odometry arrays to memory...")
        self.odom_pos = self.odom_z['pose_pos'][:]
        self.odom_orien = self.odom_z['pose_orien'][:]
        self.odom_twist_lin = self.odom_z['twist_lin'][:]
        self.odom_twist_ang = self.odom_z['twist_ang'][:]

        with open(os.path.join(hdr_path, ".zattrs"), "r", encoding="utf-8") as source:
            hdr_attrs = json.load(source)
        hdr_camera = hdr_attrs["camera_info"]
        self.rgb_K = np.asarray(hdr_camera["K"], dtype=np.float64).reshape(3, 3)
        self.rgb_model = hdr_camera["distortion_model"]
        self.rgb_D = validate_camera_distortion(self.rgb_model, hdr_camera["D"])
        self.rgb_frame = hdr_camera["frame_id"]
        self.distortion_model = self.rgb_model
        self.camera_matrix = self.rgb_K
        self.dist_coeffs = self.rgb_D

        if hdr_attrs["transform"]["child_frame_id"] != self.rgb_frame:
            raise ValueError("HDR transform child_frame_id does not match camera frame")
        with open(os.path.join(self.dataset_dir, "data/tf/.zattrs"), "r", encoding="utf-8") as source:
            tf_records = dict(json.load(source)["tf"])
        tf_records[self.rgb_frame] = hdr_attrs["transform"]
        self.trans_cam_base_mat = _grandtour_static_transform(tf_records, self.rgb_frame, "base")
        self.rgb_from_depth = None
        if self.params["save_depth"]:
            depth_path = os.path.join(self.dataset_dir, "data/depth_camera_front_upper")
            self.depth_z = zarr.open_group(depth_path, mode="r")
            self.depth_times = self.depth_z["timestamp"][:]
            with open(os.path.join(depth_path, ".zattrs"), "r", encoding="utf-8") as source:
                depth_attrs = json.load(source)
            depth_camera = depth_attrs["camera_info"]
            self.depth_K = np.asarray(depth_camera["K"], dtype=np.float64).reshape(3, 3)
            self.depth_model = depth_camera["distortion_model"]
            self.depth_D = validate_camera_distortion(self.depth_model, depth_camera["D"])
            if depth_attrs["transform"]["child_frame_id"] != depth_camera["frame_id"]:
                raise ValueError("Depth transform child_frame_id does not match camera frame")
            depth_frame = depth_camera["frame_id"]
            tf_records[depth_frame] = depth_attrs["transform"]
            self.rgb_from_depth = _grandtour_static_transform(tf_records, self.rgb_frame, depth_frame)

    def get_closest_idx(self, times, t, max_diff=None) -> int | np.intp | None:
        """
        Find the index of the timestamp closest to `t`.

        :param times: np.ndarray | None - sorted timestamps in seconds
        :param t: float - query time in seconds
        :param max_diff: float | None - max allowed difference in seconds, None = config tolerance
        :return: int | None - closest index, None if empty or farther than max_diff
        """
        if max_diff is None:
            max_diff = self.params["odom_time_tolerance"]
        if times is None or len(times) == 0:
            return None
        idx = np.searchsorted(times, t)
        if idx >= len(times):
            idx = len(times) - 1

        if idx > 0 and abs(times[idx-1] - t) < abs(times[idx] - t):
            idx = idx - 1

        if abs(times[idx] - t) > max_diff:
            return None
        return idx

    def check_distance(self, current_pos) -> bool | np.bool_:
        """
        Check the position is at least min_dist from the last saved one.

        :param current_pos: np.ndarray - position, x and y in meters first
        :return: bool - True if nothing saved yet or planar distance >= min_dist
        """
        if self.last_saved_pos is None:
            return True
        dx = current_pos[0] - self.last_saved_pos[0]
        dy = current_pos[1] - self.last_saved_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        return dist >= self.params["min_dist"]

    def _project(self, pts_cam) -> np.ndarray:
        """
        Project camera-frame points to pixels with the distortion model.

        :param pts_cam: np.ndarray - (N, 3) camera-frame points in meters
        :return: np.ndarray - (N, 2) pixel coordinates
        """
        if self.distortion_model == "equidistant":
            pts_2d, _ = cv2.fisheye.projectPoints(
                pts_cam.reshape(-1, 1, 3).astype(np.float64),
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                self.camera_matrix.astype(np.float64),
                self.dist_coeffs.astype(np.float64).reshape(4, 1),
            )
        else:
            pts_2d, _ = cv2.projectPoints(
                pts_cam.astype(np.float64), np.zeros(3), np.zeros(3),
                self.camera_matrix.astype(np.float64), self.dist_coeffs.astype(np.float64),
            )
        return pts_2d.reshape(-1, 2)

    def _get_mat_from_odom(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Build an odometry pose from the xyzw quaternion.

        :param idx: int - odometry sample index
        :return: tuple[np.ndarray, np.ndarray] - 4x4 pose matrix and position in meters
        """
        position = self.odom_pos[idx]
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat(self.odom_orien[idx]).as_matrix()
        matrix[:3, 3] = position
        return matrix, position

    def generate_trajectory_mask(self, img_shape, current_t, current_odom_idx) -> np.ndarray:
        """
        Project the path around the frame time into the camera image, using past odometry when reversing.

        :param img_shape: tuple[int, int] - mask (H, W)
        :param current_t: float - frame time in seconds
        :param current_odom_idx: int - odometry index of the current frame
        :return: np.ndarray - (H, W) uint8 mask, path pixels 255
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        horizon = self.params["horizon_sec"]
        offset = self.params["start_offset"]

        v = self.odom_twist_lin[current_odom_idx][0]

        if v < -0.05:
            valid_indices = np.where((self.odom_times >= current_t - horizon) & (self.odom_times <= current_t))[0]
            valid_indices = valid_indices[::-1]
        else:
            valid_indices = np.where((self.odom_times >= current_t - offset) & (self.odom_times <= current_t + horizon))[0]

        if len(valid_indices) == 0:
            return mask

        valid_indices = np.insert(valid_indices, 0, current_odom_idx)

        mat_curr, _ = self._get_mat_from_odom(current_odom_idx)
        mat_base_odom = np.linalg.inv(mat_curr)

        cam_ext = self.trans_cam_base_mat

        prev_left = None
        prev_right = None

        for idx in valid_indices:
            mat_p, _ = self._get_mat_from_odom(idx)

            mat_p_base = mat_base_odom @ mat_p

            rotation = mat_p_base[:3, :3]
            yaw_rel = np.arctan2(rotation[1, 0], rotation[0, 0])

            x_rel = mat_p_base[0, 3]
            y_rel = mat_p_base[1, 3]
            z_rel = mat_p_base[2, 3] - 0.50

            num_offsets = 3
            offsets = np.linspace(-self.params["robot_width"]/2, self.params["robot_width"]/2, num_offsets)

            points_base = np.zeros((num_offsets, 4))
            points_base[:, 0] = x_rel + np.cos(yaw_rel + 1.5708) * offsets
            points_base[:, 1] = y_rel + np.sin(yaw_rel + 1.5708) * offsets
            points_base[:, 2] = z_rel
            points_base[:, 3] = 1.0

            pts_cam = (cam_ext @ points_base.T).T

            valid_z = pts_cam[:, 2] > 0.1
            if not np.any(valid_z):
                prev_left = None
                prev_right = None
                continue

            pts_2d = self._project(pts_cam[:, :3])

            curr_left = None
            curr_right = None

            if valid_z[0]:
                curr_left = (int(pts_2d[0, 0]), int(pts_2d[0, 1]))
            if valid_z[-1]:
                curr_right = (int(pts_2d[-1, 0]), int(pts_2d[-1, 1]))

            if getattr(self.cli_args, 'traj_shape', 'polygon') == 'point':
                for line_idx in range(num_offsets):
                    if valid_z[line_idx]:
                        px, py = int(pts_2d[line_idx, 0]), int(pts_2d[line_idx, 1])
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(mask, (px, py), 3, 255, -1)
            else:
                if curr_left is not None and curr_right is not None and prev_left is not None and prev_right is not None:
                    poly_pts = np.array([prev_left, curr_left, curr_right, prev_right], dtype=np.int32)
                    cv2.fillPoly(mask, [poly_pts], 255)

            if curr_left is not None:
                prev_left = curr_left
            if curr_right is not None:
                prev_right = curr_right

        return mask

    def process_dataset(self) -> None:
        """
        Convert selected frames to the output dataset and append them to data.csv.

        Skips frames without close odometry, too near the last saved frame, without RGB, or reversing.
        """
        count_saved = 0
        total_images = len(self.hdr_times)

        print(f"Total images to process: {total_images}")

        hdr_images_dir = os.path.join(self.dataset_dir, "images/hdr_front")
        depth_images_dir = os.path.join(self.dataset_dir, "images/depth_camera_front_upper") if self.params["save_depth"] else None

        for i, t_img in enumerate(self.hdr_times):
            if self.params["max_images"] is not None and count_saved >= self.params["max_images"]:
                print(f"\nReached limit of {self.params['max_images']} images.")
                break

            odom_idx = self.get_closest_idx(self.odom_times, t_img)
            if odom_idx is None:
                continue

            p_curr = self.odom_pos[odom_idx]
            if not self.check_distance(p_curr):
                continue

            img_path = os.path.join(hdr_images_dir, f"{i:06d}.jpeg")
            if not os.path.exists(img_path):
                img_path = os.path.join(hdr_images_dir, f"{i:06d}.png")

            if not os.path.exists(img_path):
                img_path = os.path.join(hdr_images_dir, f"{i}.png")

            if not os.path.exists(img_path):
                img_path = os.path.join(hdr_images_dir, f"{i}.jpeg")

            if not os.path.exists(img_path):
                print(f"\nWarning: Image {i:06d} not found.")
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            if getattr(self.cli_args, 'remove_reverse', 'true').lower() == 'true':
                if self.odom_twist_lin[odom_idx][0] < -0.05:
                    continue

            orig_shape = img.shape[:2]
            out_shape = orig_shape

            if hasattr(self.cli_args, 'width') and hasattr(self.cli_args, 'height'):
                img = cv2.resize(img, (int(self.cli_args.width), int(self.cli_args.height)))
                out_shape = img.shape[:2]

            depth_img = None
            if self.params["save_depth"] and depth_images_dir is not None:
                depth_idx = self.get_closest_idx(self.depth_times, t_img)
                if depth_idx is not None:
                    dimg_path = os.path.join(depth_images_dir, f"{depth_idx:06d}.png")
                    if not os.path.exists(dimg_path):
                        dimg_path = os.path.join(depth_images_dir, f"{depth_idx}.png")
                    if os.path.exists(dimg_path):
                        source_depth = cv2.imread(dimg_path, cv2.IMREAD_UNCHANGED)
                        if source_depth is not None:
                            if source_depth.dtype != np.uint16 or source_depth.ndim != 2:
                                raise ValueError(
                                    f"Expected 16-bit millimeter depth PNG, got {source_depth.dtype} "
                                    f"with shape {source_depth.shape}: {dimg_path}"
                                )
                            # project at saved resolution, a later resize would drop readings
                            depth_m = reproject_depth_to_rgb(
                                source_depth.astype(np.float32) / 1000.0,
                                self.depth_K,
                                self.depth_D,
                                self.depth_model,
                                scale_intrinsics(self.rgb_K, orig_shape, out_shape),
                                self.rgb_D,
                                self.rgb_model,
                                self.rgb_from_depth,
                                out_shape,
                            )
                            depth_img = depth_to_uint16_mm(depth_m)

            if self.params["save_trajectory_mask"]:
                mask = self.generate_trajectory_mask(orig_shape, t_img, odom_idx)
                if hasattr(self.cli_args, 'width') and hasattr(self.cli_args, 'height'):
                    mask = cv2.resize(mask, (int(self.cli_args.width), int(self.cli_args.height)), interpolation=cv2.INTER_NEAREST)
            else:
                mask = None

            print(f"\rProcessed frame {i}/{total_images} (Saved: {count_saved})", end="")

            self.save_data(img, mask, count_saved, odom_idx, t_img, depth_img)
            self.last_saved_pos = p_curr
            count_saved += 1

        print("\nFinished processing dataset.")

    def save_data(self, img, mask, count, odom_idx, t_img, depth_img) -> None:
        """
        Write one frame's images and append its data.csv row.

        :param img: np.ndarray - BGR image
        :param mask: np.ndarray | None - trajectory mask, if enabled
        :param count: int - frame index used in file names
        :param odom_idx: int - odometry index for velocity and position
        :param t_img: float - image timestamp in seconds
        :param depth_img: np.ndarray | None - depth image
        """
        rgb_filename = f"rgb_{self.bag_name}_{count:06d}.jpg"
        rgb_path = os.path.join("rgb", rgb_filename)
        cv2.imwrite(os.path.join(self.params["output_dir"], rgb_path), img)

        if self.params["save_depth"] and depth_img is not None:
            depth_filename = f"depth_{self.bag_name}_{count:06d}.tif"
            depth_path = os.path.join("depth", depth_filename)
            cv2.imwrite(os.path.join(self.params["output_dir"], depth_path), depth_img)
        else:
            depth_path = ""

        if self.params["save_trajectory_mask"] and mask is not None:
            traj_filename = f"trajectory_{self.bag_name}_{count:06d}.jpg"
            traj_path = os.path.join("trajectory_masks", traj_filename)
            cv2.imwrite(os.path.join(self.params["output_dir"], traj_path), mask)
        else:
            traj_path = ""

        linear_vel = self.odom_twist_lin[odom_idx][0]
        if self.params["invert_velocity"]:
            linear_vel = -linear_vel

        angular_vel = self.odom_twist_ang[odom_idx][2]
        pos_x = self.odom_pos[odom_idx][0]
        pos_y = self.odom_pos[odom_idx][1]

        row = [rgb_path]
        if self.params["save_depth"]:
             row.append(depth_path)
        if self.params["save_trajectory_mask"]:
             row.append(traj_path)

        row.extend([linear_vel, angular_vel, pos_x, pos_y, t_img])

        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def main() -> None:
    """Process the command-line GrandTour mission and write the completion marker."""
    parser = argparse.ArgumentParser(description="Process Grand Tour Dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to grand tour dataset directory.")
    parser.add_argument("--width", type=int, default=640, help="Output width in pixels (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Output height in pixels (default: 480)")
    parser.add_argument("--remove_reverse", type=str, default="true", help="Remove reverse motion frames")
    parser.add_argument("--traj_shape", type=str, default="polygon", help="Trajectory shape: point or polygon")
    parser.add_argument("--robot_width", type=float, default=None, help="Override robot width for trajectory masks")
    args = parser.parse_args()

    processor = GrandTourProcessor(config_path=args.config, dataset_dir=args.dataset, cli_args=args)
    processor.process_dataset()
    with open(os.path.join(processor.params["output_dir"], "processing_completed.txt"), "w") as f: f.write("done")

if __name__ == "__main__":
    main()
