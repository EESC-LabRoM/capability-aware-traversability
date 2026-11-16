import os
import sys
import warnings
import csv
import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from data_generation.configuration import load_config, resolve_config
from data_generation.utils.common_utils import require_empty_output
from data_generation.dataset_creation.bag_io import depth_to_uint16_mm
from data_generation.dataset_creation.geometry import project_points_to_depth, scale_intrinsics

def _pose(translation, quaternion_xyzw) -> np.ndarray:
    """
    Build a homogeneous pose from translation and xyzw quaternion.

    :param translation: list[float] - child origin in the parent frame (m)
    :param quaternion_xyzw: list[float] - child orientation in the parent frame
    :return: np.ndarray - (4, 4) matrix mapping child to parent coordinates
    """
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(quaternion_xyzw).as_matrix()
    pose[:3, 3] = translation
    return pose

def _load_timestamps(path, label, problems) -> np.ndarray | None:
    """
    Load a timestamps.txt file, recording problems.

    :param path: str - timestamps file, one timestamp in seconds per line
    :param label: str - name used in problem messages
    :param problems: list[str] - receives a message for each problem found
    :return: np.ndarray | None - 1-D sorted timestamps in seconds, None if the file is invalid
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # loadtxt warns on empty files, handled as a problem below
            times = np.loadtxt(path)
    except (ValueError, OSError) as exc:
        problems.append(f"{label} is unreadable ({exc}).")
        return None
    if times.size == 0:
        problems.append(f"{label} is empty.")
        return None
    if times.ndim != 1 or times.size < 2:
        problems.append(f"{label} must contain one timestamp per line and at least 2 lines.")
        return None
    if not np.all(np.isfinite(times)):
        problems.append(f"{label} contains NaN or infinite values.")
        return None
    if np.any(np.diff(times) < 0):
        problems.append(f"{label} is not sorted in increasing order.")
        return None
    return times

def validate_tartan_input(dataset_dir, save_depth=True, config=None) -> None:
    """
    Reject missing, corrupt or mismatched raw input before creating output.

    :param dataset_dir: str - raw TartanDrive trajectory containing gps_odom/ and image_left_color/
    :param save_depth: bool - whether depth is rendered from the full_cloud LiDAR stream
    :param config: dict | None - complete generation config, None = default
    :raises ValueError: Listing every problem found and how to fix it
    """
    config = resolve_config(config)
    validation = config["validation"]
    image_extensions = tuple(validation["image_extensions"])
    tolerance = validation["odom_time_tolerance"]
    remediation = validation["tartan_remediation"]
    dataset_dir = os.path.abspath(dataset_dir)
    odom_dir = os.path.join(dataset_dir, "gps_odom")
    rgb_dir = os.path.join(dataset_dir, "image_left_color")
    lidar_dir = os.path.join(dataset_dir, "full_cloud")

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"TartanDrive dataset directory not found: {dataset_dir}." + remediation)

    if not os.path.isdir(rgb_dir) and (
        os.path.exists(os.path.join(dataset_dir, "data.csv")) or os.path.isdir(os.path.join(dataset_dir, "rgb"))
    ):
        raise ValueError(
            f"{dataset_dir} looks like an already processed dataset (data.csv/rgb found, but no raw "
            "image_left_color directory). Select a raw TartanDrive trajectory with gps_odom/ and "
            "image_left_color/ instead; processed datasets can be used from Manage Local Datasets "
            "or the segmentation menus."
        )

    problems = []
    odom_ts_path = os.path.join(odom_dir, "timestamps.txt")
    odom_npy_path = os.path.join(odom_dir, "odometry.npy")
    rgb_ts_path = os.path.join(rgb_dir, "timestamps.txt")

    for path in (odom_ts_path, odom_npy_path, rgb_ts_path):
        if not os.path.isfile(path):
            problems.append(f"Missing required file: {os.path.relpath(path, dataset_dir)}.")
    if not os.path.isdir(rgb_dir):
        rgb_files = []
        problems.append("Missing required directory: image_left_color.")
    else:
        rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(image_extensions))
        if not rgb_files:
            problems.append("image_left_color contains no .png/.jpg/.jpeg images.")

    odom_times = _load_timestamps(odom_ts_path, "gps_odom/timestamps.txt", problems) if os.path.isfile(odom_ts_path) else None
    rgb_times = _load_timestamps(rgb_ts_path, "image_left_color/timestamps.txt", problems) if os.path.isfile(rgb_ts_path) else None

    if os.path.isfile(odom_npy_path):
        try:
            odom_data = np.load(odom_npy_path, mmap_mode="r")
        except (ValueError, OSError, EOFError) as exc:
            problems.append(f"gps_odom/odometry.npy is unreadable ({exc}).")
        else:
            if odom_data.ndim != 2 or odom_data.shape[1] < 13:
                problems.append(
                    f"gps_odom/odometry.npy must be a 2-D array with at least 13 columns "
                    f"(position, orientation, linear and angular twist); found shape {odom_data.shape}."
                )
            elif not np.issubdtype(odom_data.dtype, np.number):
                problems.append(f"gps_odom/odometry.npy must be numeric; found dtype {odom_data.dtype}.")
            elif odom_times is not None and len(odom_times) != odom_data.shape[0]:
                problems.append(
                    f"gps_odom/timestamps.txt has {len(odom_times)} entries but odometry.npy has "
                    f"{odom_data.shape[0]} rows."
                )

    if rgb_times is not None and rgb_files and len(rgb_times) != len(rgb_files):
        problems.append(
            f"image_left_color/timestamps.txt has {len(rgb_times)} entries but the directory has "
            f"{len(rgb_files)} images."
        )

    if odom_times is not None and rgb_times is not None:
        idx = np.clip(np.searchsorted(odom_times, rgb_times), 1, len(odom_times) - 1)
        nearest = np.minimum(np.abs(odom_times[idx - 1] - rgb_times), np.abs(odom_times[idx] - rgb_times))
        if not np.any(nearest <= tolerance):
            problems.append(
                f"No RGB timestamp is within {tolerance} s of an odometry timestamp "
                "(the image and odometry clocks do not overlap), so no frames could be produced."
            )

    for name in dict.fromkeys((rgb_files[:1] + rgb_files[-1:])):
        if cv2.imread(os.path.join(rgb_dir, name)) is None:
            problems.append(f"image_left_color/{name} cannot be read as an image.")

    if save_depth:
        lidar_times_path = os.path.join(lidar_dir, "timestamps.txt")
        if not os.path.isdir(lidar_dir) or not os.path.isfile(lidar_times_path):
            problems.append(
                "Missing full_cloud/ with timestamps.txt, the LiDAR source of TartanDrive depth. "
                "Download the trajectory again, or set bag_processing.save_depth: false."
            )
        else:
            lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".npy"))
            lidar_times = _load_timestamps(lidar_times_path, "full_cloud/timestamps.txt", problems)
            if lidar_times is not None and len(lidar_times) != len(lidar_files):
                problems.append(
                    f"full_cloud/timestamps.txt has {len(lidar_times)} entries but full_cloud has "
                    f"{len(lidar_files)} .npy files."
                )
            for name in lidar_files:
                try:
                    cloud = np.load(os.path.join(lidar_dir, name), mmap_mode="r")
                except (ValueError, OSError, EOFError) as exc:
                    problems.append(f"full_cloud/{name} is unreadable ({exc}).")
                    continue
                if cloud.ndim != 2 or cloud.shape[1] < 3 or not np.issubdtype(cloud.dtype, np.number):
                    problems.append(
                        f"full_cloud/{name} must be a numeric (N, 3+) array of x, y, z in the vehicle frame; "
                        f"found shape {cloud.shape}, dtype {cloud.dtype}."
                    )
        if rgb_files:
            rgb = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
            image_shape = tuple(config["tartan_calibration"]["image_shape"])
            if rgb is not None and rgb.shape[:2] != image_shape:
                problems.append(
                    f"image_left_color frames are {rgb.shape[:2]}, but the left-camera calibration used to "
                    f"project full_cloud is for {image_shape}."
                )
    if problems:
        raise ValueError(f"Invalid TartanDrive recording: {dataset_dir}. " + " ".join(problems) + remediation)

class TartanProcessor:
    def __init__(self, config_path="configs/config.yaml", dataset_dir=None, cli_args=None) -> None:
        """
        Load config and data, validate input, prepare the output directory.

        :param config_path: str - YAML config file
        :param dataset_dir: str | None - raw trajectory directory, None = bag_processing.dataset_dir
        :param cli_args: argparse.Namespace | None - optional config overrides
        :raises ValueError: If the input is invalid or the output directory is not empty
        """
        self.cli_args = cli_args
        self.config = self.load_config(config_path)
        self.dataset_dir = dataset_dir if dataset_dir else self.config.get("bag_processing", {}).get("dataset_dir", "bags/warehouse_to_fence")

        self.dataset_dir = os.path.abspath(self.dataset_dir)
        self.bag_name = os.path.basename(self.dataset_dir)

        self.setup_parameters()
        validate_tartan_input(self.dataset_dir, save_depth=self.params["save_depth"], config=self.config)
        require_empty_output(self.params["output_dir"])
        self.setup_directories()

        self.csv_path = os.path.join(self.params["output_dir"], "data.csv")
        self.init_csv()
        self.load_data()

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
            "image_extensions": tuple(self.config["validation"]["image_extensions"]),
            "odom_time_tolerance": self.config["validation"]["odom_time_tolerance"],
        }

        self.params["output_dir"] = os.path.join(self.params["output_dir"], "dataset_tartan_" + self.bag_name)

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
        """Load odometry, image and LiDAR metadata; set the camera model and extrinsics."""
        print(f"Loading data from {self.dataset_dir}...")
        odom_path = os.path.join(self.dataset_dir, "gps_odom")
        rgb_path = os.path.join(self.dataset_dir, "image_left_color")
        lidar_path = os.path.join(self.dataset_dir, "full_cloud")

        self.odom_times = np.loadtxt(os.path.join(odom_path, "timestamps.txt"))
        self.odom_data = np.load(os.path.join(odom_path, "odometry.npy"))

        self.odom_pos = self.odom_data[:, 0:3]
        self.odom_orien = self.odom_data[:, 3:7]
        self.odom_twist_lin = self.odom_data[:, 7:10]
        self.odom_twist_ang = self.odom_data[:, 10:13]

        self.rgb_times = np.loadtxt(os.path.join(rgb_path, "timestamps.txt"))
        self.rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(self.params["image_extensions"])])

        if self.params["save_depth"]:
            self.lidar_times = np.loadtxt(os.path.join(lidar_path, "timestamps.txt"))
            self.lidar_files = sorted(f for f in os.listdir(lidar_path) if f.endswith(".npy"))
        else:
            self.lidar_times = None
            self.lidar_files = []

        calibration = self.config["tartan_calibration"]
        self.image_shape = tuple(calibration["image_shape"])
        self.camera_matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        vehicle_from_head = _pose(calibration["vehicle_from_head"]["translation_m"], calibration["vehicle_from_head"]["quaternion_xyzw"])
        head_from_left_optical = _pose(calibration["head_from_left_optical"]["translation_m"], calibration["head_from_left_optical"]["quaternion_xyzw"])
        self.left_optical_from_vehicle = np.linalg.inv(vehicle_from_head @ head_from_left_optical)
        self.dist_coeffs = np.zeros(5)

        self.trans_cam_base_mat = np.eye(4)
        self.trans_cam_base_mat[:3, :3] = np.array([
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
            [ 1.0,  0.0,  0.0]
        ])
        self.trans_cam_base_mat[:3, 3] = [0.0, 1.0, 0.0]

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

    def render_lidar_depth(self, cloud_path, output_shape) -> np.ndarray:
        """
        Project a full_cloud scan (vehicle frame) into the left camera at the saved image size.

        :param cloud_path: str - full_cloud .npy array whose first three columns are x, y, z (m)
        :param output_shape: tuple[int, int] - (H, W) of the saved RGB image
        :return: np.ndarray - depth in metres registered to the RGB image, 0 where no return lands
        """
        try:
            cloud = np.load(cloud_path)
        except (ValueError, OSError, EOFError) as e:
            raise ValueError(f"cannot read TartanDrive LiDAR scan {cloud_path}: {e}") from e
        # project at output size, resizing afterwards would drop returns
        points = np.asarray(cloud[:, :3], dtype=np.float64)
        points_cam = points @ self.left_optical_from_vehicle[:3, :3].T + self.left_optical_from_vehicle[:3, 3]
        K = scale_intrinsics(self.camera_matrix, self.image_shape, output_shape)
        return project_points_to_depth(points_cam, K, output_shape)

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

    def _get_mat_from_odom(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the pose of an odometry sample.

        :param idx: int - odometry sample index
        :return: tuple[np.ndarray, np.ndarray] - 4x4 pose matrix and the position in meters
        """
        p = self.odom_pos[idx]
        q = self.odom_orien[idx]

        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(q).as_matrix()
        mat[:3, 3] = p
        return mat, p

    def generate_trajectory_mask(self, img_shape, current_t, current_odom_idx) -> tuple[np.ndarray, bool]:
        """
        Project the odometry path around the frame time into the camera image as a mask.

        :param img_shape: tuple[int, int] - mask (H, W)
        :param current_t: float - frame time in seconds
        :param current_odom_idx: int - odometry index of the current frame
        :return: tuple[np.ndarray, bool] - (H, W) uint8 mask with path pixels 255, and a status that is always True
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        horizon = self.params["horizon_sec"]
        offset = self.params["start_offset"]

        valid_indices = np.where((self.odom_times >= current_t - offset) & (self.odom_times <= current_t + horizon))[0]

        if len(valid_indices) == 0:
            return mask, True

        valid_indices = np.insert(valid_indices, 0, current_odom_idx)

        mat_curr, _ = self._get_mat_from_odom(current_odom_idx)
        mat_base_odom = np.linalg.inv(mat_curr)

        num_offsets = 3
        offsets = np.linspace(-self.params["robot_width"] / 2, self.params["robot_width"] / 2, num_offsets)

        cam_ext = self.trans_cam_base_mat
        prev_left = None
        prev_right = None

        last_drawn_pos = np.array([-999.0, -999.0])

        for idx in valid_indices:
            mat_p, _ = self._get_mat_from_odom(idx)
            mat_p_base = mat_base_odom @ mat_p

            x_rel = mat_p_base[0, 3]
            y_rel = mat_p_base[1, 3]

            if x_rel < -0.1 or x_rel > 15.0:
                continue

            curr_pos = np.array([x_rel, y_rel])
            if np.linalg.norm(curr_pos - last_drawn_pos) < 0.20:
                continue

            last_drawn_pos = curr_pos

            local_pts = np.zeros((num_offsets, 4))
            local_pts[:, 1] = offsets
            local_pts[:, 2] = -0.60
            local_pts[:, 3] = 1.0

            points_base = (mat_p_base @ local_pts.T).T

            pts_cam = (cam_ext @ points_base.T).T

            pts_cam_filtered = pts_cam[:, :3].copy()
            valid_z = pts_cam_filtered[:, 2] > 0.2

            if not np.any(valid_z):
                prev_left = None
                prev_right = None
                continue

            pts_2d, _ = cv2.projectPoints(
                pts_cam_filtered.astype(np.float32),
                np.zeros(3), np.zeros(3),
                self.camera_matrix, self.dist_coeffs
            )

            pts_2d = pts_2d.reshape(-1, 2)

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

        return mask, True

    def process_dataset(self) -> None:
        """
        Convert selected frames to the output dataset and append them to data.csv.

        Skips frames without nearby odometry, closer than min_dist, reversing (if enabled) or unreadable.
        """
        count_saved = 0
        total_images = len(self.rgb_times)

        print(f"Total images to process: {total_images}")

        rgb_dir = os.path.join(self.dataset_dir, "image_left_color")
        lidar_dir = os.path.join(self.dataset_dir, "full_cloud")

        for i, t_img in enumerate(self.rgb_times):
            if self.params["max_images"] is not None and count_saved >= self.params["max_images"]:
                print(f"\nReached limit of {self.params['max_images']} images.")
                break

            odom_idx = self.get_closest_idx(self.odom_times, t_img)
            if odom_idx is None:
                continue

            p_curr = self.odom_pos[odom_idx]
            if not self.check_distance(p_curr):
                continue

            if i >= len(self.rgb_files):
                continue

            if getattr(self.cli_args, 'remove_reverse', 'true').lower() == 'true':
                x_vel = self.odom_twist_lin[odom_idx][0]
                if x_vel < -0.05:
                    continue

            img_path = os.path.join(rgb_dir, self.rgb_files[i])
            img = cv2.imread(img_path)
            if img is None:
                continue

            orig_shape = img.shape[:2]

            if hasattr(self.cli_args, 'width') and hasattr(self.cli_args, 'height'):
                img = cv2.resize(img, (int(self.cli_args.width), int(self.cli_args.height)))
            depth_img = None

            if self.params["save_depth"]:
                lidar_idx = self.get_closest_idx(self.lidar_times, t_img)
                if lidar_idx is not None:
                    depth_img = self.render_lidar_depth(os.path.join(lidar_dir, self.lidar_files[lidar_idx]), img.shape[:2])

            if self.params["save_trajectory_mask"]:
                mask, _ = self.generate_trajectory_mask(orig_shape, t_img, odom_idx)
                if mask is not None and hasattr(self.cli_args, 'width') and hasattr(self.cli_args, 'height'):
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
        :param depth_img: np.ndarray | None - LiDAR depth in metres registered to `img`, 0 = no return
        """
        rgb_filename = f"rgb_{self.bag_name}_{count:06d}.jpg"
        rgb_path = os.path.join("rgb", rgb_filename)
        cv2.imwrite(os.path.join(self.params["output_dir"], rgb_path), img)

        if self.params["save_depth"] and depth_img is not None:
            depth_filename = f"depth_{self.bag_name}_{count:06d}.tif"
            depth_path = os.path.join("depth", depth_filename)
            depth_uint16 = depth_to_uint16_mm(depth_img)
            cv2.imwrite(os.path.join(self.params["output_dir"], depth_path), depth_uint16)
        else:
            depth_path = ""

        if self.params["save_trajectory_mask"] and mask is not None:
            traj_filename = f"trajectory_{self.bag_name}_{count:06d}.jpg"
            traj_path = os.path.join("trajectory_masks", traj_filename)
            cv2.imwrite(os.path.join(self.params["output_dir"], traj_path), mask)
        else:
            traj_path = ""

        linear_vel = np.linalg.norm(self.odom_twist_lin[odom_idx][:2])
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
    """Process the command-line TartanDrive trajectory, exiting with status 1 on failure."""
    parser = argparse.ArgumentParser(description="Process Tartan Dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a single Tartan dataset directory.")
    parser.add_argument("--width", type=int, default=640, help="Resize image width")
    parser.add_argument("--height", type=int, default=480, help="Resize image height")
    parser.add_argument("--remove_reverse", type=str, default="true", help="Remove reverse motion frames")
    parser.add_argument("--traj_shape", type=str, default="polygon", help="Trajectory shape: point or polygon")
    parser.add_argument("--robot_width", type=float, default=None, help="Override robot width for trajectory masks")
    args = parser.parse_args()

    ds = args.dataset
    print("\n==========================================")
    print(f"Processing bag: {ds}")
    print("==========================================")
    try:
        processor = TartanProcessor(config_path=args.config, dataset_dir=ds, cli_args=args)
        processor.process_dataset()
        with open(os.path.join(processor.params["output_dir"], "processing_completed.txt"), "w") as f: f.write("done")
    except Exception as e:
        print(f"Failed to process {ds}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
