import pandas as pd
import numpy as np
import cv2
import torch

from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from cat.configuration import resolve_config
from cat.data.depth import cover_crop_geometry, prepare_depth_input

class TraversabilityDataset(Dataset):
    """Image and depth dataset feeding embedding generation."""
    def __init__(
        self,
        folder_path,
        csv_path,
        resolution,
        in_memory=False,
        use_depth=True,
        max_frames=None,
        config=None,
    ) -> None:
        """
        Read the label CSV, index file paths and compute per-sensor depth statistics.

        :param folder_path: str | Path - image folder
        :param csv_path: str | Path - label csv
        :param resolution: tuple[int, int] - (H, W)
        :param in_memory: bool - cache decoded samples
        :param use_depth: bool - include depth images
        :param max_frames: int | None - keep only the first max_frames CSV rows
        :param config: dict | None - CAT configuration, defaults loaded when None
        """
        super().__init__()

        config = resolve_config(config)
        # per-sensor (min, max) clip in m; raw depth is uint16 mm
        self.depth_clip_by_sensor = {
            sensor: tuple(clip) for sensor, clip in config["data"]["depth_clip_by_sensor"].items()
        }
        # below this valid fraction at the training resolution, depth counts as missing
        self.depth_min_valid_fraction = float(config["data"]["depth_min_valid_fraction"])

        self._samples = None
        self._folder_path = Path(folder_path)
        self._csv = pd.read_csv(csv_path)
        if max_frames is not None:
            self._csv = self._csv.iloc[:max_frames].reset_index(drop=True)
        self._resolution = resolution # (h, w)
        # imagenet mean/std
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        self._use_depth = use_depth

        self._prepare_data()

        if self._use_depth:
            self._get_depth_stats()

        if in_memory:
            self._samples = [self._read_item(i) for i in range(self._csv.shape[0])]

    def _prepare_data(self) -> None:
        """Build the per-row path and sensor lists from the CSV."""
        self._rgb_fname_list = []
        self._depth_fname_list = []
        self._artificial_mask_fname_list = []
        self._traj_fname_list = []
        self._score_fname_list = []
        self._sensor_list = []

        for row in self._csv.itertuples(index=False):
            self._rgb_fname_list.append(self._folder_path / str(row.rgb_img_name))
            self._artificial_mask_fname_list.append(self._folder_path / str(row.positive_mask_name))
            self._traj_fname_list.append(self._folder_path / str(row.trajectory_img_name))
            self._score_fname_list.append(self._folder_path / str(row.scores_yaml))
            # no second token -> empty sensor, rejected below when depth is used
            source_parts = str(row.source_bag).split('_')
            self._sensor_list.append(source_parts[1] if len(source_parts) > 1 else '')

            # missing depth gets a nonexistent path, handled in _read_item
            depth_name = str(row.depth_img_name) if pd.notna(row.depth_img_name) else "missing_depth"
            self._depth_fname_list.append(self._folder_path / depth_name)

        if self._use_depth:
            for source_bag, sensor in dict(zip(self._csv.source_bag, self._sensor_list)).items():
                if sensor not in self.depth_clip_by_sensor:
                    raise ValueError(
                        f"Sequence '{source_bag}' has sensor token '{sensor}' (from source_bag.split('_')[1]), "
                        f"which is not in data.depth_clip_by_sensor {sorted(self.depth_clip_by_sensor)}. "
                        f"Name the sequence dataset_<sensor>_<name> and add that sensor's depth clip "
                        f"to cat/configs/train.yaml."
                    )

    def _get_depth_stats(self) -> None:
        """Compute per-sensor depth mean and std in meters over valid (finite, > 0) pixels; raise ValueError for a sensor with none."""
        psum = {}
        psum_sq = {}
        count = {}

        print("Calculating per-sensor depth statistics...")
        for idx, depth_fname in enumerate(tqdm(self._depth_fname_list)):
            # missing file = no measurement
            if not depth_fname.exists():
                continue

            sensor = self._sensor_list[idx]
            depth_img = cv2.imread(str(depth_fname), -1)
            if depth_img is None:
                raise ValueError(f"Depth file exists but cannot be read: {depth_fname}")

            clip_min, clip_max = self.depth_clip_by_sensor[sensor]
            if sensor not in count:
                psum[sensor] = 0.0
                psum_sq[sensor] = 0.0
                count[sensor] = 0

            depth_img = depth_img.astype(np.float64) * 1e-3
            # raw 0 = sensor no-return
            valid = np.isfinite(depth_img) & (depth_img > 0)
            depth_img = np.clip(depth_img, clip_min, clip_max)
            valid_vals = depth_img[valid]

            psum[sensor] += np.sum(valid_vals)
            psum_sq[sensor] += np.sum(valid_vals**2)
            count[sensor] += valid_vals.size

        self._depth_mean = {}
        self._depth_std = {}
        for sensor, sensor_count in count.items():
            if sensor_count == 0:
                raise ValueError(f"Depth files exist for sensor '{sensor}', but contain no valid depth pixels")

            mean = psum[sensor] / sensor_count
            std = np.sqrt(psum_sq[sensor] / sensor_count - (mean**2))
            self._depth_mean[sensor] = mean
            self._depth_std[sensor] = std

    def _resize(self, image, size, interpolation=cv2.INTER_AREA) -> np.ndarray:
        """
        Resize with aspect ratio preserved and center crop.

        :param image: np.ndarray - (H, W, C) or (H, W) image
        :param size: tuple[int, int] - target (H, W)
        :param interpolation: int - cv2 interpolation method
        :return: np.ndarray - cropped image
        """
        target_h, target_w = size
        new_h, new_w, start_y, start_x = cover_crop_geometry(image.shape[0], image.shape[1], size)
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return resized[start_y:start_y + target_h, start_x:start_x + target_w]

    def _read_item(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read the RGB image and normalized depth of one row; unavailable depth is zeros.

        :param idx: int - row index
        :return: tuple[torch.Tensor, torch.Tensor] - RGB [3, H, W] in [0, 1] and normalized depth [1, H, W]
        """

        rgb_file = self._rgb_fname_list[idx]
        depth_file = self._depth_fname_list[idx]

        rgb_image = cv2.imread(str(rgb_file))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = self._resize(rgb_image, self._resolution, interpolation=cv2.INTER_AREA)
        rgb_image = transforms.ToTensor()(rgb_image)

        depth_image = None
        if self._use_depth and depth_file.exists():
            raw_depth = cv2.imread(str(depth_file), -1)
            if raw_depth is not None:
                sensor = self._sensor_list[idx]
                depth_image = prepare_depth_input(
                    raw_depth, self._resolution, self.depth_clip_by_sensor[sensor],
                    self._depth_mean[sensor], self._depth_std[sensor], self.depth_min_valid_fraction,
                )
        if depth_image is None:
            depth_image = np.zeros(self._resolution, dtype=np.float32)
        depth_image = torch.from_numpy(depth_image).unsqueeze(0)  # [1, h, w]

        return rgb_image, depth_image

    def __len__(self) -> int:
        """
        Get the row count.

        :return: int - number of rows
        """
        return self._csv.shape[0]

    def __getitem__(self, idx) -> dict:
        """
        Get one row with RGB normalized by ImageNet statistics.

        :param idx: int - row index
        :return: dict - {'data': (rgb [3, H, W], depth [1, H, W])}
        """
        if self._samples is None:
            rgb_image, depth_image = self._read_item(idx)
        else:
            rgb_image, depth_image = self._samples[idx]

        return {'data': (self._normalize(rgb_image), depth_image)}
