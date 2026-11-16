import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cat.configuration import resolve_config

def _fail(manifest, reason) -> None:
    """
    Raise a ValueError for an invalid ego-mask reference.

    :param manifest: Path - ego manifest
    :param reason: str - failure reason
    """
    raise ValueError(f"Invalid ego-mask reference {manifest}: {reason}")

def load_ego_asset(dataset_root, rgb_shape=None, config=None) -> np.ndarray | None:
    """
    Read a dataset root's ego mask at native resolution; raise ValueError if the manifest or asset is invalid.

    :param dataset_root: str | Path - directory that may contain the ego manifest
    :param rgb_shape: tuple | None - (H, W[, C]) of the RGB frames to validate against
    :param config: dict | None - CAT configuration, defaults loaded when None
    :return: np.ndarray | None - [H, W] bool, True = own robot (ignored); None if no manifest exists
    """
    root = Path(dataset_root)
    manifest = root / resolve_config(config)["data"]["ego_manifest_name"]
    if not manifest.exists():
        return None

    try:
        metadata = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        _fail(manifest, f"cannot parse JSON ({exc})")
    if not isinstance(metadata, dict):
        _fail(manifest, "expected a JSON object")
    for key in ("path", "width", "height"):
        if key not in metadata:
            _fail(manifest, f"missing key '{key}'")
    path, width, height = metadata["path"], metadata["width"], metadata["height"]
    if not isinstance(path, str) or not path:
        _fail(manifest, "'path' must be a non-empty string")
    for key, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(manifest, f"'{key}' must be a positive integer")

    source = root / path
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        _fail(manifest, f"cannot read mask asset {source}")
    if not np.isin(image, (0, 255)).all():
        _fail(manifest, f"mask asset must be binary 0/255: {source}")
    if image.shape != (height, width):
        _fail(manifest, f"mask asset {source} has shape {image.shape}, manifest declares {(height, width)}")
    if rgb_shape is not None and tuple(rgb_shape[:2]) != image.shape:
        _fail(manifest, f"mask shape {image.shape} differs from RGB frame shape {tuple(rgb_shape[:2])}")
    return image == 255

class ValidityMasks:
    """Per-frame validity masks (True = supervised, False = ego) at training resolution, one per dataset root."""
    def __init__(self, rgb_files, size, resize, config=None) -> None:
        """
        Load the ego mask of each distinct dataset root (rgb_path.parent.parent).

        :param rgb_files: list[Path] - per-frame RGB image paths
        :param size: tuple[int, int] - training (H, W)
        :param resize: Callable[..., np.ndarray] - resize(image, size, interpolation), same as the image and label masks
        :param config: dict | None - CAT configuration, defaults loaded when None
        """
        config = resolve_config(config)
        self._config = config
        self._manifest_name = config["data"]["ego_manifest_name"]
        self._size = tuple(size)
        self._valid = []
        self._frame_slot = np.full(len(rgb_files), -1, dtype=np.int32)  # -1 = no ego mask

        slot_by_root = {}
        for i, rgb_file in enumerate(rgb_files):
            root = Path(rgb_file).parent.parent
            if root not in slot_by_root:
                slot_by_root[root] = self._load_root(root, rgb_file, resize)
            self._frame_slot[i] = slot_by_root[root]

    def _load_root(self, root, first_rgb, resize) -> int:
        """
        Load, resize and invert one root's ego mask; raise ValueError if the first RGB frame is unreadable.

        :param root: Path - dataset root directory
        :param first_rgb: Path - root's first RGB frame, validates the mask shape
        :param resize: Callable[..., np.ndarray] - resize(image, size, interpolation)
        :return: int - index into the shared validity list, -1 if the root has no ego manifest
        """
        if not (root / self._manifest_name).exists():
            return -1
        rgb = cv2.imread(str(first_rgb), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise ValueError(f"Cannot read RGB frame {first_rgb} to validate ego mask of {root}")
        ego = load_ego_asset(root, rgb.shape, self._config)
        # matches image pipeline: nearest resize + center crop
        ego_resized = resize(ego.astype(np.uint8) * 255, self._size, interpolation=cv2.INTER_NEAREST) > 127
        self._valid.append(~ego_resized)
        return len(self._valid) - 1

    @property
    def num_masked_frames(self) -> int:
        """
        Count frames with an ego mask.

        :return: int - frame count
        """
        return int((self._frame_slot >= 0).sum())

    def get(self, idx) -> torch.Tensor:
        """
        Get the validity mask of one frame.

        :param idx: int - frame index
        :return: torch.Tensor - [1, H, W] bool, all True when the frame has no ego mask
        """
        slot = self._frame_slot[idx]
        if slot < 0:
            return torch.ones((1, *self._size), dtype=torch.bool)
        return torch.from_numpy(self._valid[slot]).unsqueeze(0).clone()
