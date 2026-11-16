from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from data_generation.configuration import resolve_config

import cv2
import numpy as np

@dataclass(frozen=True)
class EgoMask:
    """
    Camera region ignored for supervision, such as the vehicle body.

    :ivar mask: np.ndarray | None - (H, W) bool, True on ignored pixels, None if not configured
    :ivar source: Path | None - resolved path of the mask image
    """
    mask: np.ndarray | None = None
    source: Path | None = None

    @property
    def description(self) -> str:
        """
        Describe the ego mask for logging.

        :return: str - source file name and ignored fraction, or a not-configured message
        """
        if self.mask is None:
            return "Ego mask: not configured"
        return f"Ego mask: {self.source.name} ({self.mask.mean():.1%} ignored)"

    def exclude(self, array) -> np.ndarray:
        """
        Return a copy of labels with ego pixels zeroed; raises ValueError if the last two axes differ from the mask shape.

        :param array: np.ndarray - labels whose last two axes are (H, W)
        :return: np.ndarray - copy with ego pixels set to 0
        """
        result = array.copy()
        if self.mask is not None:
            if result.shape[-2:] != self.mask.shape:
                raise ValueError(f"Ego mask shape {self.mask.shape} does not match labels {result.shape[-2:]}")
            result[..., self.mask] = 0
        return result

    def overlay(self, image_bgr) -> np.ndarray:
        """
        Return a copy with ego pixels tinted; raises ValueError if the size differs from the mask shape.

        :param image_bgr: np.ndarray - (H, W, 3) BGR uint8 image
        :return: np.ndarray - copy with ego pixels blended 35% image, 65% (255, 180, 0)
        """
        result = image_bgr.copy()
        if self.mask is not None:
            if result.shape[:2] != self.mask.shape:
                raise ValueError("Ego mask and review image dimensions differ")
            result[self.mask] = (
                result[self.mask].astype(np.float32) * 0.35
                + np.array([255, 180, 0], dtype=np.float32) * 0.65
            ).astype(np.uint8)
        return result

def _read_mask(source, image_shape) -> EgoMask:
    """
    Read a binary ego mask; raises ValueError if it is unreadable, not strictly 0/255, or differs in shape from the image.

    :param source: Path - grayscale mask image, 255 = ignored
    :param image_shape: tuple[int, ...] | None - image shape to match, None skips the check
    :return: EgoMask - mask with the resolved source path
    """
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read configured ego mask: {source}")
    if not np.isin(image, [0, 255]).all():
        raise ValueError(f"Ego mask must be binary 0/255: {source}")
    if image_shape is not None and tuple(image_shape[:2]) != image.shape:
        raise ValueError(
            f"Ego mask {source} has shape {image.shape}, but images have shape {tuple(image_shape[:2])}. "
            "Use the mask for this camera framing; do not silently stretch it."
        )
    return EgoMask(image == 255, source.resolve())

def load_ego_mask(dataset_dir, image_shape=None, config=None) -> EgoMask:
    """
    Load the ego mask referenced by the dataset manifest; raises ValueError if the manifest or mask is invalid.

    :param dataset_dir: str | Path - dataset directory that may contain the manifest
    :param image_shape: tuple[int, ...] | None - expected image shape, None skips the check
    :param config: dict | None - generation config, None loads the default
    :return: EgoMask - loaded mask, empty if the manifest is absent
    """
    root = Path(dataset_dir)
    config = resolve_config(config)
    manifest = root / config["data_format"]["ego_manifest_name"]
    if not manifest.exists():
        return EgoMask()
    try:
        metadata = json.loads(manifest.read_text())
        source = root / metadata["path"]
        expected = (metadata["height"], metadata["width"])
        ego = _read_mask(source, expected)
        if image_shape is not None and ego.mask.shape != tuple(image_shape[:2]):
            raise ValueError(f"Recorded ego mask geometry {ego.mask.shape} differs from image {tuple(image_shape[:2])}")
        return ego
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid ego-mask reference {manifest}: {exc}") from exc

def prepare_ego_mask(dataset_dir, config, image_shape) -> EgoMask:
    """
    Copy the family's configured ego mask into the dataset and write its manifest.

    The family is parsed from a `dataset_<family>_...` directory name; an existing differing `ego_mask.png` is never overwritten.

    :param dataset_dir: str | Path - dataset directory to receive the mask and manifest
    :param config: dict - generation config, `ego_masks` maps dataset families to mask paths
    :param image_shape: tuple[int, ...] - image shape the mask must match
    :return: EgoMask - configured mask, or the existing manifest mask if the family has no path
    """
    root = Path(dataset_dir)
    tokens = root.name.split("_")
    family = tokens[1] if len(tokens) > 2 and tokens[0] == "dataset" else None
    masks = config.get("ego_masks") or {}
    if not isinstance(masks, dict):
        raise ValueError("ego_masks must map dataset families to mask paths")
    source = masks.get(family)
    if not source:
        return load_ego_mask(root, image_shape, config)
    ego = _read_mask(Path(source), image_shape)
    local_asset = root / "ego_mask.png"
    if local_asset.exists():
        if local_asset.read_bytes() != ego.source.read_bytes():
            raise ValueError(f"Refusing to overwrite differing ego-mask asset {local_asset}")
    else:
        shutil.copyfile(ego.source, local_asset)
    metadata = {"path": local_asset.name,
                "width": int(ego.mask.shape[1]), "height": int(ego.mask.shape[0])}
    (root / config["data_format"]["ego_manifest_name"]).write_text(json.dumps(metadata, indent=2) + "\n")
    return EgoMask(ego.mask, local_asset.resolve())

def review_overlay(image_bgr, positive_mask, ego) -> np.ndarray:
    """
    Draw positives in green and tint ego pixels.

    :param image_bgr: np.ndarray - (H, W, 3) BGR uint8 image
    :param positive_mask: np.ndarray - positive pixels, ego pixels removed
    :param ego: EgoMask - ego mask to exclude and tint
    :return: np.ndarray - review image
    """
    positive = ego.exclude(np.asarray(positive_mask, dtype=bool))
    overlay = image_bgr.copy()
    overlay[positive] = (0, 255, 0)
    rendered = cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0)
    return ego.overlay(rendered)

def render_review_panel(image_bgr, panel_lines) -> np.ndarray:
    """
    Append a word-wrapped text panel to the right of an image.

    :param image_bgr: np.ndarray - (H, W, 3) BGR uint8 image
    :param panel_lines: list[str] - text lines, empty lines stay blank
    :return: np.ndarray - uint8 canvas, taller than the image if the text needs room
    """
    image_height, image_width = image_bgr.shape[:2]
    panel_width = max(250, min(360, int(image_width * 0.45)))
    font_scale = max(0.3, min(0.65, image_height / 800))
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_padding = max(8, int(image_height * 0.05))
    max_text_width = panel_width - 32
    wrapped_lines = []
    for line in panel_lines:
        words = line.split()
        if not words:
            wrapped_lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if cv2.getTextSize(candidate, font, font_scale, 1)[0][0] <= max_text_width:
                current_line = candidate
            else:
                wrapped_lines.append(current_line)
                current_line = word
        wrapped_lines.append(current_line)
    text_height = cv2.getTextSize("Ag", font, font_scale, 1)[0][1]
    line_gap = max(16, int(text_height * 1.6))
    display_height = max(image_height, top_padding * 2 + (len(wrapped_lines) + 1) * line_gap)
    display = np.full((display_height, image_width + panel_width, 3), (32, 32, 32), dtype=np.uint8)
    display[:image_height, :image_width] = image_bgr
    for line_idx, line in enumerate(wrapped_lines):
        if line:
            y = top_padding + (line_idx + 1) * line_gap
            cv2.putText(display, line, (image_width + 16, y), font, font_scale,
                        (240, 240, 240), 1, cv2.LINE_AA)
    return display
