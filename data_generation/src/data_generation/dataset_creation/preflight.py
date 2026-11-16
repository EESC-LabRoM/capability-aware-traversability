import csv
from pathlib import Path

import cv2

from data_generation.dataset_creation.create_dataset_tartan import validate_tartan_input
from data_generation.dataset_creation.create_dataset_grand_tour import validate_grandtour_input
from data_generation.dataset_creation.create_dataset_ts import validate_bag_input
from data_generation.utils.ego_mask import load_ego_mask

def validate_raw_input(kind, path, config) -> None:
    """
    Validate raw input for the dataset type; raise ValueError if unsupported or invalid.

    :param kind: str - "Tartandrive", "Grandtour" or "WayFASTER"
    :param path: str | Path - raw recording or bag
    :param config: dict - generation configuration
    """
    if kind == "Tartandrive":
        validate_tartan_input(path, save_depth=config.get("bag_processing", {}).get("save_depth", True), config=config)
    elif kind == "Grandtour":
        validate_grandtour_input(
            path,
            save_depth=config.get("bag_processing", {}).get("save_depth", True),
        )
    elif kind == "WayFASTER":
        validate_bag_input(path, config)
    else:
        raise ValueError(f"Unsupported raw dataset type: {kind}")

def validate_processed_input(dataset_dir, require_masks=False, config=None) -> None:
    """
    Check data.csv and its referenced files are present and readable; raise ValueError otherwise.

    :param dataset_dir: str | Path - directory containing data.csv and the images
    :param require_masks: bool - also require readable positive masks matching the RGB size
    :param config: dict | None - generation configuration, None = default
    """
    root = Path(dataset_dir)
    csv_path = root / "data.csv"
    try:
        with csv_path.open(newline="") as source:
            reader = csv.DictReader(source)
            required = {"rgb_img_name"}
            if require_masks:
                required.add("positive_mask_name")
            missing_columns = required - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")
            rows = list(reader)
        if not rows:
            raise ValueError("data.csv has no frames")
        missing = []
        for index, row in enumerate(rows, start=2):
            columns = ["rgb_img_name", "depth_img_name", "trajectory_img_name"]
            if require_masks:
                columns.append("positive_mask_name")
            for column in columns:
                name = row.get(column)
                if not name:
                    if column in required:
                        missing.append(f"row {index}: empty {column}")
                    continue
                path = root / name
                if not path.is_file():
                    missing.append(f"row {index}: missing {name}")
        if missing:
            raise ValueError(f"{len(missing)} invalid file references; " + "; ".join(missing[:3]))
        first_rgb = root / rows[0]["rgb_img_name"]
        rgb = cv2.imread(str(first_rgb))
        if rgb is None:
            raise ValueError(f"Unreadable RGB image: {first_rgb}")
        load_ego_mask(root, rgb.shape[:2], config)
        if require_masks:
            first_mask = root / rows[0]["positive_mask_name"]
            mask = cv2.imread(str(first_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Unreadable positive mask: {first_mask}")
            if mask.shape != rgb.shape[:2]:
                raise ValueError("RGB and positive-mask dimensions differ")
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        raise ValueError(
            f"Incomplete or invalid processed dataset: {root}. {exc}. "
            "Restore missing files or download the session again. "
            "For SegFormer training, generate complete positive masks first."
        ) from exc
