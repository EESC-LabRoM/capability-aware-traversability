# This file originally included helpers adapted from Grounded-SAM-2
# (https://github.com/IDEA-Research/Grounded-SAM-2), utils/common_utils.py
# Copyright 2023 - present, IDEA Research.
# Licensed under the Apache License, Version 2.0, see data_generation/licenses/Grounded-SAM-2-LICENSE
# Modified for CAT. The remaining mask-processing functions were written for CAT.
import os
import cv2
import numpy as np
import csv
from pathlib import Path
import tempfile
from data_generation.configuration import resolve_config

def require_empty_output(dataset_dir) -> None:
    """
    Raise FileExistsError unless the output dataset directory is missing or empty.

    :param dataset_dir: str | Path - output dataset directory
    """
    root = Path(dataset_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(
            f"Output dataset already contains files: {root}. Raw processing cannot resume safely. "
            "Use a fresh bag_processing.output_dataset_dir or move the existing sequence aside."
        )

def update_positive_mask_references(dataset_dir, config=None, allow_partial=False) -> bool:
    """
    Write each frame's positive mask name into data.csv; raises ValueError for missing masks unless `allow_partial`.

    :param dataset_dir: str | Path - dataset directory with data.csv and positive masks
    :param config: dict | None - generation config, None loads the default
    :param allow_partial: bool - leave missing references blank instead of raising
    :return: bool - whether every frame has a mask
    """
    root = Path(dataset_dir)
    prefix = resolve_config(config)["data_format"]["positive_mask_prefix"]
    csv_path = root / "data.csv"
    with csv_path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if "rgb_img_name" not in fields:
            raise ValueError(f"{csv_path}: missing rgb_img_name column")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path}: no frames to label")
    if "positive_mask_name" not in fields:
        fields.append("positive_mask_name")
    missing = []
    all_labeled = True
    for row in rows:
        rgb_name = row["rgb_img_name"]
        if not rgb_name:
            raise ValueError(f"{csv_path}: empty RGB reference")
        name = f"positive_masks/{prefix}{Path(rgb_name).stem}.tif"
        if not (root / name).is_file():
            missing.append(name)
            all_labeled = False
            row["positive_mask_name"] = "" if allow_partial else name
        else:
            row["positive_mask_name"] = name
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete label export in {root}: {len(missing)} missing masks "
            f"({', '.join(missing[:3])}). No completion marker can be written; "
            "rerun labeling for the missing frames."
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=root, prefix=".data.csv.", delete=False
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return all_labeled

class CommonUtils:
    """Collection of static helpers for binary mask cleanup."""
    @staticmethod
    def fill_small_holes(binary_mask, max_area=500) -> np.ndarray:
        """
        Fill small holes in a binary mask.

        :param binary_mask: np.ndarray - uint8 mask, foreground 255, background 0
        :param max_area: int - largest hole area in pixels to fill
        :return: np.ndarray - filled mask
        """
        padded_mask = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        inverted_mask = cv2.bitwise_not(padded_mask)

        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max_area:
                cv2.drawContours(padded_mask, [cnt], -1, 255, -1)

        binary_mask = padded_mask[1:-1, 1:-1]
        return binary_mask

    @staticmethod
    def remove_small_islands(binary_mask, max_area=100) -> np.ndarray:
        """
        Remove small islands from a binary mask.

        :param binary_mask: np.ndarray - uint8 mask, foreground 255, background 0; modified in place
        :param max_area: int - islands below this area in pixels are cleared
        :return: np.ndarray - the input mask
        """

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 0:
                if area < max_area:
                    cv2.drawContours(binary_mask, [cnt], -1, 0, -1)

        return binary_mask

    @staticmethod
    def fill_surgical_gaps(binary_mask, max_depth=10, min_depth=2) -> np.ndarray:
        """
        Fill narrow gaps using convexity defects, on contours of at least 200 pixels.

        :param binary_mask: np.ndarray - uint8 mask, foreground 255, background 0; modified in place
        :param max_depth: int - exclusive upper bound of the filled defect depth in pixels
        :param min_depth: int - exclusive lower bound of the filled defect depth in pixels
        :return: np.ndarray - the input mask
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 200: continue

            epsilon = 0.001 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            try:
                hull_indices = cv2.convexHull(cnt, returnPoints=False)

                if len(hull_indices) < 3: continue

                defects = cv2.convexityDefects(cnt, hull_indices)

                if defects is None: continue

                for i in range(defects.shape[0]):
                    s_idx, e_idx, f_idx, d = defects[i, 0]

                    start = tuple(cnt[s_idx][0])
                    end = tuple(cnt[e_idx][0])
                    far = tuple(cnt[f_idx][0])

                    depth = d / 256.0

                    if min_depth < depth < max_depth:
                        cv2.drawContours(binary_mask, [np.array([start, end, far])], -1, 255, -1)
            except cv2.error:
                continue

        return binary_mask


