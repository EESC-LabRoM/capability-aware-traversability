import cv2
import numpy as np

def cover_crop_geometry(height, width, size) -> tuple[int, int, int, int]:
    """
    Compute the RGB path's aspect-preserving cover resize and center crop.

    :param height: int - source height
    :param width: int - source width
    :param size: tuple[int, int] - target (H, W)
    :return: tuple[int, int, int, int] - resized height and width, crop row and column offsets
    """
    target_h, target_w = size
    scale = max(target_w / width, target_h / height)
    new_w = max(target_w, int(width * scale))
    new_h = max(target_h, int(height * scale))
    return new_h, new_w, new_h // 2 - target_h // 2, new_w // 2 - target_w // 2

def pool_depth(depth_m, size) -> np.ndarray:
    """
    Resize sparse depth to `size`, keeping the nearest reading per output pixel.

    :param depth_m: np.ndarray - (H, W) depth in meters, <= 0 or non-finite = no reading
    :param size: tuple[int, int] - target (H, W)
    :return: np.ndarray - (H, W) float32 depth in meters, 0 = no reading
    """
    height, width = depth_m.shape
    target_h, target_w = size
    new_h, new_w, top, left = cover_crop_geometry(height, width, size)
    depth = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0).astype(np.float32)

    if new_h >= height and new_w >= width:
        resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized[top:top + target_h, left:left + target_w]

    rows, cols = np.nonzero(depth)
    out_rows = ((rows + 0.5) * (new_h / height)).astype(np.int64) - top
    out_cols = ((cols + 0.5) * (new_w / width)).astype(np.int64) - left
    inside = (out_rows >= 0) & (out_rows < target_h) & (out_cols >= 0) & (out_cols < target_w)
    pooled = np.full(target_h * target_w, np.inf, dtype=np.float32)
    np.minimum.at(pooled, out_rows[inside] * target_w + out_cols[inside], depth[rows[inside], cols[inside]])
    pooled[np.isinf(pooled)] = 0
    return pooled.reshape(target_h, target_w)

def fill_nearest(depth) -> np.ndarray:
    """
    Fill pixels without a reading from the nearest pixel with one.

    :param depth: np.ndarray - (H, W) depth, 0 = no reading; needs at least one reading
    :return: np.ndarray - (H, W) filled depth
    """
    valid = depth > 0
    # each zero pixel labels its own voronoi cell, row-major
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    return depth[valid][labels - 1]

def prepare_depth_input(depth_mm, size, clip, mean, std, min_valid_fraction) -> np.ndarray | None:
    """
    Pool, fill, clip and normalize millimeter depth into the embedder's input.

    :param depth_mm: np.ndarray - (H, W) depth in millimeters, 0 = no reading
    :param size: tuple[int, int] - output (H, W)
    :param clip: tuple[float, float] - sensor (min, max) range in meters
    :param mean: float - sensor depth mean in meters
    :param std: float - sensor depth std in meters
    :param min_valid_fraction: float - minimum fraction of pooled pixels with a reading
    :return: np.ndarray | None - (H, W) float32 normalized depth, None if coverage < `min_valid_fraction`
    """
    pooled = pool_depth(depth_mm.astype(np.float32) * 1e-3, size)
    if np.count_nonzero(pooled) < max(1, min_valid_fraction * pooled.size):
        return None
    depth = np.clip(fill_nearest(pooled), clip[0], clip[1])
    return ((depth - mean) / std).astype(np.float32)
