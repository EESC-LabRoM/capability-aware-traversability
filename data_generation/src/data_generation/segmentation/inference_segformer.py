import os
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import glob
import cv2
from data_generation.configuration import load_config, resolve_config
from data_generation.utils.common_utils import CommonUtils, update_positive_mask_references
from data_generation.utils.ego_mask import prepare_ego_mask

class EgoMaskSizeError(ValueError):
    """Image shape differs from the ego mask shape; must abort, never skip."""

def check_ego_size(ego, image_hw, name) -> None:
    """
    Raise EgoMaskSizeError on image/ego mask shape mismatch.

    :param ego: EgoMask - registered ego mask, a mask of None disables the check
    :param image_hw: tuple[int, int] - image (height, width)
    :param name: str - image file name for the error
    """
    if ego.mask is not None and tuple(image_hw) != ego.mask.shape:
        raise EgoMaskSizeError(f"Ego mask shape {ego.mask.shape} does not match image {name} shape {tuple(image_hw)}")

def postprocess_prediction(predicted_mask, ego, last_mask, segformer) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Clean a predicted class map and zero ego pixels; fall back to the last non-empty mask if the result is empty.

    :param predicted_mask: np.ndarray - (H, W) class map with values 0/1
    :param ego: EgoMask - ego mask whose pixels are excluded
    :param segformer: dict - postprocessing settings
    :param last_mask: np.ndarray | None - last non-empty cleaned 0/255 mask
    :return: tuple[np.ndarray, np.ndarray | None] - (H, W) 0/255 uint8 mask and updated fallback mask
    """
    binary_mask_img = (predicted_mask * 255).astype(np.uint8)

    binary_mask_img = CommonUtils.fill_small_holes(
        binary_mask_img, max_area=segformer["max_hole_size"]
    )
    binary_mask_img = CommonUtils.remove_small_islands(
        binary_mask_img, max_area=segformer["max_island_size"]
    )

    contours, _ = cv2.findContours(binary_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary_mask_img.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (y + h) < (height * segformer["max_component_height_ratio"]):
            cv2.drawContours(binary_mask_img, [cnt], -1, 0, -1)

    if np.all(ego.exclude(binary_mask_img) == 0) and last_mask is not None:
        binary_mask_img = last_mask.copy()

    if not np.all(ego.exclude(binary_mask_img) == 0):
        last_mask = binary_mask_img.copy()

    return ego.exclude(binary_mask_img), last_mask

def render_visualization(image_rgb, mask_bool, ego) -> np.ndarray:
    """
    Overlay positives in green and tint ego pixels.

    :param image_rgb: np.ndarray - (H, W, 3) RGB uint8 image
    :param mask_bool: np.ndarray | None - (H, W) bool positive mask, None for no positives
    :param ego: EgoMask - ego mask overlay
    :return: np.ndarray - contiguous (H, W, 3) RGB uint8 image
    """
    final_image = image_rgb.copy()
    if mask_bool is not None:
        green = np.array([0, 255, 0], dtype=np.float32)
        final_image[mask_bool] = (image_rgb[mask_bool] * 0.4 + green * 0.6).astype(np.uint8)
    return np.ascontiguousarray(ego.overlay(final_image[..., ::-1])[..., ::-1])

def collect_dataset_images(ds_name, config) -> tuple[str, list[str]]:
    """
    List a dataset's RGB frames.

    :param ds_name: str - dataset directory name under the processed dataset directory
    :param config: dict - generation config
    :return: tuple[str, list[str]] - dataset path and sorted unique RGB image paths
    """
    ds_path = os.path.join(config["bag_processing"]["output_dataset_dir"], ds_name)
    rgb_dir = os.path.join(ds_path, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Dataset {ds_name}: RGB directory not found at {rgb_dir}")
    if not os.path.isfile(os.path.join(ds_path, "data.csv")):
        raise FileNotFoundError(f"Dataset {ds_name}: data.csv not found in {ds_path}; training needs it to reference the generated masks")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
        image_files.extend(glob.glob(os.path.join(rgb_dir, ext.upper())))
    image_files = sorted(set(image_files))
    if not image_files:
        raise ValueError(f"Dataset {ds_name}: no RGB images found in {rgb_dir}")
    return ds_path, image_files

def predict_dataset(model_path, datasets, config=None, overwrite=False, overwrite_empty_masks=False) -> None:
    """
    Run SegFormer inference and write 0/255 positive masks.

    :param model_path: str - trained SegFormer model directory
    :param datasets: list[str] - dataset names under the processed dataset directory
    :param config: dict | None - generation config, None loads the default
    :param overwrite: bool - replace existing nonempty masks
    :param overwrite_empty_masks: bool - regenerate existing all-zero masks instead of keeping deliberate negatives
    """
    config = resolve_config(config)
    if not isinstance(config, dict):
        raise ValueError("Inference config must be a YAML mapping.")
    all_images = []
    for ds_name in datasets:
        ds_path, image_files = collect_dataset_images(ds_name, config)
        all_images.append((ds_name, ds_path, image_files))
    if not all_images:
        raise ValueError("No datasets requested.")

    prefix = config["data_format"]["positive_mask_prefix"]
    existing_nonempty = []
    for ds_name, ds_path, image_files in all_images:
        mask_dir = os.path.join(ds_path, "positive_masks")
        for image_path in image_files:
            name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_dir, f"{prefix}{name}.tif")
            if os.path.isfile(mask_path):
                with Image.open(mask_path) as existing_image:
                    if np.asarray(existing_image).any():
                        existing_nonempty.append(mask_path)
    if existing_nonempty and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing nonempty positive masks; pass --overwrite: {existing_nonempty[0]}"
        )

    configured_device = config["device"]
    if not isinstance(configured_device, str):
        raise ValueError(f"Invalid configured device {configured_device!r}; expected 'cpu' or 'cuda'")
    try:
        device = torch.device(configured_device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid configured device {configured_device!r}; expected 'cpu' or CUDA") from error
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and configured_device != "cpu"):
        raise ValueError(f"Unsupported configured device: {configured_device!r}; expected 'cpu' or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        elif device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Configured CUDA device {configured_device!r} is unavailable")
    if device.type == "cuda" and device.index is not None and device.index < 0:
        raise ValueError(f"Configured CUDA device {configured_device!r} is invalid")

    if overwrite:
        for _, ds_path, _ in all_images:
            sam_marker = os.path.join(ds_path, "sam_completed.txt")
            if os.path.exists(sam_marker):
                os.remove(sam_marker)

    print(f"Loading model from '{model_path}'...", flush=True)
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)

    model.to(device)
    model.eval()

    total_images = sum(len(imgs) for _, _, imgs in all_images)
    global_step = 0

    for ds_name, ds_path, image_files in all_images:
        marker_path = os.path.join(ds_path, "segformer_completed.txt")
        if os.path.exists(marker_path):
            os.remove(marker_path)  # drop the stale completion marker before outputs change

        vis_dir = os.path.join(ds_path, "visualization")
        mask_dir = os.path.join(ds_path, "positive_masks")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        with Image.open(image_files[0]) as first_image:
            ego = prepare_ego_mask(ds_path, config, first_image.size[::-1])
        print(f"{ds_name}: {ego.description}", flush=True)

        last_mask = None
        for i, image_path in enumerate(image_files):
            image = Image.open(image_path).convert("RGB")

            base_name = os.path.basename(image_path)
            name = os.path.splitext(base_name)[0]

            vis_save_path = os.path.join(vis_dir, f"vis_{name}.jpg")
            mask_save_path = os.path.join(mask_dir, f"{config['data_format']['positive_mask_prefix']}{name}.tif")

            original_np = np.array(image)
            check_ego_size(ego, original_np.shape[:2], base_name)

            if os.path.exists(mask_save_path):
                with Image.open(mask_save_path) as existing_image:
                    existing_mask = np.array(existing_image)
                if existing_mask.shape != original_np.shape[:2]:
                    raise ValueError(f"Existing mask geometry differs from RGB: {mask_save_path}")
                if not existing_mask.any() and not overwrite_empty_masks:
                    Image.fromarray(render_visualization(original_np, None, ego)).save(vis_save_path)
                    global_step += 1
                    print(f"Processed frame {global_step}/{total_images} | {ds_name} - {base_name} [preserved empty label]", flush=True)
                    continue

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            predicted_mask = logits.argmax(dim=1)[0].cpu().numpy()

            binary_mask_img, last_mask = postprocess_prediction(
                predicted_mask, ego, last_mask, config["segformer"]
            )
            final_image = render_visualization(original_np, binary_mask_img > 0, ego)
            Image.fromarray(final_image).save(vis_save_path)
            Image.fromarray(binary_mask_img, mode="L").save(mask_save_path)

            global_step += 1
            print(f"Processed frame {global_step}/{total_images} | {ds_name} - {base_name}", flush=True)

        update_positive_mask_references(ds_path, config)
        with open(marker_path, "w") as f:
            f.write("completed")

def main(argv=None) -> None:
    """
    Run SegFormer inference from CLI arguments.

    :param argv: list[str] | None - arguments, None uses sys.argv
    """
    parser = argparse.ArgumentParser(description="Test SegFormer on datasets.")
    parser.add_argument("--dataset", nargs="+", required=True, help="List of dataset names in dataset_processed/")
    parser.add_argument("--model_path", type=str, default="checkpoints/segformer_finetuned", help="Directory of the trained model.")
    parser.add_argument("--config", default=None, help="YAML overrides for the generation configuration")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing nonempty positive masks")
    parser.add_argument("--overwrite_empty_masks", action="store_true", help="Regenerate existing all-zero labels instead of preserving deliberate negatives")

    args = parser.parse_args(argv)

    config = load_config(args.config)
    predict_dataset(args.model_path, args.dataset, config, args.overwrite, args.overwrite_empty_masks)

if __name__ == "__main__":
    main()
