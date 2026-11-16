import hashlib
import json
import os
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import tqdm
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from omegaconf import OmegaConf

from cat.data.embedding_dataset import semantic_maps_directory
from cat.training.paths import compose_train_cfg, resolve_repo_path

def _resize_center_crop(pil_img, size) -> Image.Image:
    """
    Resize the short edge to ``size`` and center crop, matching embedding generation.

    :param pil_img: Image.Image - source RGB image
    :param size: int - edge length of the square output in pixels
    :return: Image.Image - resized, center-cropped square image
    """
    w, h = pil_img.size
    scale = max(size / w, size / h)
    new_w, new_h = max(size, int(w * scale)), max(size, int(h * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return pil_img.crop((left, top, left + size, top + size))

def _map_signature(model_name, prompts, resolution, rgb_paths) -> dict:
    """
    Describe the inputs that determine the cache.

    :param model_name: str - CLIPSeg model identifier
    :param prompts: list[str] - terrain prompts in channel order
    :param resolution: int - square output resolution
    :param rgb_paths: list[str] - source images in cache row order
    :return: dict - JSON-serializable cache signature
    """
    # hash image list so row order changes invalidate cache
    return {
        'model': model_name,
        'prompts': list(prompts),
        'resolution': resolution,
        'num_images': len(rgb_paths),
        'image_files_sha256': hashlib.sha256('\n'.join(rgb_paths).encode()).hexdigest(),
    }

def precompute_semantic_maps(embeddings_path, prompts, output_dir, resolution=224, device='cuda', batch_size=16) -> Path:
    """
    Precompute CLIPSeg semantic maps as a uint8 [N, K, 224, 224] ``semantic_maps.npy`` memmap of softmax probabilities scaled to 0-255.

    Existing maps are reused only if ``semantic_maps.json`` matches the current inputs.

    :param embeddings_path: str | Path - embeddings .npy file
    :param prompts: list[str] - text prompts, one per semantic channel (K)
    :param output_dir: str | Path - directory to save semantic_maps.npy
    :param resolution: int - must be 224 (decoder output size), otherwise ValueError
    :param device: str | torch.device - CLIPSeg inference device
    :param batch_size: int - images per batch
    :return: Path - output directory
    """
    if resolution != 224:
        raise ValueError(f"Semantic map resolution must be 224 to match the decoder, got {resolution}")

    device = torch.device(device)

    output_dir = Path(output_dir)
    model_name = "CIDAS/clipseg-rd64-refined"

    metadata = np.load(Path(embeddings_path).parent / 'metadata.npy', allow_pickle=True).item()
    rgb_paths = [str(p) for p in metadata['image_files']]
    num_images = len(rgb_paths)
    num_classes = len(prompts)

    consolidated_path = output_dir / 'semantic_maps.npy'
    signature_path = output_dir / 'semantic_maps.json'
    signature = _map_signature(model_name, prompts, resolution, rgb_paths)
    if consolidated_path.exists():
        stored = json.loads(signature_path.read_text()) if signature_path.exists() else None
        if stored == signature:
            print(f"Semantic maps already precomputed ({num_images} maps in {consolidated_path}), skipping.")
            return output_dir
        reason = "no semantic_maps.json" if stored is None else "prompts, resolution or images changed"
        print(f"Recomputing semantic maps in {output_dir}: {reason}.")

    # drop stale signature so partial runs never look valid
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path.unlink(missing_ok=True)

    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print(f"Precomputing {num_images} semantic maps at {resolution}x{resolution} "
          f"with {num_classes} prompts (batch_size={batch_size})...")

    # uint8 memmap, renamed into place when complete
    partial_path = output_dir / 'semantic_maps.partial.npy'
    out_mmap = np.lib.format.open_memmap(
        str(partial_path), mode='w+', dtype=np.uint8,
        shape=(num_images, num_classes, resolution, resolution)
    )

    # fp16 autocast only on CUDA
    with torch.inference_mode(), torch.amp.autocast(
        device.type, dtype=torch.float16, enabled=device.type == 'cuda'
    ):
        for batch_start in tqdm.trange(0, num_images, batch_size, desc="Precomputing semantic maps"):
            batch_end = min(batch_start + batch_size, num_images)
            M = batch_end - batch_start

            batch_imgs = []
            for idx in range(batch_start, batch_end):
                img = Image.open(rgb_paths[idx]).convert('RGB')
                img = _resize_center_crop(img, resolution)
                batch_imgs.append(img)

            # images repeated k times, prompts m times
            images_repeated = [img for img in batch_imgs for _ in range(num_classes)]
            texts_repeated = prompts * M

            # images pre-cropped, skip processor resize
            inputs = processor(
                text=texts_repeated,
                images=images_repeated,
                padding=True,
                return_tensors="pt",
                do_resize=False,
            ).to(device)

            logits = model(**inputs).logits  # [m*k, h, w]
            # fp32 for softmax, reshape to [m, k, h, w]
            logits = logits.float().reshape(M, num_classes, resolution, resolution)
            probs = torch.softmax(logits, dim=1)

            # quantize to uint8
            out_mmap[batch_start:batch_end] = (probs.cpu() * 255).clamp(0, 255).byte().numpy()

            if batch_start == 0:
                _save_debug_image(
                    batch_imgs[0], probs[0].cpu(), prompts,
                    output_dir / "debug_semantic_map.png"
                )

    # atomic publish of maps and signature
    out_mmap.flush()
    del out_mmap
    os.replace(partial_path, consolidated_path)
    signature_path.write_text(json.dumps(signature, indent=2))

    disk_mb = consolidated_path.stat().st_size / 1e6
    print(f"Saved {num_images} semantic maps to {consolidated_path} ({disk_mb:.0f} MB)")
    return output_dir

def _save_debug_image(pil_img, probs, prompts, save_path) -> None:
    """
    Save a debug visualization: RGB | argmax class map with legend.

    :param pil_img: Image.Image - original RGB image
    :param probs: torch.Tensor - [K, H, W] class probabilities
    :param prompts: list[str] - K prompt strings
    :param save_path: str | Path - debug image path
    """
    h, w = probs.shape[1], probs.shape[2]

    rgb = np.array(pil_img.resize((w, h)))
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    argmax = probs.argmax(dim=0).numpy()  # [h, w]
    num_classes = len(prompts)
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        color = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in color))

    class_map = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        class_map[argmax == c] = colors[c]

    combined = np.hstack([rgb_bgr, class_map])

    legend_h = 20 * ((num_classes + 3) // 4)  # 4 columns, 20 px per row
    legend = np.zeros((legend_h, combined.shape[1], 3), dtype=np.uint8)
    col_w = combined.shape[1] // 4
    for i, prompt in enumerate(prompts):
        col = i % 4
        row = i // 4
        x = col * col_w + 5
        y = row * 20 + 15
        cv2.rectangle(legend, (x, y - 10), (x + 12, y + 2), colors[i], -1)
        label = prompt.replace("a photo of ", "")
        cv2.putText(legend, label, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    final = np.vstack([combined, legend])
    cv2.imwrite(str(save_path), final)
    print(f"Saved debug semantic map to {save_path}")

def main() -> None:
    """
    Precompute semantic maps for the Hydra-selected dataset.
    """
    parser = argparse.ArgumentParser(description="Precompute CLIPSeg semantic maps")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra overrides of cat/configs/train.yaml, e.g. "
                             "data.embeddings_path=/path/embeddings.npy")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: data.semantic_maps_dir or <embeddings dir>/semantic_maps)")
    parser.add_argument("--resolution", type=int, default=None, help="Output resolution (default: data.resolution)")
    parser.add_argument("--batch_size", type=int, default=16, help="Images per batch for CLIPSeg inference")
    args = parser.parse_args()


    # cli flags take precedence over hydra overrides
    settings = OmegaConf.to_container(compose_train_cfg(args.overrides), resolve=True)
    prompts = [p['text'] for p in settings['prompts']]
    embeddings_path = str(resolve_repo_path(settings['data']['embeddings_path'], settings))
    output_dir = semantic_maps_directory(settings) if args.output_dir is None else Path(args.output_dir)
    resolution = settings['data']['resolution'][0] if args.resolution is None else args.resolution

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precompute_semantic_maps(embeddings_path, prompts, output_dir, resolution, device, args.batch_size)

if __name__ == "__main__":
    main()
