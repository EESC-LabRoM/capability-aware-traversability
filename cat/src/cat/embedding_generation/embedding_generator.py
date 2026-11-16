import os
import json
import tqdm
import random
import shutil
import hashlib
import argparse
import datetime
import torch
import timm
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from cat.configuration import load_config, resolve_config
from cat.data.traversability_dataset import TraversabilityDataset
from cat.embedding_generation.saver import Save
from cat.embedding_generation.models import get_model

def _sha1_of_file(path) -> str:
    """
    SHA1 hash of a file's raw bytes.

    :param path: str | Path - file to hash
    :return: str - SHA1 hex digest
    """
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def bytes_needed(feat_shape, cls_shape, use_depth, dtype, N) -> int:
    """
    Compute the total bytes of the cache arrays for N rows.

    :param feat_shape: tuple[int, int, int] - (D, patch_h, patch_w)
    :param cls_shape: tuple[int] - (D,)
    :param use_depth: bool - also write depth_embeddings.npy
    :param dtype: type - numpy storage dtype
    :param N: int - number of rows
    :return: int - total bytes
    """
    itemsize = np.dtype(dtype).itemsize
    per_frame = (np.prod(feat_shape) + np.prod(cls_shape)) * itemsize
    if use_depth:
        per_frame += np.prod(feat_shape) * itemsize
    return int(per_frame) * N

def preflight_disk_check(save_dir, feat_shape, cls_shape, use_depth, dtype, N, resume_from, warn_fraction) -> None:
    """
    Raise RuntimeError if the projected cache exceeds free disk space on a fresh run; on resume only print.

    :param save_dir: Path - output directory, selects the filesystem
    :param feat_shape: tuple[int, int, int] - (D, patch_h, patch_w)
    :param cls_shape: tuple[int] - (D,)
    :param use_depth: bool - also write depth_embeddings.npy
    :param dtype: type - numpy storage dtype
    :param N: int - total rows
    :param resume_from: int - resume row index, 0 if fresh
    :param warn_fraction: float - free-disk share that triggers a warning
    """
    projected = bytes_needed(feat_shape, cls_shape, use_depth, dtype, N)
    free = shutil.disk_usage(save_dir).free

    # resume reuses allocated arrays, so only report
    if resume_from > 0:
        print(f"Resuming: skipping disk preflight hard-check (arrays already allocated). "
              f"Projected total {projected / 1e9:.1f} GB, free {free / 1e9:.1f} GB.")
        return

    if projected > free:
        raise RuntimeError(
            f"Projected cache size ({projected / 1e9:.1f} GB) exceeds free disk space "
            f"({free / 1e9:.1f} GB) on {save_dir}"
        )
    if projected > warn_fraction * free:
        print(f"WARNING: projected cache size ({projected / 1e9:.1f} GB) exceeds "
              f"{warn_fraction * 100:.0f}% of free disk space ({free / 1e9:.1f} GB)")

def write_checksums(save, arrays, block_rows) -> None:
    """
    Atomically write per-block sha256 digests of each memmap to checksums.json.

    :param save: Save - output directory
    :param arrays: dict - {array_name: memmap} to checksum
    :param block_rows: int - rows per checksum block
    """
    checksums = {'block_rows': block_rows}
    for name, mmap in arrays.items():
        digests = []
        for start in range(0, mmap.shape[0], block_rows):
            block = mmap[start:start + block_rows]
            digests.append(hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest())
        checksums[name] = digests

    # tmp + replace = atomic
    tmp = Path(str(save.folder / 'checksums.json') + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(checksums, f, indent=4)
    os.replace(tmp, save.folder / 'checksums.json')

def sha1_of_paths(image_files, artificial_mask_files, traj_files, score_files) -> str:
    """
    SHA1 of the ordered file names of the rows used.

    :param image_files: list[str] - rgb file paths
    :param artificial_mask_files: list[str] - positive mask file paths
    :param traj_files: list[str] - trajectory mask file paths
    :param score_files: list[str] - score yaml file paths
    :return: str - SHA1 hex digest
    """
    hasher = hashlib.sha1()
    for rgb, mask, traj, score in zip(image_files, artificial_mask_files, traj_files, score_files):
        hasher.update(str(rgb).encode('utf-8'))
        hasher.update(str(mask).encode('utf-8'))
        hasher.update(str(traj).encode('utf-8'))
        hasher.update(str(score).encode('utf-8'))
    return hasher.hexdigest()

def build_fingerprint(network, dataset, settings, resolution, feat_shape, depth_stats_path,
                       image_files, artificial_mask_files, traj_files, score_files) -> dict:
    """
    Build the fingerprint dict (essential fields plus provenance) for this run.

    :param network: FeatureEmbedder - backbone wrapper
    :param dataset: TraversabilityDataset - dataset in use
    :param settings: dict - parsed command-line settings, uses 'dtype' and 'use_depth'
    :param resolution: tuple[int, int] - (H, W) input resolution
    :param feat_shape: tuple[int, int, int] - (D, patch_h, patch_w)
    :param depth_stats_path: Path - depth_stats.json, hashed when depth is used
    :param image_files: list[str] - rgb file paths of the rows used
    :param artificial_mask_files: list[str] - positive mask file paths of the rows used
    :param traj_files: list[str] - trajectory mask file paths of the rows used
    :param score_files: list[str] - score yaml file paths of the rows used
    :return: dict - fingerprint fields for this run
    """
    # prefix tokens (cls etc.) precede patch tokens
    num_prefix = network.dino.num_prefix_tokens
    normalize = dataset._normalize

    return {
        'backbone_id': network.dino_model,
        'embed_dim': network.dino.embed_dim,
        'patch_grid': [feat_shape[1], feat_shape[2]],
        'resolution': list(resolution),
        'token_selection': f"patches[{num_prefix}:], num_prefix_tokens={num_prefix}",
        'dtype': settings['dtype'],
        'rgb_mean': list(normalize.mean),
        'rgb_std': list(normalize.std),
        'use_depth': settings['use_depth'],
        'depth_stats_sha1': _sha1_of_file(depth_stats_path) if settings['use_depth'] else None,
        'resize': "aspect-preserving resize + center crop",
        'num_frames': len(dataset),
        'dataset_id': sha1_of_paths(image_files, artificial_mask_files, traj_files, score_files),
        'created': datetime.datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'timm_version': timm.__version__,
    }

def check_fingerprint_match(existing, current, essential_fields) -> None:
    """
    Raise ValueError listing each differing essential field.

    :param existing: dict - fingerprint from the cache directory
    :param current: dict - fingerprint of the current run
    :param essential_fields: list[str] - fields that must match
    """
    mismatches = []
    for field in essential_fields:
        if existing[field] != current[field]:
            mismatches.append(f"{field}: expected {existing[field]}, found {current[field]}")
    if mismatches:
        raise ValueError(
            "Existing cache is incompatible with the current run config:\n  "
            + "\n  ".join(mismatches)
            + "\nUse --force to wipe and regenerate."
        )

def extract_embeddings(network, dataloader, device, save, use_depth, resolution, dtype, resume_from, N,
                        image_files, artificial_mask_files, traj_files, score_files, sensors, embedding) -> None:
    """
    Extract raw DINOv3 patch, CLS and depth embeddings into the cache, resumable per batch.

    :param network: FeatureEmbedder - backbone wrapper with a .dino attribute
    :param dataloader: DataLoader - loader over the rows still to write
    :param device: torch.device - cuda, or cpu without fp16 autocast
    :param save: Save - output directory
    :param use_depth: bool - also extract depth embeddings
    :param resolution: tuple[int, int] - (H, W) input, gives the patch grid
    :param dtype: type - numpy storage dtype (np.float16 or np.float32)
    :param resume_from: int - resume row index, 0 if fresh
    :param N: int - total number of rows
    :param image_files: list[str] - rgb file paths of all rows
    :param artificial_mask_files: list[str] - positive mask file paths of all rows
    :param traj_files: list[str] - trajectory mask file paths of all rows
    :param score_files: list[str] - score yaml file paths of all rows
    :param sensors: list[str] - sensor of every row of the whole dataset, not just the rows this run extracts
    :param embedding: dict - embedding config section (throughput, checksums, disk)
    """
    save_dir = save.folder

    # dinov3 patch size is 16
    network.eval()
    num_prefix = network.dino.num_prefix_tokens
    D = network.dino.embed_dim
    patch_h, patch_w = resolution[0] // 16, resolution[1] // 16
    feat_shape = (D, patch_h, patch_w)
    cls_shape = (D,)

    print(f"Dataset size: {N}")
    print(f"Raw DINOv3 feature shape per image: {feat_shape}")
    print(f"CLS token shape per image: {cls_shape}")

    emb_path = save_dir / 'embeddings.npy'
    cls_path = save_dir / 'cls_tokens.npy'
    depth_emb_path = save_dir / 'depth_embeddings.npy'

    preflight_disk_check(save_dir, feat_shape, cls_shape, use_depth, dtype, N, resume_from,
                         embedding['disk']['warn_fraction'])

    if resume_from > 0:
        print(f"Resuming from row {resume_from} (of {N})")
        emb_mmap = np.lib.format.open_memmap(str(emb_path), mode='r+')
        cls_mmap = np.lib.format.open_memmap(str(cls_path), mode='r+')
        if use_depth:
            depth_emb_mmap = np.lib.format.open_memmap(str(depth_emb_path), mode='r+')
    else:
        emb_mmap = np.lib.format.open_memmap(
            str(emb_path), mode='w+', dtype=dtype, shape=(N, *feat_shape)
        )
        cls_mmap = np.lib.format.open_memmap(
            str(cls_path), mode='w+', dtype=dtype, shape=(N, *cls_shape)
        )
        if use_depth:
            depth_emb_mmap = np.lib.format.open_memmap(
                str(depth_emb_path), mode='w+', dtype=dtype, shape=(N, *feat_shape)
            )

    if use_depth:
        print(f"Depth embeddings will be saved to: {depth_emb_path}")

    bytes_per_frame = bytes_needed(feat_shape, cls_shape, use_depth, dtype, 1)
    total_gb = bytes_needed(feat_shape, cls_shape, use_depth, dtype, N) / 1e9
    frames_per_second = embedding['throughput']['frames_per_second']
    eta_sec = (N - resume_from) / frames_per_second
    print(f"Estimate: {N} frames, {bytes_per_frame:.0f} bytes/frame ({dtype.__name__}), "
          f"projected total {total_gb:.1f} GB, "
          f"ETA ~{eta_sec / 60:.1f} min at {frames_per_second} frames/s "
          f"(measured on NVIDIA L40S; scales with CPU preprocessing)")

    offset = resume_from
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
        for item in tqdm.tqdm(dataloader):
            data = item['data']
            rgb_images = data[0].to(device, non_blocking=True)

            dino_out = network.dino.forward_features(rgb_images)
            dino_cls = dino_out[:, 0].float()                    # [b, d]
            dino_patches = dino_out[:, num_prefix:].float()      # [b, n_patches, d]
            B = dino_patches.shape[0]
            features = dino_patches.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)

            emb_mmap[offset:offset+B] = features.cpu().numpy().astype(dtype)
            cls_mmap[offset:offset+B] = dino_cls.cpu().numpy().astype(dtype)

            if use_depth:
                depth_images = data[1].to(device, non_blocking=True)
                depth_3ch = depth_images.repeat(1, 3, 1, 1)  # [b, 1, h, w] -> [b, 3, h, w]
                depth_dino_out = network.dino.forward_features(depth_3ch)
                depth_patches = depth_dino_out[:, num_prefix:].float()
                depth_feats = depth_patches.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)
                depth_emb_mmap[offset:offset+B] = depth_feats.cpu().numpy().astype(dtype)

            offset += B
            save.write_progress(offset)

    checksum_arrays = {'embeddings.npy': emb_mmap, 'cls_tokens.npy': cls_mmap}
    if use_depth:
        checksum_arrays['depth_embeddings.npy'] = depth_emb_mmap
    write_checksums(save, checksum_arrays, embedding['checksums']['block_rows'])

    # del memmaps to flush to disk
    del emb_mmap
    del cls_mmap
    if use_depth:
        del depth_emb_mmap

    np.save(
        save_dir / 'metadata.npy',
        {'image_files': image_files, 'artificial_mask_files': artificial_mask_files, 'traj_files': traj_files, 'score_files': score_files,
         'sensors': list(sensors)}
    )

    print(f"Embeddings shape: ({N}, {', '.join(str(s) for s in feat_shape)})")
    print(f"CLS tokens shape: ({N}, {', '.join(str(s) for s in cls_shape)})")
    if use_depth:
        print(f"Depth embeddings shape: ({N}, {', '.join(str(s) for s in feat_shape)})")
    print(f"Total images: {N}")
    print(f"Saved to: {save_dir}")

def main(settings, config=None) -> None:
    """
    Generate, resume, or skip (if complete) an embeddings cache.

    :param settings: dict - parsed command-line settings
    :param config: dict | None - CAT configuration, defaults loaded when None
    """
    config = resolve_config(config)
    if (settings['res_y'], settings['res_x']) != (224, 224):
        raise ValueError("Embedding generation supports only resolution=(224, 224)")
    save_folder = Path(settings['save_folder'])

    if settings['force'] and save_folder.exists():
        shutil.rmtree(save_folder)

    if save_folder.exists():
        existing_fingerprint_file = save_folder / 'fingerprint.json'
        if not existing_fingerprint_file.exists():
            raise ValueError(
                f"{save_folder} exists but has no fingerprint.json; cannot verify compatibility. "
                f"Use --force to wipe and regenerate."
            )
        with open(existing_fingerprint_file, 'r') as f:
            existing_fingerprint = json.load(f)
    else:
        existing_fingerprint = None

    # CUBLAS_WORKSPACE_CONFIG is required by deterministic cuBLAS
    seed = settings['seed']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA unavailable, extracting on CPU (slow)")
    folder_path = Path(settings['data_folder'])
    csv_file = Path(settings['data_csv'])
    resolution = (settings['res_y'], settings['res_x'])
    in_memory = True if settings['in_memory'] else False
    use_depth = True if settings['use_depth'] else False
    dtype = np.dtype(settings['dtype']).type

    dataset = TraversabilityDataset(
        folder_path,
        csv_file,
        resolution,
        in_memory=in_memory,
        use_depth=use_depth,
        max_frames=settings['max_frames'],
        config=config,
    )

    network = get_model(settings['model'])
    network.to(device)

    D = network.dino.embed_dim
    patch_h, patch_w = resolution[0] // 16, resolution[1] // 16
    feat_shape = (D, patch_h, patch_w)

    image_files = [str(p) for p in dataset._rgb_fname_list]
    artificial_mask_files = [str(p) for p in dataset._artificial_mask_fname_list]
    traj_files = [str(p) for p in dataset._traj_fname_list]
    score_files = [str(p) for p in dataset._score_fname_list]
    sensors = list(dataset._sensor_list)

    # stage depth stats in a tmp file so the fingerprint can hash them without touching the cache
    save = Save(save_folder)
    depth_stats_path = save.folder / 'depth_stats.json'
    depth_stats_tmp = Path(str(depth_stats_path) + '.tmp')
    try:
        if use_depth:
            with open(depth_stats_tmp, 'w') as f:
                json.dump({
                    'depth_mean_by_sensor': {s: float(m) for s, m in dataset._depth_mean.items()},
                    'depth_std_by_sensor': {s: float(s_) for s, s_ in dataset._depth_std.items()},
                    'depth_clip_by_sensor': {s: list(c) for s, c in dataset.depth_clip_by_sensor.items()},
                    'depth_min_valid_fraction': dataset.depth_min_valid_fraction,
                    'invalid_rule': 'raw depth == 0 or non-finite excluded from stats; readings min-pooled to the '
                                    'input size, gaps take the nearest reading; frames below depth_min_valid_fraction '
                                    'coverage use the zero (mean) input',
                }, f, indent=4)

        # existing cache must match this fingerprint before anything is replaced
        current_fingerprint = build_fingerprint(
            network, dataset, settings, resolution, feat_shape, depth_stats_tmp,
            image_files, artificial_mask_files, traj_files, score_files,
        )
        if existing_fingerprint is not None:
            check_fingerprint_match(existing_fingerprint, current_fingerprint,
                                    config['embedding']['essential_fingerprint_fields'])

        save.write_settings(settings)
        if use_depth:
            os.replace(depth_stats_tmp, depth_stats_path)
    finally:
        if use_depth:
            depth_stats_tmp.unlink(missing_ok=True)

    if existing_fingerprint is not None:
        if existing_fingerprint['complete']:
            print("cache already complete")
            return
        resume_from = save.read_progress()
    else:
        resume_from = 0
        fingerprint = dict(current_fingerprint)
        fingerprint['complete'] = False
        save.write_fingerprint(fingerprint)

    N = len(dataset)
    loader_dataset = dataset if resume_from == 0 else Subset(dataset, range(resume_from, N))
    dataloader = DataLoader(
        loader_dataset,
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        pin_memory=device.type == 'cuda',
        persistent_workers=settings['num_workers'] > 0,
        prefetch_factor=2 if settings['num_workers'] > 0 else None,
    )

    extract_embeddings(
        network, dataloader, device, save, use_depth, resolution, dtype, resume_from, N,
        image_files=image_files,
        artificial_mask_files=artificial_mask_files,
        traj_files=traj_files,
        score_files=score_files,
        sensors=sensors,
        embedding=config['embedding'],
    )

    current_fingerprint['complete'] = True
    save.write_fingerprint(current_fingerprint)

    print("Finished generating embeddings!")

def parse_config_path(argv=None) -> str | None:
    """
    Read only the --config option.

    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    :return: str | None - YAML override path
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_known_args(argv)[0].config

def parse_args(config, argv=None) -> argparse.Namespace:
    """
    Parse the command line.

    :param config: dict - resolved CAT configuration, supplies the allowed dtypes
    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    :return: argparse.Namespace - parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate a DINOv3 embeddings cache for the traversability dataset")

    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML overriding the CAT configuration (embedding/data sections)")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--use_depth", default=False, action='store_true')

    parser.add_argument("--res_x", type=int, default=224)
    parser.add_argument("--res_y", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--in_memory", default=False, action='store_true')
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Truncate the dataset to the first max_frames rows (CSV order); for smoke tests")
    parser.add_argument("--force", default=False, action='store_true',
                        help="Wipe an existing (incompatible) cache directory and regenerate")

    parser.add_argument("--dtype", type=str, default="float16", choices=config['embedding']['dtypes'],
                        help="Storage dtype for embeddings/cls_tokens/depth_embeddings")

    parser.add_argument("--model", type=str, default="dino3",
                        choices=["dino3"],
                        help="Model to use for embedding generation")

    return parser.parse_args(argv)

def cli(argv=None) -> None:
    """
    Run embedding generation from the CLI.

    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    """
    config = load_config(parse_config_path(argv))
    main(vars(parse_args(config, argv)), config)

if __name__ == "__main__":
    cli()
