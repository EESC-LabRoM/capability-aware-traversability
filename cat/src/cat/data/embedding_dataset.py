import cv2
import json
import yaml
import numpy as np
import pandas as pd
import warnings
import torch

from pathlib import Path
from torch.utils.data import Dataset

from cat.configuration import resolve_config
from cat.data.ego_mask import ValidityMasks
from cat.training.paths import resolve_repo_path

def embeddings_directory(settings) -> Path:
    """
    Locate the embedding cache directory.

    :param settings: dict - training or checkpoint settings
    :return: Path - directory with embeddings.npy and metadata.npy
    """
    return resolve_repo_path(settings['data']['embeddings_path'], settings).parent

def semantic_maps_directory(settings) -> Path:
    """
    Locate the semantic map folder, default <embeddings dir>/semantic_maps.

    :param settings: dict - training or checkpoint settings
    :return: Path - directory with semantic_maps.npy
    """
    configured = settings['data'].get('semantic_maps_dir')
    if configured is None:
        return embeddings_directory(settings) / 'semantic_maps'
    return resolve_repo_path(configured, settings)

def frame_sensors(image_files, data_csv) -> list[str]:
    """
    Look up each frame's sensor in the extraction CSV by rgb file name.

    :param image_files: list[str] - cached rgb file paths
    :param data_csv: str | Path | None - extraction CSV
    :return: list[str] - sensor per frame, '' when unknown
    """
    if data_csv is None or not Path(data_csv).exists():
        warnings.warn(f"Cannot determine frame sensors: training CSV not found at {data_csv}. "
                      "Set data.data_csv to the extraction CSV to enable path ranking metrics.", stacklevel=2)
        return [''] * len(image_files)
    csv = pd.read_csv(data_csv, usecols=lambda column: column in ('rgb_img_name', 'source_bag'))
    if 'source_bag' not in csv.columns:
        warnings.warn(f"Cannot determine frame sensors: {data_csv} has no source_bag column. "
                      "Path ranking metrics will be unavailable.", stacklevel=2)
        return [''] * len(image_files)
    by_name = {
        Path(str(rgb)).name: str(bag).split('_')[1] if '_' in str(bag) else ''
        for rgb, bag in zip(csv['rgb_img_name'], csv['source_bag'])
    }
    sensors = [by_name.get(Path(str(p)).name, '') for p in image_files]
    unknown = sum(not sensor for sensor in sensors)
    if unknown:
        warnings.warn(f"Sensor lookup failed for {unknown}/{len(sensors)} cached frames in {data_csv}. "
                      "Use the extraction CSV in data.data_csv to retain path ranking metrics.", stacklevel=2)
    return sensors

def build_embedding_dataset(settings, use_depth) -> 'EmbeddingDataset':
    """
    Build the embedding dataset; raise FileNotFoundError if use_depth is set but depth_embeddings.npy is missing.

    :param settings: dict - training or checkpoint settings
    :param use_depth: bool - load depth embeddings
    :return: EmbeddingDataset - dataset over the configured cache
    """
    emb_dir = embeddings_directory(settings)
    metadata = np.load(emb_dir / 'metadata.npy', allow_pickle=True).item()
    required_files = ('traj_files', 'score_files')
    missing_files = [key for key in required_files if metadata.get(key) is None]
    if missing_files:
        raise ValueError(
            f"Invalid embedding cache at {emb_dir}: metadata is missing required "
            f"file lists {missing_files}; regenerate the cache."
        )
    data = {
        'embeddings_path': emb_dir / 'embeddings.npy',
        'cls_tokens_path': emb_dir / 'cls_tokens.npy',
        'image_files': metadata['image_files'],
        'artificial_mask_files': metadata['artificial_mask_files'],
        'traj_files': metadata['traj_files'],
        'score_files': metadata['score_files'],
        'sensors': metadata['sensors'] if 'sensors' in metadata else frame_sensors(
            metadata['image_files'], settings['data']['data_csv']),
    }
    if use_depth:
        depth_path = emb_dir / 'depth_embeddings.npy'
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth model requires depth embeddings at {depth_path}")
        data['depth_embeddings_path'] = depth_path
    return EmbeddingDataset(
        data,
        tuple(settings['data']['resolution']),
        semantic_maps_dir=semantic_maps_directory(settings),
        preload_embeddings=settings['data'].get('preload_embeddings', False),
        num_prompts=len(settings['prompts']),
        config=settings,
    )

class EmbeddingDataset(Dataset):
    """Embedding dataset for traversability prediction."""
    def __init__(
        self,
        data,
        mask_resolution,
        semantic_maps_dir=None,
        preload_embeddings=False,
        num_prompts=None,
        config=None,
    ) -> None:
        """
        Load embeddings, costs, semantic maps, masks and ego validity.

        :param data: dict - embeddings, cls tokens, image, mask, trajectory and score file paths
        :param mask_resolution: tuple[int, int] - (H, W) to resize masks to
        :param semantic_maps_dir: str | Path | None - directory with precomputed semantic maps
        :param preload_embeddings: bool - load embeddings into RAM instead of memory-mapping
        :param num_prompts: int | None - number of terrain prompts (K), validates score files
        :param config: dict | None - CAT configuration, defaults loaded when None
        """
        super().__init__()
        self._config = resolve_config(config)
        self._robot_name_map = self._config["robots"]["name_map"]
        self._robot_order = list(self._config["robots"]["order"])

        self._mask_resolution = mask_resolution  # (h, w)
        self._semantic_maps_dir = Path(semantic_maps_dir) if semantic_maps_dir else None

        self._validate_cache(Path(data['embeddings_path']))

        if preload_embeddings:
            print("Preloading embeddings into RAM...")
            self._emb_data = np.load(str(data['embeddings_path']))
            self._cls_data = np.load(str(data['cls_tokens_path']))
            print(f"  Embeddings: {self._emb_data.nbytes / 1e9:.1f} GB in RAM")
        else:
            self._emb_data = np.load(str(data['embeddings_path']), mmap_mode='r')
            self._cls_data = np.load(str(data['cls_tokens_path']), mmap_mode='r')

        depth_emb_path = data.get('depth_embeddings_path')
        if depth_emb_path is not None and Path(str(depth_emb_path)).exists():
            if preload_embeddings:
                self._depth_emb_data = np.load(str(depth_emb_path))
            else:
                self._depth_emb_data = np.load(str(depth_emb_path), mmap_mode='r')
            print(f"Depth embeddings loaded: {self._depth_emb_data.shape}")
        else:
            self._depth_emb_data = None

        self.mask_files = data['artificial_mask_files']
        self.rgb_files = data['image_files']
        self.traj_files = data.get('traj_files')
        self.sensors = data.get('sensors') or [''] * len(self.rgb_files)  # '' when unknown

        score_files = data.get('score_files')
        if score_files is None:
            raise ValueError("Invalid embedding cache: required score_files are missing")
        self._costs = self._load_all_costs(score_files, num_prompts)
        self.robot_names = list(self._robot_order)
        self.num_robots = len(self._robot_order)

        self._sem_mmap = None
        if self._semantic_maps_dir is not None:
            consolidated = self._semantic_maps_dir / 'semantic_maps.npy'
            if not consolidated.exists():
                raise FileNotFoundError(f"Semantic maps not found at {consolidated}; run precompute_semantic_maps first")
            self._sem_mmap = np.load(str(consolidated), mmap_mode='r')
            print(f"Using consolidated semantic maps: {consolidated}")

        self._load_all_masks()

        self._validity = ValidityMasks(self.rgb_files, self._mask_resolution, self._resize, self._config)
        if self._validity.num_masked_frames:
            print(f"Ego masks: {self._validity.num_masked_frames}/{len(self.rgb_files)} frames have ignored pixels")

    def _validate_cache(self, embeddings_path) -> None:
        """
        Check the cache against its fingerprint.json; skipped for legacy caches without one.

        :param embeddings_path: Path - path to embeddings.npy
        :raises AssertionError: cache is incomplete or disagrees with its fingerprint
        """
        fingerprint_path = embeddings_path.parent / 'fingerprint.json'
        if not fingerprint_path.exists():
            print("legacy cache without fingerprint — skipping validation")
            return

        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)

        assert fingerprint['complete'], \
            f"Cache at {embeddings_path.parent} is incomplete (fingerprint complete=False)"

        emb_header = np.load(str(embeddings_path), mmap_mode='r')
        assert emb_header.dtype == np.dtype(fingerprint['dtype']), \
            f"Cache dtype {emb_header.dtype} does not match fingerprint dtype {fingerprint['dtype']}"
        assert emb_header.shape[0] == fingerprint['num_frames'], \
            f"Cache has {emb_header.shape[0]} rows but fingerprint says num_frames={fingerprint['num_frames']}"

    def _load_all_costs(self, score_files, num_prompts) -> np.ndarray:
        """
        Parse the YAML score files into a cost array; raise AssertionError if a robot entry lacks num_prompts categories.

        :param score_files: list[Path] - N score file paths, duplicates allowed
        :param num_prompts: int - expected number of terrain categories (K)
        :return: np.ndarray - [N, R, K] float32 costs, 0 for robots absent from a file
        """
        N = len(score_files)
        R = len(self._robot_order)
        K = num_prompts

        # path_str -> {generic_name: [k] scores}
        yaml_cache = {}
        unique_paths = set(str(p) for p in score_files)
        print(f"Loading traversability costs from {len(unique_paths)} unique YAML files...")

        for path_str in unique_paths:
            with open(path_str, 'r') as f:
                raw = yaml.safe_load(f)
            parsed = {}
            for yaml_name, categories in raw.items():
                generic_name = self._robot_name_map.get(yaml_name)
                if generic_name is None:
                    continue
                scores = [entry['score'] for entry in categories]
                assert len(scores) == K, \
                    f"YAML {path_str} robot '{yaml_name}' has {len(scores)} categories, expected {K}"
                parsed[generic_name] = scores
            yaml_cache[path_str] = parsed

        # robots absent from a file stay 0
        costs = np.zeros((N, R, K), dtype=np.float32)
        for i, path in enumerate(score_files):
            parsed = yaml_cache[str(path)]
            for r, robot_name in enumerate(self._robot_order):
                if robot_name in parsed:
                    costs[i, r, :] = parsed[robot_name]
        print(f"  Traversability costs: {costs.nbytes / 1e6:.1f} MB ({N} samples × {R} robots × {K} classes)")
        return costs

    def _resize(self, image, size, interpolation=cv2.INTER_AREA) -> np.ndarray:
        """
        Aspect-preserving resize and center crop, as in embedding generation.

        :param image: np.ndarray - (H, W, C) or (H, W) image
        :param size: tuple[int, int] - target (H, W)
        :param interpolation: int - cv2 interpolation method
        :return: np.ndarray - cropped image
        """
        target_h, target_w = size
        h, w = image.shape[:2]

        scale = max(target_w / w, target_h / h)
        new_w = max(target_w, int(w * scale))
        new_h = max(target_h, int(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        start_x = new_w // 2 - target_w // 2
        start_y = new_h // 2 - target_h // 2

        return resized[start_y:start_y+target_h, start_x:start_x+target_w]

    def _load_masks_from_paths(self, paths, label="masks") -> torch.Tensor:
        """
        Load grayscale masks as bool, resized like the RGB images.

        :param paths: list[Path] - grayscale mask image paths
        :param label: str - progress label
        :return: torch.Tensor - [N, 1, H, W] bool
        """
        print(f"Loading {label} into memory ({len(paths)} files)...")
        masks = []
        for mask_path in paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # nearest keeps masks binary
            mask = self._resize(mask, self._mask_resolution, interpolation=cv2.INTER_NEAREST)
            mask = mask > 127
            masks.append(mask)
        return torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1)  # [n, 1, h, w] bool

    def _load_all_masks(self) -> None:
        """Load positive masks and, if trajectory files exist, trajectory masks into memory."""
        self.masks = self._load_masks_from_paths(self.mask_files, "positive masks")

        if self.traj_files is not None:
            self.traj_masks = self._load_masks_from_paths(self.traj_files, "trajectory masks")
        else:
            self.traj_masks = None

    def __len__(self) -> int:
        """
        Get the number of embedding frames.

        :return: int - number of frames
        """
        return self._emb_data.shape[0]

    def __getitem__(self, idx) -> dict:
        """
        Get one frame.

        :param idx: int - frame index
        :return: dict - 'embedding', 'mask', 'valid_mask' [1, H, W] bool (False = own robot, ignored), 'cls_token',
            'rgb_path' and 'sensor' (str, '' when unknown); plus 'depth_embedding', 'trajectory_mask',
            'traversability_costs' [R, K] and 'semantic_map' (cached values divided by 255) when loaded
        """
        # copy() detaches from the memmap
        item = {
            'embedding': torch.from_numpy(self._emb_data[idx].copy()).float(),
            'mask': self.masks[idx].float(),
            'valid_mask': self._validity.get(idx),  # [1, h, w] bool, false = ego/ignored
            'cls_token': torch.from_numpy(self._cls_data[idx].copy()).float(),
            'rgb_path': str(self.rgb_files[idx]),
            'sensor': self.sensors[idx],
        }

        if self._depth_emb_data is not None:
            item['depth_embedding'] = torch.from_numpy(self._depth_emb_data[idx].copy()).float()

        if self.traj_masks is not None:
            item['trajectory_mask'] = self.traj_masks[idx].float()

        if self._costs is not None:
            item['traversability_costs'] = torch.from_numpy(self._costs[idx])  # [r, k]

        if self._sem_mmap is not None:
            item['semantic_map'] = torch.from_numpy(self._sem_mmap[idx].copy()).float() / 255.0

        return item