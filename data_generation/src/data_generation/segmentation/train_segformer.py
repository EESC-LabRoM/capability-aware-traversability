import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import argparse
import csv
import math
import os
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.feature_extraction_utils import BatchFeature
import transformers
from torch.utils.data import ConcatDataset

from data_generation.configuration import load_config, resolve_config
from data_generation.utils.ego_mask import load_ego_mask
import warnings

class EarlyStopping:
    """
    Stops training once validation loss stops improving.
    """
    def __init__(self, patience=3, min_delta=0) -> None:
        """
        Set up early stopping.

        :param patience: int - non-improving checks before stopping
        :param min_delta: float - minimum decrease counted as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss) -> None:
        """
        Record a validation loss; set early_stop when patience runs out.

        :param val_loss: float - latest validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"\nEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def build_label_map(mask_np, ego_mask, ignore_index) -> np.ndarray:
    """
    Convert a 0/255 positive mask to 0/1 labels, with ignore_index on ego pixels.

    :param mask_np: np.ndarray - (H, W) grayscale positive mask, 0/255
    :param ego_mask: np.ndarray | None - (H, W) bool ego mask, True = ignored
    :param ignore_index: int - label assigned to ego pixels
    :return: np.ndarray - (H, W) int64 labels with values 0, 1 or ignore_index
    """
    labels = np.where(mask_np > 128, 1, 0).astype(np.int64)
    if ego_mask is not None:
        if ego_mask.shape != labels.shape:
            raise ValueError(f"Ego mask shape {ego_mask.shape} does not match positive mask shape {labels.shape}")
        labels[ego_mask] = ignore_index
    return labels

def semantic_loss(logits, labels, ignore_index) -> torch.Tensor:
    """
    Compute cross-entropy over supervised pixels only.

    :param logits: torch.Tensor - (B, C, h, w) raw class scores
    :param labels: torch.Tensor - (B, H, W) int64 class ids or ignore_index
    :param ignore_index: int - label excluded from the loss
    :return: torch.Tensor - scalar mean loss over non-ignored pixels
    """
    upsampled = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    per_pixel = nn.functional.cross_entropy(upsampled, labels, ignore_index=ignore_index, reduction="sum")
    valid = (labels != ignore_index).sum().clamp(min=1)
    return per_pixel / valid

def supervised_pixel_count(labels, ignore_index) -> int:
    """
    Count non-ignored label pixels.

    :param labels: torch.Tensor - class ids or ignore_index
    :param ignore_index: int - ignored label
    :return: int - pixels not equal to ignore_index
    """
    return int((labels != ignore_index).sum().item())

def _summarize(names, limit=5) -> str:
    """
    Join up to `limit` sorted names for error messages.

    :param names: Iterable[str] - offending names
    :param limit: int - maximum names listed
    :return: str - comma-separated names with a "(+N more)" suffix when truncated
    """
    names = sorted(names)
    text = ", ".join(names[:limit])
    return text + (f", ... (+{len(names) - limit} more)" if len(names) > limit else "")

def _list_files(directory, extensions) -> list[str]:
    """
    List files ending in one of the extensions (case-insensitive); raises ValueError if the directory is missing.

    :param directory: str - directory to list
    :param extensions: list[str] | tuple[str, ...] - accepted lowercase suffixes
    :return: list[str] - sorted matching file names
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(tuple(extensions)))

def _index_by_stem(names, directory, kind, strip_prefix="") -> dict[str, str]:
    """
    Map frame identity to file name; raises ValueError on a name without the prefix or two files for one frame.

    :param names: Iterable[str] - file names to index
    :param directory: str - directory, for error messages
    :param kind: str - file kind, for error messages
    :param strip_prefix: str - prefix removed from each stem to get the frame identity
    :return: dict[str, str] - frame identity to file name
    """
    index = {}
    for name in names:
        stem = os.path.splitext(name)[0]
        if strip_prefix:
            if not stem.startswith(strip_prefix) or len(stem) == len(strip_prefix):
                raise ValueError(
                    f"{kind} '{name}' in {directory} is not named {strip_prefix}<rgb-stem>; "
                    "cannot determine which frame it labels")
            stem = stem[len(strip_prefix):]
        if stem in index:
            raise ValueError(f"Duplicate {kind} for frame '{stem}' in {directory}: {index[stem]} and {name}")
        index[stem] = name
    return index

def pair_frames_by_name(image_dir, mask_dir, config) -> list[tuple[str, str, str]]:
    """
    Pair every RGB frame with its `positive_<rgb-stem>.<ext>` mask; raises ValueError unless one to one.

    :param image_dir: str - directory of RGB frames
    :param mask_dir: str - directory of positive 0/255 masks
    :param config: dict - generation config
    :return: list[tuple[str, str, str]] - (frame_id, image_path, mask_path) sorted by frame_id
    """
    images = _index_by_stem(_list_files(image_dir, config["segformer"]["image_extensions"]), image_dir, "RGB frame")
    masks = _index_by_stem(
        _list_files(mask_dir, config["segformer"]["mask_extensions"]), mask_dir, "mask",
        config["data_format"]["positive_mask_prefix"])
    if not images:
        raise ValueError(f"No RGB frames found in {image_dir}")
    missing = images.keys() - masks.keys()
    orphaned = masks.keys() - images.keys()
    if missing or orphaned:
        parts = []
        if missing:
            parts.append(f"{len(missing)} RGB frames without a mask ({_summarize(missing)})")
        if orphaned:
            parts.append(f"{len(orphaned)} masks without an RGB frame ({_summarize(orphaned)})")
        raise ValueError(f"Incomplete labels in {mask_dir}: " + "; ".join(parts))
    return [(s, os.path.join(image_dir, images[s]), os.path.join(mask_dir, masks[s])) for s in sorted(images)]

def pair_frames_from_csv(dataset_dir, image_dir, mask_dir, config) -> list[tuple[str, str, str]]:
    """
    Pair frames using the dataset's data.csv.

    :param dataset_dir: str - dataset root holding data.csv
    :param image_dir: str - directory of RGB frames
    :param mask_dir: str - positive mask directory, used without a positive_mask_name column
    :param config: dict - generation config
    :return: list[tuple[str, str, str]] - (frame_id, image_path, mask_path) in csv row order
    """
    csv_path = os.path.join(dataset_dir, "data.csv")
    with open(csv_path, newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if "rgb_img_name" not in fields:
            raise ValueError(f"{csv_path}: missing rgb_img_name column")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path}: no frames listed")

    pairs, seen_images, seen_masks, missing = [], {}, {}, []
    for line, row in enumerate(rows, start=2):
        rgb_ref = (row.get("rgb_img_name") or "").strip()
        if not rgb_ref:
            raise ValueError(f"{csv_path} line {line}: empty rgb_img_name")
        frame = os.path.splitext(os.path.basename(rgb_ref))[0]
        if "positive_mask_name" in fields:
            mask_ref = (row.get("positive_mask_name") or "").strip()
            if not mask_ref:
                raise ValueError(f"{csv_path} line {line}: no positive_mask_name for {rgb_ref}; labeling is incomplete")
            mask_path = os.path.normpath(os.path.join(dataset_dir, mask_ref))
        else:
            mask_ref = f"{config['data_format']['positive_mask_prefix']}{frame}.tif"
            mask_path = os.path.normpath(os.path.join(mask_dir, mask_ref))
        image_path = os.path.normpath(os.path.join(dataset_dir, rgb_ref))
        if image_path in seen_images:
            raise ValueError(f"{csv_path} line {line}: {rgb_ref} already listed on line {seen_images[image_path]}")
        if mask_path in seen_masks:
            raise ValueError(f"{csv_path} line {line}: mask {mask_ref} already used by line {seen_masks[mask_path]}")
        seen_images[image_path], seen_masks[mask_path] = line, line
        if not os.path.isfile(image_path):
            raise ValueError(f"{csv_path} line {line}: RGB frame {rgb_ref} not found")
        if not os.path.isfile(mask_path):
            missing.append(mask_ref)
        pairs.append((frame, image_path, mask_path))
    if missing:
        raise ValueError(f"Incomplete labels in {dataset_dir}: {len(missing)} missing masks ({_summarize(missing)})")

    listed = {os.path.realpath(p) for _, p, _ in pairs}
    unlisted = [f for f in _list_files(image_dir, config["segformer"]["image_extensions"])
                if os.path.realpath(os.path.join(image_dir, f)) not in listed]
    if unlisted:
        raise ValueError(f"{len(unlisted)} RGB frames in {image_dir} are not listed in {csv_path} ({_summarize(unlisted)})")
    return pairs

class TraversabilityDataset(Dataset):
    """
    Dataset of RGB frames and positive masks; ego pixels are labelled with ignore_index.
    """
    def __init__(self, image_dir, mask_dir, processor, dataset_dir=None, config=None) -> None:
        """
        Pair RGB frames with positive masks and load the ego mask.

        :param image_dir: str - directory of RGB frames
        :param mask_dir: str - directory of positive 0/255 masks
        :param processor: SegformerImageProcessor | None - image processor, may be attached later
        :param dataset_dir: str | None - dataset root, defaults to the parent of image_dir
        :param config: dict | None - generation config, None loads the default
        """
        config = resolve_config(config)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        root = dataset_dir or os.path.dirname(os.path.abspath(image_dir))
        self.ego = load_ego_mask(root, config=config)
        self.ignore_index = config["segformer"]["ignore_index"]
        if os.path.isfile(os.path.join(root, "data.csv")):
            self.samples = pair_frames_from_csv(root, image_dir, mask_dir, config)
        else:
            self.samples = pair_frames_by_name(image_dir, mask_dir, config)
        empty = [os.path.basename(p) for _, i, m in self.samples for p in (i, m) if os.path.getsize(p) == 0]
        if empty:
            raise ValueError(f"Empty files in {root}: {_summarize(empty)}")

    def __len__(self) -> int:
        """
        Return the sample count.

        :return: int - sample count
        """
        return len(self.samples)

    def _check_geometry(self, frame, image_size, mask_size) -> None:
        """
        Raise ValueError if the RGB size differs from the ego mask or the positive mask.

        :param frame: str - frame identity for error messages
        :param image_size: tuple[int, int] - PIL (width, height) of the RGB before any processor resize
        :param mask_size: tuple[int, int] - PIL (width, height) of the positive mask
        """
        image_shape = (image_size[1], image_size[0])
        if self.ego.mask is not None and image_shape != self.ego.mask.shape:
            raise ValueError(f"Ego mask shape {self.ego.mask.shape} does not match image {frame} shape {image_shape}")
        if image_size != mask_size:
            raise ValueError(
                f"RGB/mask size mismatch for frame {frame}: RGB {image_size[0]}x{image_size[1]} "
                f"vs mask {mask_size[0]}x{mask_size[1]}")

    def validate_geometry(self) -> None:
        """
        Check every sample's RGB, mask and ego geometry from file headers; raises ValueError if unreadable or mismatched.
        """
        for frame, image_path, mask_path in self.samples:
            sizes = []
            for path in (image_path, mask_path):
                try:
                    with Image.open(path) as handle:
                        sizes.append(handle.size)
                except OSError as error:
                    raise ValueError(f"Unreadable file {path}: {error}") from error
            self._check_geometry(frame, *sizes)

    def __getitem__(self, idx) -> BatchFeature:
        """
        Load one frame and its labels through the image processor.

        Raises RuntimeError if no processor is attached, ValueError if a file is unreadable or sizes differ.

        :param idx: int - sample index
        :return: BatchFeature - processor tensors, batch dimension squeezed
        """
        if self.processor is None:
            raise RuntimeError("TraversabilityDataset has no image processor attached")
        frame, image_path, mask_path = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except OSError as error:
            raise ValueError(f"Unreadable frame {frame}: {error}") from error

        self._check_geometry(frame, image.size, mask.size)
        labels = build_label_map(np.array(mask), self.ego.mask, self.ignore_index)

        # image and labels (ego included) share one resize, labels use nearest
        inputs = self.processor(images=image, segmentation_maps=labels, return_tensors="pt", do_reduce_labels=False)
        for k, v in inputs.items():
            inputs[k] = v.squeeze()
        return inputs

def validate_training_args(epochs, batch_size, lr, patience, early_stopping) -> None:
    """
    Raise ValueError if any hyperparameter is out of range.

    :param epochs: int - training epochs
    :param batch_size: int - batch size
    :param lr: float - learning rate
    :param patience: int - early stopping patience
    :param early_stopping: bool - early stopping enabled
    """
    problems = []
    if epochs < 1:
        problems.append(f"epochs must be >= 1 (got {epochs})")
    if batch_size < 1:
        problems.append(f"batch_size must be >= 1 (got {batch_size})")
    if not (lr > 0 and math.isfinite(lr)):
        problems.append(f"lr must be a positive finite number (got {lr})")
    if early_stopping and patience < 1:
        problems.append(f"patience must be >= 1 when early stopping is enabled (got {patience})")
    if problems:
        raise ValueError("Invalid training arguments: " + "; ".join(problems))

def split_sizes(total) -> tuple[int, int]:
    """
    Compute 90/10 train/validation sizes; raises ValueError for fewer than 2 samples.

    :param total: int - number of labeled samples
    :return: tuple[int, int] - train size and validation size
    """
    if total < 2:
        raise ValueError(
            f"Need at least 2 labeled frames for a train/validation split, found {total}")
    train_size = int(0.9 * total)
    return train_size, total - train_size

def main() -> None:
    """
    Train SegFormer on the command-line datasets and save the model and processor.
    """
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings('ignore')

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config = load_config(config_parser.parse_known_args()[0].config)
    training_config = config["segformer"]

    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument("--dataset", nargs="+", required=True, help="List of dataset names in the configured processed dataset directory")
    parser.add_argument("--epochs", type=int, default=training_config["total_epochs"], help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=training_config["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=training_config["learning_rate"], help="Learning rate")
    parser.add_argument("--early_stopping", type=str, default=str(training_config["early_stopping"]).lower(), help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=training_config["patience"], help="Early stopping patience")
    parser.add_argument("--output_dir", default="checkpoints/segformer_finetuned", help="Directory for the trained model and image processor")
    parser.add_argument("--config", default=None, help="YAML overrides for the generation configuration")
    args = parser.parse_args()
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
    model_name = config["segformer"]["model_id"]

    use_early_stopping = args.early_stopping.lower() == "true"
    try:
        validate_training_args(args.epochs, args.batch_size, args.lr, args.patience, use_early_stopping)
    except ValueError as error:
        parser.error(str(error))

    processed_dir = config["bag_processing"]["output_dataset_dir"]
    ignore_index = config["segformer"]["ignore_index"]
    selected_datasets = args.dataset

    # validate every frame before downloading the model or writing output
    datasets_list = []
    for ds_name in selected_datasets:
        ds_path = os.path.join(processed_dir, ds_name)
        rgb_dir = os.path.join(ds_path, "rgb")

        mask_dir = os.path.join(ds_path, "positive_masks")
        if os.path.isdir(rgb_dir) and os.path.isdir(mask_dir):
            dataset = TraversabilityDataset(rgb_dir, mask_dir, None, dataset_dir=ds_path, config=config)
            dataset.validate_geometry()
            print(f"{ds_name}: {len(dataset)} labeled frames; {dataset.ego.description}", flush=True)
            datasets_list.append(dataset)
        else:
            raise FileNotFoundError(f"Requested dataset is incomplete: {ds_path}; both rgb/ and positive_masks/ are required")

    if not datasets_list:
        print("No valid datasets found after checking paths. Exiting.")
        exit(1)

    full_dataset = ConcatDataset(datasets_list)
    try:
        train_size, val_size = split_sizes(len(full_dataset))
    except ValueError as error:
        parser.error(str(error))

    print("Checking and loading base SegFormer model (will download if not present)...")
    processor = SegformerImageProcessor.from_pretrained(model_name, do_reduce_labels=False)
    for dataset in datasets_list:
        dataset.processor = processor
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "untraversable", 1: "traversable"},
        label2id={"untraversable": 0, "traversable": 1},
        semantic_loss_ignore_index=ignore_index,
        use_safetensors=True,
    )
    model.to(device)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")
    best_state = None

    total_epochs = args.epochs
    total_train_steps = len(train_loader)
    total_val_steps = len(val_loader)
    total_steps = total_epochs * (total_train_steps + total_val_steps)
    global_step = 0

    print(f"\nStarting training: {len(train_dataset)} training images, {len(val_dataset)} validation images across {len(selected_datasets)} dataset(s).", flush=True)

    for epoch in range(total_epochs):
        model.train()
        epoch_train_loss = 0
        supervised_train_batches = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            global_step += 1

            if supervised_pixel_count(labels, ignore_index) == 0:
                print(f"Processed frame {global_step}/{total_steps} | Epoch {epoch+1}/{total_epochs} [Train] Skipped: every pixel ignored", flush=True)
                continue

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = semantic_loss(outputs.logits, labels, ignore_index)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            supervised_train_batches += 1
            print(f"Processed frame {global_step}/{total_steps} | Epoch {epoch+1}/{total_epochs} [Train] Loss: {loss.item():.4f}", flush=True)

        if supervised_train_batches == 0:
            raise ValueError("No supervised pixels in the training split; no trained checkpoint can be saved.")

        model.eval()
        epoch_val_loss = 0
        supervised_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                if supervised_pixel_count(labels, ignore_index) == 0:
                    global_step += 1
                    print(f"Processed frame {global_step}/{total_steps} | Epoch {epoch+1}/{total_epochs} [Val] Skipped: every pixel ignored", flush=True)
                    continue
                outputs = model(pixel_values=pixel_values)
                val_loss = semantic_loss(outputs.logits, labels, ignore_index).item()
                epoch_val_loss += val_loss
                supervised_val_batches += 1
                global_step += 1
                print(f"Processed frame {global_step}/{total_steps} | Epoch {epoch+1}/{total_epochs} [Val] Loss: {val_loss:.4f}", flush=True)

        if supervised_val_batches == 0:
            raise ValueError("No supervised pixels in the validation split; validation loss is undefined.")
        avg_train_loss = epoch_train_loss / supervised_train_batches
        avg_val_loss = epoch_val_loss / supervised_val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Processed frame {global_step}/{total_steps} | Epoch {epoch+1}/{total_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

        if use_early_stopping:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                # report total_steps so the ui progress bar completes
                print(f"Processed frame {total_steps}/{total_steps} | Early Stopping criteria met. Halting training.", flush=True)
                break

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if best_state is None:
        raise ValueError("No validation checkpoint was produced; no trained model can be saved.")
    model.load_state_dict(best_state)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Processed frame {total_steps}/{total_steps} | Final model saved successfully at {output_dir}", flush=True)

if __name__ == "__main__":
    main()
