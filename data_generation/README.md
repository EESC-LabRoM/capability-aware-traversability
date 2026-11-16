# CAT Dataset and Label Generation

Downloads robot datasets, projects trajectories into the camera, and generates dense positive masks with Grounding DINO + SAM 2 (optionally SegFormer). For the method and training, see the [main README](../README.md).

## ⚙️ Setup

Requires Linux, [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python 3.11/3.12), and, for labeling, an NVIDIA GPU with ≥12 GB VRAM. Run all commands from `data_generation/`.

```bash
cd data_generation
make setup          # downloads and raw processing
make setup-models   # + Grounding DINO, SAM 2, SegFormer (PyTorch)
make run            # launch the terminal UI
```

The TUI menu covers every step: **Download Datasets**, **Process Raw Dataset**, **Grounding SAM Segmentation** (with optional human-in-the-loop correction), **SegFormer Label Model**, **Export CAT Dataset**, and **Manage Local Datasets**. The commands below are the command-line equivalents.

## 📦 Datasets

| Dataset | Robot | Raw data | Processed sequence | Robot width |
| :--- | :--- | :--- | :--- | :--- |
| [TartanDrive 2.0](https://arxiv.org/abs/2402.01913) | Yamaha Viking ATV | `dataset_raw/tartandrive/bags/<bag>/` | `dataset_processed/dataset_tartan_<bag>/` | 1.80 m |
| [GrandTour](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) | ANYmal D | `dataset_raw/grandtour/<mission>/` | `dataset_processed/dataset_grandtour_<mission>/` | 0.50 m |
| [WayFAST / WayFASTER](https://huggingface.co/datasets/CAT-IROS-2026/wayfast_ts_dataset/tree/main) | TerraSentia | — | `dataset_processed/dataset_ts_<session>/` | Configured in YAML |

```bash
# TartanDrive 2.0
uv run --locked --no-sync python -m data_generation.download.download_tartandrive <bag>
uv run --locked --no-sync python -m data_generation.dataset_creation.create_dataset_tartan \
  --dataset dataset_raw/tartandrive/bags/<bag> --robot_width 1.80

# GrandTour
uv run --locked --no-sync python -m data_generation.download.download_grandtour <mission>
uv run --locked --no-sync python -m data_generation.dataset_creation.create_dataset_grand_tour \
  --dataset dataset_raw/grandtour/<mission> --robot_width 0.50

# WayFAST / WayFASTER: list sessions, then download one
uv run --locked --no-sync python -m data_generation.download.download_wayfast
uv run --locked --no-sync python -m data_generation.download.download_wayfast dataset_ts_2022_04_21_16h07m26s.zip
```

### Your own robot

Record a ROS1 `.bag` while driving your robot and process it with `create_dataset_ts`:

| Stream | Message type | Required for |
| :--- | :--- | :--- |
| RGB | `sensor_msgs/Image` or `CompressedImage` | Always |
| Odometry | `nav_msgs/Odometry` | Always |
| Depth | `sensor_msgs/Image` or `CompressedImage` | Trajectory masks |
| Calibration | `sensor_msgs/CameraInfo` | Trajectory masks |
| TF, robot base → camera | `/tf`, `/tf_static` | Trajectory masks |

Set `bag_processing.ros_topics` and `bag_processing.robot.width` in `configs/config.yaml`. A supplied `--robot_width` overrides the configured width. Trajectory masks require compatible depth, calibration, and TF data. TartanDrive depth requirements are given in the Datasets section above.

Name the bag `<sensor>_<name>.bag` and add `<sensor>` to `data.depth_clip_by_sensor` in `cat/configs/train.yaml`.

```bash
uv run --locked --no-sync python -m data_generation.dataset_creation.create_dataset_ts \
  --config configs/config.yaml --dataset /path/to/ts_recording.bag
```

Bags under `dataset_raw/wayfaster/bags/` also appear in the TUI.

## 🏷️ Labels

Keep the `dataset_<family>_...` directory name: it selects family-specific settings, including the ego mask.

```bash
DATASET=dataset_tartan_sequence
DATASET_PATH=/absolute/path/to/dataset_processed/$DATASET
MODEL=/absolute/path/to/segformer_checkpoint
uv run --locked --no-sync python -m data_generation.segmentation.gdino_sam_segment \
  --dataset "$DATASET_PATH" --hil false --resume false --batch_size 12 --step_positive 6
uv run --locked --no-sync python -m data_generation.segmentation.train_segformer \
  --dataset "$DATASET" --epochs 2 --batch_size 2 --early_stopping false --output_dir "$MODEL"
uv run --locked --no-sync python -m data_generation.segmentation.inference_segformer \
  --dataset "$DATASET" --model_path "$MODEL"
```

**Ego masks.** TartanDrive images show the ATV body. Segmentation writes `ego_mask.json` and a sequence-local `ego_mask.png` (640×480, white = robot) at the sequence root. Those pixels are ignored by labeling and CAT training, not treated as obstacles.

## 📂 Output

Generated sequences, weights, and checkpoints are stored under this directory.

```text
dataset_processed/
├── cat_data.csv          # Merged CAT training CSV (Handoff to CAT)
├── scores/               # Robot terrain-score YAMLs
└── dataset_name/
    ├── data.csv          # Frame metadata and file references
    ├── rgb/
    ├── depth/
    ├── ego_mask.json     # When this sequence has an ego exclusion mask
    ├── ego_mask.png      # Local mask asset referenced by the manifest
    ├── positive_masks/
    ├── trajectory_masks/
    └── visualization/    # Mask overlays for debugging
```

All settings live in [`configs/config.yaml`](configs/config.yaml). Processors, labeling tools, the CAT export, and the WayFAST downloader accept `--config PATH` with partial overrides. `data_format.ego_manifest_name` must match `data.ego_manifest_name` in `cat/configs/train.yaml`.

## 🤝 Handoff to CAT

CAT trains on a single CSV. **Export CAT Dataset** in the TUI, or the command below, merges labeled sequences into `dataset_processed/cat_data.csv`:

```bash
uv run --locked --no-sync python -m data_generation.export.cat_csv \
  --dataset dataset_tartan_<bag> dataset_grandtour_<mission>   # default: every sequence
```

Each sequence needs positive and trajectory masks and a score YAML in `dataset_processed/scores/`, named `<sequence>.yaml` or `results-<sequence>-<YYYYMMDD>-<HHMMSS>.yaml`. `--fallback_scores FILE` covers sequences without one. The export checks every input and writes nothing on failure.

A score YAML lists one `score` per CLIPSeg prompt, in the order of `prompts` in [`cat/configs/train.yaml`](../cat/configs/train.yaml), for each robot in `robots.name_map`:

```yaml
"TerraSentia (EarthSense)":
  - score: 0.90   # grass
  - score: 1.00   # pavement
  # ... 10 entries by default
"ANYmal-D (ANYbotics)":
  - score: 0.95
  # ...
```

Paths in the CSV are relative to `dataset_processed/`. `source_bag` is the sequence name, whose second `_` token selects the depth sensor (`grandtour`, `ts`, `tartan`, `terrasentia`). CAT reads `dataset_processed/cat_data.csv` by default, so continue with [embedding extraction and training](../README.md#-training).

### Using the bundled scores

The LLM scores are included in [`scores/`](scores). To use them for a bag, select its processed dataset and point the export to that folder. Run from `data_generation/`:

```bash
uv run --locked --no-sync python -m data_generation.export.cat_csv \
  --dataset dataset_terrasentia_2023-11-21-21-13-31 \
  --scores_dir scores
```

The exporter matches the bag name to its score file:

```text
dataset_terrasentia_2023-11-21-21-13-31
  → scores/results-dataset_terrasentia_2023-11-21-21-13-31-20260225-194348.yaml
```

Replace `--dataset` with your bag's processed directory name. You can select several bags in the same command; each gets its matching YAML. The exported CSV stores the assignment in `scores_yaml` for every frame.

For your own scores, save `<dataset_name>.yaml` in the score folder, keeping the robot names and terrain order shown above.

To reuse all saved bag-to-score assignments, including shared scores, replace `--scores_dir scores` with `--scores_manifest scores/manifest.json`.

Then continue with [embedding extraction and training](../README.md#-training). Re-extract embeddings if you change score-file paths.

