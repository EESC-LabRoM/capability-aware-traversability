import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from data_generation.configuration import configuration_path, load_config, resolve_config
from data_generation.dataset_creation.preflight import validate_processed_input


def processed_root(config=None) -> Path:
    """
    Get the directory holding the processed sequences.

    :param config: dict | None - generation config, None loads the default
    :return: Path - processed dataset root
    """
    return Path(resolve_config(config)["bag_processing"]["output_dataset_dir"])

def next_step(root, output, config=None) -> str:
    """
    Describe the next CAT step, with extraction paths when CAT defaults do not apply.

    :param root: str | Path - processed root the CSV paths are relative to
    :param output: str | Path - written CAT CSV
    :param config: dict | None - generation config, None loads the default
    :return: str - instructions for the user
    """
    config = resolve_config(config)
    training_csv = (configuration_path().parents[1] / config["cat_export"]["training_csv"]).resolve()
    step = "Next: extract embeddings and train CAT (main README, Training)."
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == training_csv.parent and output == training_csv:
        return step
    return (f"{step}\nNon-default location: pass --data_folder {root} --data_csv {output} "
            "to cat/scripts/generate_embeddings.py.")

def default_scores_dir(config=None) -> Path:
    """
    Get the default terrain-score YAML directory.

    :param config: dict | None - generation config, None loads the default
    :return: Path - score directory inside the processed root
    """
    config = resolve_config(config)
    return processed_root(config) / config["data_format"]["scores_dir"]

def find_scores(scores_dir, sequence) -> Path | None:
    """
    Find the score YAML of a sequence (`<sequence>.yaml` or the newest `results-<sequence>-<date>-<time>.yaml`).

    :param scores_dir: str | Path - directory of score YAMLs
    :param sequence: str - sequence directory name
    :return: Path | None - score YAML, None if the sequence has none
    """
    scores_dir = Path(scores_dir)
    if not scores_dir.is_dir():
        return None
    pattern = re.compile(rf"(?:{re.escape(sequence)}|results-{re.escape(sequence)}-\d{{8}}-\d{{6}})\.yaml")
    matches = sorted(p for p in scores_dir.iterdir() if p.is_file() and pattern.fullmatch(p.name))
    return matches[-1] if matches else None

def validate_scores(path) -> None:
    """
    Check a score YAML maps each robot to equal-length scores in [0, 1]; raise ValueError otherwise.

    :param path: str | Path - score YAML
    """
    try:
        with open(path) as source:
            scores = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read score YAML {path}: {exc}") from exc
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"Score YAML {path} must map robot names to score lists")
    lengths = set()
    for robot, entries in scores.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Score YAML {path}: '{robot}' must list one entry per terrain prompt")
        for entry in entries:
            value = entry.get("score") if isinstance(entry, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Score YAML {path}: '{robot}' has an entry without a score in [0, 1]: {entry}")
        lengths.add(len(entries))
    if len(lengths) > 1:
        raise ValueError(f"Score YAML {path}: robots list different numbers of scores {sorted(lengths)}")

def read_sequence(root, sequence, config) -> tuple[list[str], list[dict]]:
    """
    Read a labeled sequence's CSV rows; raise ValueError if files, labels or trajectory masks are missing.

    :param root: Path - processed dataset root
    :param sequence: str - sequence directory name
    :param config: dict - generation config
    :return: tuple[list[str], list[dict]] - CSV column names and rows
    """
    sequence_dir = root / sequence
    if not sequence_dir.is_dir():
        raise ValueError(f"Sequence not found: {sequence_dir}")
    try:
        validate_processed_input(sequence_dir, require_masks=True, config=config)
    except ValueError as exc:
        raise ValueError(f"{sequence}: {exc}") from exc
    with (sequence_dir / "data.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "trajectory_img_name" not in fields or any(not row["trajectory_img_name"] for row in rows):
        raise ValueError(f"{sequence}: every frame needs a trajectory mask (trajectory_img_name)")
    return fields, rows

def list_sequences(root) -> list[str]:
    """
    List processed sequence directories.

    :param root: str | Path - processed dataset root
    :return: list[str] - sorted `dataset_*` names, empty if the root is missing
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("dataset_"))

def manifest_scores(path, sequences) -> dict[str, Path]:
    """
    Resolve sequence assignments and verify the selected score files.

    :param path: str | Path - score manifest, with file paths relative to its directory
    :param sequences: list[str] - selected sequence names
    :return: dict[str, Path] - verified score paths by sequence
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["version"] != 1:
        raise ValueError(f"Unsupported score manifest version: {manifest['version']}")
    missing = [sequence for sequence in sequences if sequence not in manifest["sequences"]]
    if missing:
        raise ValueError(f"No score manifest assignment for: {', '.join(missing)}")
    scores = {}
    verified = set()
    for sequence in sequences:
        relative = manifest["sequences"][sequence]
        score_path = (path.parent / relative).resolve()
        if not score_path.is_relative_to(path.parent.resolve()):
            raise ValueError(f"Score path escapes manifest directory: {relative}")
        if relative not in verified:
            digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
            if digest != manifest["files"][relative]:
                raise ValueError(f"Score checksum mismatch: {score_path}")
            raw = yaml.safe_load(score_path.read_text())
            if set(raw) != set(manifest["robot_names"]):
                raise ValueError(f"Score robot profiles do not match manifest: {score_path}")
            for robot, entries in raw.items():
                if [entry["category"] for entry in entries] != manifest["category_order"]:
                    raise ValueError(f"Score category order mismatch: {score_path}, {robot}")
            verified.add(relative)
        scores[sequence] = score_path
    return scores


def export_cat_csv(sequences, output=None, scores_dir=None, fallback_scores=None, config=None, scores_manifest=None) -> int:
    """
    Merge labeled sequences into one CAT training CSV; raise ValueError, writing nothing, on invalid input.

    :param sequences: list[str] - sequence directory names under the processed root
    :param output: str | Path | None - output CSV, None uses `data_format.cat_csv_name` in the root
    :param scores_dir: str | Path | None - score YAML directory, None uses `data_format.scores_dir` in the root
    :param fallback_scores: str | Path | None - score YAML for sequences without their own
    :param config: dict | None - generation config, None loads the default
    :param scores_manifest: str | Path | None - explicit sequence assignments and score checksums
    :return: int - number of exported frames
    """
    config = resolve_config(config)
    export_config = config["cat_export"]
    path_columns = export_config["path_columns"]
    sequence_column = export_config["sequence_column"]
    scores_column = export_config["scores_column"]
    export_columns = (sequence_column, scores_column)
    root = processed_root(config)
    output = Path(output) if output else root / config["data_format"]["cat_csv_name"]
    if not sequences:
        raise ValueError("No sequences selected")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Sequences are listed more than once")

    if scores_manifest is not None and (scores_dir is not None or fallback_scores is not None):
        raise ValueError("--scores_manifest cannot be combined with --scores_dir or --fallback_scores")
    scores_dir = Path(scores_dir) if scores_dir else default_scores_dir(config)
    score_files = manifest_scores(scores_manifest, sequences) if scores_manifest is not None else {
        s: find_scores(scores_dir, s) or (Path(fallback_scores) if fallback_scores else None) for s in sequences
    }
    missing = [s for s, path in score_files.items() if path is None]
    if missing:
        raise ValueError(
            f"No score YAML in {scores_dir} for {len(missing)} sequence(s): {', '.join(missing)}. "
            "Add <sequence>.yaml files or pass --fallback_scores."
        )
    for path in set(score_files.values()):
        validate_scores(path)

    fields = list(path_columns)
    merged = []
    for index, sequence in enumerate(sequences, start=1):
        sequence_fields, rows = read_sequence(root, sequence, config)
        fields.extend(f for f in sequence_fields if f not in fields and f not in export_columns)
        # cat joins scores_yaml onto data_dir, absolute paths survive that join
        score_path = score_files[sequence].resolve()
        scores = score_path.relative_to(root.resolve()).as_posix() if score_path.is_relative_to(root.resolve()) else str(score_path)
        for row in rows:
            for column in path_columns:
                if row.get(column):
                    row[column] = f"{sequence}/{row[column]}"
            row[sequence_column] = sequence
            row[scores_column] = scores
            merged.append(row)
        print(f"Exported sequence {index}/{len(sequences)}: {sequence} ({len(rows)} frames)", flush=True)
    fields.extend(export_columns)

    # atomic write: no partial csv on failure
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", dir=output.parent, suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(merged)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(merged)

def main(argv=None) -> None:
    """
    Export the CAT training CSV from CLI arguments, exiting with status 1 on failure.

    :param argv: list[str] | None - arguments, None uses sys.argv
    """
    parser = argparse.ArgumentParser(description="Merge labeled sequences into the CSV that CAT trains on.")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Sequence names in the processed root (default: every dataset_* directory)")
    parser.add_argument("--scores_dir", default=None, help="Score YAML directory (default: <processed root>/scores)")
    parser.add_argument("--fallback_scores", default=None, help="Score YAML for sequences without their own")
    parser.add_argument("--scores_manifest", default=None, help="Score manifest with explicit sequence assignments and checksums")
    parser.add_argument("--output", default=None, help="Output CSV (default: <processed root>/cat_data.csv)")
    parser.add_argument("--config", default=None, help="YAML overrides for the generation configuration")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = processed_root(config)
    sequences = args.dataset or list_sequences(root)
    output = Path(args.output) if args.output else root / config["data_format"]["cat_csv_name"]
    frames = export_cat_csv(sequences, output, args.scores_dir, args.fallback_scores, config, args.scores_manifest)
    print(f"Wrote {frames} frames to {output}", flush=True)
    print(next_step(root, output, config), flush=True)

if __name__ == "__main__":
    main()
