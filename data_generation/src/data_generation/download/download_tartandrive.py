import argparse
import os
import tarfile
import urllib.request

import yaml

from data_generation.configuration import resolve_config

def select_required_files(file_map, bag_name, required_components) -> list[str]:
    """
    Select the manifest path of every required component of one trajectory.

    A component counts as present only if its path below the trajectory directory equals the component name.

    :param file_map: dict - parsed `_tartandrive.yaml`
    :param bag_name: str - trajectory name
    :param required_components: list[str] - component paths below the trajectory directory
    :return: list[str] - manifest paths, one per required component in that order
    :raises ValueError: If the trajectory is not in the manifest, or the manifest lacks any required component
    """
    kitti = file_map.get("kitti") or {}
    full_bag_key = f"kitti/all_topics/{bag_name}/"
    if full_bag_key not in kitti:
        raise ValueError(f"Bag {bag_name} not found in TartanDrive assets (kitti format).")

    listed = []

    def collect_files(obj) -> None:
        """Append paths from nested metadata."""
        if isinstance(obj, list):
            listed.extend(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                collect_files(value)

    for key, value in kitti.items():
        if key.startswith(full_bag_key):
            collect_files(value)

    by_component = {path.removeprefix(full_bag_key): path for path in listed if path.startswith(full_bag_key)}
    missing = [component for component in required_components if component not in by_component]
    if missing:
        raise ValueError(
            f"TartanDrive bag {bag_name} is incomplete in the download manifest, missing required components: "
            f"{', '.join(missing)}. Nothing was downloaded; choose a complete trajectory."
        )
    return [by_component[component] for component in required_components]

def download_tartandrive(bag_name, progress_callback=None, config=None) -> None:
    """
    Download and extract the required files of one TartanDrive trajectory.

    Every required component must be listed in the manifest before the output directory or any marker is touched.
    The `.download_complete` marker is cleared first, written only after every file is downloaded and extracted, and cleared again on failure.

    :param bag_name: str - trajectory name
    :param progress_callback: Callable[[float, str], None] | None - receives percent and status text
    :param config: dict | None - generation configuration, None = default
    :raises ValueError: If the trajectory is not in the manifest or the manifest lacks a required component
    """
    required_components = resolve_config(config)["downloads"]["tartandrive"]["required_components"]
    base_url = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2/"
    yaml_path = os.path.join("assets", "metadata", "_tartandrive.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        file_map = yaml.safe_load(f)

    files_to_download = select_required_files(file_map, bag_name, required_components)

    bag_dir = os.path.join("dataset_raw", "tartandrive", "bags", bag_name)
    os.makedirs(bag_dir, exist_ok=True)
    marker = os.path.join(bag_dir, ".download_complete")
    if os.path.exists(marker):
        os.remove(marker)
    try:
        for i, file_path in enumerate(files_to_download):
            url = base_url + file_path
            rel_path = file_path.replace("kitti/all_topics/", "")
            out_path = os.path.join("dataset_raw", "tartandrive", "bags", rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            overall = (i * 100.0) / len(files_to_download)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response, open(out_path, "wb") as out_file:
                    total_length = int(response.info().get("Content-Length") or 0)
                    downloaded = 0
                    while data := response.read(5 * 1024 * 1024):
                        out_file.write(data)
                        downloaded += len(data)
                        if progress_callback:
                            pct = downloaded / total_length * 100 if total_length else 0
                            overall = min(100.0, (i * 100 + pct) / len(files_to_download))
                            progress_callback(overall, f"File {i + 1}/{len(files_to_download)} ({os.path.basename(file_path)}): {downloaded / 1048576:.1f} MB")

                if out_path.endswith(".tar"):
                    if progress_callback:
                        progress_callback(overall, f"Extracting {os.path.basename(out_path)}...")
                    extract_dir = os.path.join(os.path.dirname(out_path), os.path.basename(out_path).removesuffix(".tar"))
                    os.makedirs(extract_dir, exist_ok=True)
                    with tarfile.open(out_path, "r") as archive:
                        archive.extractall(path=extract_dir)
                    os.remove(out_path)
            except Exception as exc:
                raise RuntimeError(f"Error downloading or extracting {file_path}: {exc}") from exc

        with open(marker, "w", encoding="utf-8") as complete:
            complete.write("done\n")
    except Exception:
        if os.path.exists(marker):
            os.remove(marker)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one TartanDrive trajectory.")
    parser.add_argument("bag_name", help="TartanDrive trajectory name")
    args = parser.parse_args()

    def cli_progress(percent, text) -> None:
        print(f"[{percent:3.0f}%] {text}", end="\r")

    download_tartandrive(args.bag_name, cli_progress)
    print()
