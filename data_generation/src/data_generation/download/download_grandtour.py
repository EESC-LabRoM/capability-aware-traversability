import argparse
import os
import tarfile
import urllib.request

from huggingface_hub import list_repo_files

def download_grandtour(mission, progress_callback=None) -> None:
    """
    Download and extract the selected GrandTour mission.

    The `.download_complete` marker is cleared first, written only after every archive is extracted, and cleared again on failure.

    :param mission: str - mission name
    :param progress_callback: Callable[[float, str], None] | None - receives percent and status text
    """
    if progress_callback:
        progress_callback(0, "Fetching file list from HuggingFace...")

    repo_id = "leggedrobotics/grand_tour_dataset"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    # data/*.tar hold the zarr metadata (timestamps, calibration, odometry), images/*.tar the frames
    required_paths = (
        f"{mission}/data/hdr_front.tar",
        f"{mission}/data/anymal_state_odometry.tar",
        f"{mission}/data/depth_camera_front_upper.tar",
        f"{mission}/data/tf.tar",
        f"{mission}/images/hdr_front.tar",
        f"{mission}/images/depth_camera_front_upper.tar",
    )
    metadata_paths = (
        f"{mission}/data/.zgroup",
    )
    available_files = set(all_files)
    missing_paths = [path for path in required_paths if path not in available_files]
    if missing_paths:
        raise RuntimeError(
            f"Mission {mission} is missing required GrandTour archives: "
            + ", ".join(missing_paths)
        )
    files_to_download = list(required_paths)
    files_to_download.extend(path for path in metadata_paths if path in available_files)

    target_dir = os.path.join("dataset_raw", "grandtour")
    mission_dir = os.path.join(target_dir, mission)
    os.makedirs(mission_dir, exist_ok=True)
    marker = os.path.join(mission_dir, ".download_complete")
    if os.path.exists(marker):
        os.remove(marker)
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"

    try:
        for i, file_path in enumerate(files_to_download):
            url = base_url + file_path
            out_path = os.path.join(target_dir, file_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response, open(out_path, "wb") as out_file:
                    total_length = int(response.info().get("Content-Length") or 0)
                    downloaded = 0
                    while data := response.read(5 * 1024 * 1024):
                        out_file.write(data)
                        downloaded += len(data)
                        if progress_callback:
                            file_percent = downloaded / total_length * 100 if total_length else 0
                            overall = min(100.0, (i * 100 + file_percent) / len(files_to_download))
                            progress_callback(overall, f"File {i + 1}/{len(files_to_download)} ({os.path.basename(file_path)}): {downloaded / 1048576:.1f} MB")
            except Exception as exc:
                raise RuntimeError(f"Error downloading {file_path}: {exc}") from exc

            if out_path.endswith(".tar"):
                try:
                    with tarfile.open(out_path, "r") as archive:
                        archive.extractall(path=os.path.dirname(out_path))
                    os.remove(out_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to extract {out_path}: {exc}") from exc

        with open(marker, "w", encoding="utf-8") as complete:
            complete.write("done\n")
    except Exception:
        if os.path.exists(marker):
            os.remove(marker)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one GrandTour mission.")
    parser.add_argument("mission", help="GrandTour mission name")
    args = parser.parse_args()

    def cli_progress(percent, text) -> None:
        print(f"[{percent:3.0f}%] {text}", end="\r")

    download_grandtour(args.mission, cli_progress)
    print()