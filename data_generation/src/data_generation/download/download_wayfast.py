from pathlib import Path, PurePosixPath
import argparse
import urllib.request
import zipfile
import tempfile

from huggingface_hub import HfApi, hf_hub_url

from data_generation.configuration import load_config, resolve_config
from data_generation.dataset_creation.preflight import validate_processed_input

def list_datasets(progress_callback=None, config=None) -> dict[str, int]:
    """
    List WayFAST session archives with their sizes.

    :param progress_callback: Callable[[float, str], None] | None - receives percent and status text
    :param config: dict | None - generation configuration, None = default
    :return: dict[str, int] - archive name -> compressed size in bytes
    """
    config = resolve_config(config)
    if progress_callback:
        progress_callback(0, "Fetching WayFAST/WayFASTER sessions from Hugging Face...")
    return {item.path: item.size
            for item in HfApi().list_repo_tree(config["downloads"]["wayfast"]["repository_id"], repo_type="dataset")
            if item.path.endswith(".zip")}

def dataset_path(selected_dataset, config=None) -> Path:
    """
    Return the processed directory of a session archive; raise ValueError for an invalid filename.

    :param selected_dataset: str - session archive filename
    :param config: dict | None - generation configuration, None = default
    :return: Path - destination directory
    """
    name = PurePosixPath(selected_dataset)
    if (len(name.parts) != 1 or name.is_absolute() or "\\" in selected_dataset
            or not selected_dataset.startswith("dataset_") or name.suffix != ".zip"):
        raise ValueError(f"Invalid session archive: {selected_dataset}")
    return Path(resolve_config(config)["bag_processing"]["output_dataset_dir"]) / name.stem

def download_wayfast(selected_dataset, progress_callback=None, config=None) -> None:
    """
    Download, validate and install a processed WayFAST session without overwriting existing files.

    Validates the archive in a staging directory, then renames it into place.
    Raises ValueError for an invalid archive or session, FileExistsError if the destination appears mid-download.

    :param selected_dataset: str - session archive filename
    :param progress_callback: Callable[[float, str], None] | None - receives percent and status text
    :param config: dict | None - generation configuration, None = default
    """
    config = resolve_config(config)
    target = dataset_path(selected_dataset, config)
    processed_dir = target.parent
    if target.exists():
        try:
            validate_processed_input(target, config=config)
        except ValueError as exc:
            raise ValueError(
                f"Existing session is incomplete: {target}. Move it aside before downloading again; "
                "existing user files will not be overwritten."
            ) from exc
        if progress_callback:
            progress_callback(100, "Processed session already installed; existing files preserved.")
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(config["downloads"]["wayfast"]["repository_id"], selected_dataset, repo_type="dataset")
    with tempfile.TemporaryDirectory(prefix=".wayfast-", dir=processed_dir) as temporary:
        staging = Path(temporary)
        archive_path = staging / selected_dataset
        if progress_callback:
            progress_callback(0, f"Downloading {selected_dataset}...")
        with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(5 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(min(100, downloaded * 100 / total) if total else 0,
                                      f"Downloading session: {downloaded / 1024**2:.1f} MB")
        if progress_callback:
            progress_callback(100, f"Extracting {selected_dataset}...")
        staged_target = staging / target.name
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if (member.is_absolute() or ".." in member.parts or "\\" in info.filename
                        or not member.parts or member.parts[0] != target.name):
                    raise ValueError(f"Unsafe session member: {info.filename}")
                destination = (staging / Path(*member.parts)).resolve()
                if not destination.is_relative_to(staged_target.resolve()):
                    raise ValueError(f"Session member escapes target: {info.filename}")
            archive.extractall(staging)
        validate_processed_input(staged_target, config=config)
        if target.exists():
            raise FileExistsError(f"Session appeared during download; existing files preserved: {target}")
        staged_target.rename(target)
    if progress_callback:
        progress_callback(100, "Processed session ready for segmentation.")

def main() -> None:
    """List available sessions, or download the sessions named on the command line."""
    parser = argparse.ArgumentParser(description="Download processed WayFAST sessions.")
    parser.add_argument("datasets", nargs="*", help="Session ZIP filenames.")
    parser.add_argument("--config", help="Generation YAML overrides.")
    args = parser.parse_args()
    config = load_config(args.config)
    def cli_progress(percent, text) -> None:
        """
        Print download progress.

        :param percent: float - completion percentage
        :param text: str - status text
        """
        print(f"[{percent:3.0f}%] {text}")

    if args.datasets:
        for session in args.datasets:
            download_wayfast(session, cli_progress, config)
    else:
        for session, size in list_datasets(cli_progress, config).items():
            print(f"{session}\t{size} bytes")

if __name__ == "__main__":
    main()
