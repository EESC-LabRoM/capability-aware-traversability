import os
import urllib.request
from pathlib import Path

def download_models(progress_callback=None) -> None:
    """
    Download the SAM 2.1 checkpoint unless it already exists.

    Downloads to a `.part` file, replacing the checkpoint only when complete.

    :param progress_callback: Callable[[float, str], None] | None - receives percent and status text
    """
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_url = "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt"
    out_path = os.path.join(checkpoints_dir, "sam2.1_hiera_large.pt")
    partial_path = out_path + ".part"

    if os.path.exists(out_path):
        if progress_callback:
            progress_callback(100.0, "Model sam2.1_hiera_large.pt already exists. Skipping download.")
        return

    req = urllib.request.Request(model_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(partial_path, "wb") as out_file:
            total_length = int(response.info().get("Content-Length") or 0)
            downloaded = 0
            while data := response.read(5 * 1024 * 1024):
                out_file.write(data)
                downloaded += len(data)
                if progress_callback:
                    percent = downloaded / total_length * 100 if total_length else 0
                    mb_downloaded = downloaded / 1024 / 1024
                    text = f"SAM 2.1 Hiera Large: {mb_downloaded:.1f} MB"
                    if total_length:
                        text += f" / {total_length / 1048576:.1f} MB"
                    progress_callback(percent, text)
        os.replace(partial_path, out_path)
    finally:
        Path(partial_path).unlink(missing_ok=True)

if __name__ == "__main__":
    def cli_progress(percent, text) -> None:
        """
        Print progress on the current line.

        :param percent: float - completion percentage
        :param text: str - status text
        """
        print(f"[{percent:3.0f}%] {text}", end="\r")

    download_models(cli_progress)
    print()
