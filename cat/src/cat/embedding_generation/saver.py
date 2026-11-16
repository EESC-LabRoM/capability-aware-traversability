import os
import json
from pathlib import Path

class Save:
    def __init__(self, path) -> None:
        """
        Create or reuse the cache directory.

        :param path: str | Path - save folder
        """
        path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        self.folder = path
        self.settings_file = path / Path('settings.json')
        self.fingerprint_file = path / Path('fingerprint.json')
        self.progress_file = path / Path('progress.json')

    def write_settings(self, settings) -> None:
        """
        Write settings.json.

        :param settings: dict - experiment settings
        """
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)

    def write_fingerprint(self, fingerprint) -> None:
        """
        Atomically write fingerprint.json.

        :param fingerprint: dict - fingerprint contents
        """
        # tmp + replace = atomic
        tmp = Path(str(self.fingerprint_file) + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(fingerprint, f, indent=4)
        os.replace(tmp, self.fingerprint_file)

    def write_progress(self, written_rows) -> None:
        """
        Atomically write progress.json.

        :param written_rows: int - rows written so far
        """
        # tmp + replace = atomic
        tmp = Path(str(self.progress_file) + '.tmp')
        with open(tmp, 'w') as f:
            json.dump({'written_rows': written_rows}, f, indent=4)
        os.replace(tmp, self.progress_file)

    def read_progress(self) -> int:
        """
        Read progress.json, or 0 if absent.

        :return: int - rows written so far
        """
        if not self.progress_file.exists():
            return 0
        with open(self.progress_file, 'r') as f:
            return json.load(f)['written_rows']
