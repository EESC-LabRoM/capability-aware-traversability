import json
from pathlib import Path

class Experiment:
    def __init__(self, path) -> None:
        """
        Root the experiment at ``path``; raise FileExistsError if ``settings.json`` exists.

        :param path: str | Path - experiment directory
        """
        path = Path(path)
        self.folder = path
        self.train_log_file = path / Path('train_log.txt')
        self.train_results_file = path / Path('train_results.json')
        self.settings_file = path / Path('settings.json')

        if self.settings_file.exists():
            raise FileExistsError("Experiment directory already contains a run (settings.json exists)")

        # hydra may have created the dir already
        path.mkdir(parents=True, exist_ok=True)

    def write_settings(self, settings) -> None:
        """
        Write settings to ``settings.json``.

        :param settings: dict - JSON-serializable settings
        """
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)