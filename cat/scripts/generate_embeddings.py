import argparse
import subprocess
import sys
from pathlib import Path

from cat.configuration import load_config

def parse_arguments(config, argv=None) -> tuple[argparse.Namespace, list[str]]:
    """
    Parse the wrapper options, defaulting paths from the configuration.

    :param config: dict - resolved CAT configuration
    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    :return: tuple[argparse.Namespace, list[str]] - parsed options and unrecognised arguments
    """
    data = config["data"]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML overriding the CAT configuration")
    parser.add_argument("--data_folder", type=Path, default=Path(data["data_dir"]))
    parser.add_argument("--data_csv", type=Path, default=None,
                        help=f"Label CSV (default: <data_folder>/{Path(data['data_csv']).name})")
    parser.add_argument("--save_folder", type=Path, default=Path(data["embeddings_path"]).parent)
    return parser.parse_known_args(argv)

def main(argv=None) -> int:
    """
    Run the embedding generator with the outdoor defaults.

    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    :return: int - generator exit code
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_path = config_parser.parse_known_args(argv)[0].config
    config = load_config(config_path)
    outdoor = config["embedding"]["outdoor"]
    args, extra = parse_arguments(config, argv)

    data_csv = args.data_csv if args.data_csv is not None else args.data_folder / Path(config["data"]["data_csv"]).name
    # config resolution is (h, w), generator takes res_x=w, res_y=h
    resolution = config["data"]["resolution"]
    command = [
        sys.executable, "-m", "cat.embedding_generation.embedding_generator",
        "--seed", str(config["seed"]),
        "--data_folder", str(args.data_folder),
        "--data_csv", str(data_csv),
        "--save_folder", str(args.save_folder),
        "--res_x", str(resolution[1]),
        "--res_y", str(resolution[0]),
        "--batch_size", str(outdoor["batch_size"]),
        "--num_workers", str(config["data"]["num_workers"]),
        "--model", outdoor["model"],
        *(["--use_depth"] if outdoor["use_depth"] else []),
        *(["--config", config_path] if config_path is not None else []),
        *extra,
    ]
    return subprocess.run(command).returncode

if __name__ == "__main__":
    sys.exit(main())