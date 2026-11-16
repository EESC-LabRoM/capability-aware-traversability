from omegaconf import DictConfig, OmegaConf

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

def repository_root() -> Path:
    """
    Locate the repository root.

    :return: Path - repository root
    """
    # <root>/cat/src/cat/training/paths.py
    return Path(__file__).resolve().parents[4]

def configuration_directory() -> Path:
    """
    Locate CAT's Hydra configuration directory.

    :return: Path - configuration directory
    """
    return repository_root() / "cat" / "configs"

def resolve_repo_path(path, config) -> Path:
    """
    Resolve a configured path against the repository root.

    :param path: str | Path - absolute or repository-relative path
    :param config: dict - settings containing paths.root_dir
    :return: Path - resolved absolute path
    """
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(config["paths"]["root_dir"])
    return (root / path).resolve()

def register_resolvers() -> None:
    """
    Register the ``cat_root`` OmegaConf resolver used by ``train.yaml``.
    """
    OmegaConf.register_new_resolver("cat_root", lambda: str(repository_root()), replace=True)

def compose_train_cfg(overrides=()) -> DictConfig:
    """
    Compose the training config.

    :param overrides: tuple | list[str] - Hydra overrides, e.g. ``["training.epochs=20"]``
    :return: DictConfig - unresolved config (interpolations resolve on access)
    """
    register_resolvers()
    if GlobalHydra.instance().is_initialized():
        return compose(config_name="train", overrides=list(overrides))
    with initialize_config_dir(config_dir=str(configuration_directory()), version_base="1.3"):
        return compose(config_name="train", overrides=list(overrides))
