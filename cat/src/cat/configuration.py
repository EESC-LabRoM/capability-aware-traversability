from omegaconf import OmegaConf

from cat.training.paths import compose_train_cfg, configuration_directory

def load_config(config_path=None) -> dict:
    """
    Compose train.yaml and embedding.yaml, then apply an optional override YAML.

    :param config_path: str | Path | None - additional YAML settings
    :return: dict - resolved CAT configuration
    """
    # defaults, `embedding` settings, then override file
    config = compose_train_cfg()
    OmegaConf.set_struct(config, False)
    config.embedding = OmegaConf.load(configuration_directory() / "embedding.yaml")
    # overrides may only set known keys
    OmegaConf.set_struct(config, True)
    if config_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    return OmegaConf.to_container(config, resolve=True)

def resolve_config(config=None) -> dict:
    """
    Reuse supplied settings or compose CAT defaults.

    :param config: dict | None - complete configuration from the caller
    :return: dict - resolved configuration
    """
    return load_config() if config is None else config
