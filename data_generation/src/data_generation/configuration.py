from pathlib import Path

import yaml

def configuration_path() -> Path:
    """
    Return the default configuration file path.

    :return: Path - default config.yaml
    """
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

def _merge(base, overrides) -> dict:
    """
    Recursively merge `overrides` into `base` in place.

    :param base: dict - settings to update
    :param overrides: dict - settings that take precedence
    :return: dict - `base`
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path=None) -> dict:
    """
    Load generation settings, then apply an optional override file.

    :param config_path: str | Path | None - YAML override file, None = defaults only
    :return: dict - generation configuration
    """
    default_path = configuration_path()
    with default_path.open() as source:
        settings = yaml.safe_load(source)
    if config_path is not None and Path(config_path).resolve() != default_path.resolve():
        with Path(config_path).open() as source:
            _merge(settings, yaml.safe_load(source) or {})
    return settings

def resolve_config(config=None) -> dict:
    """
    Return `config`, or the default configuration if None.

    :param config: dict | None - configuration from the caller
    :return: dict - `config` or the loaded defaults
    """
    return load_config() if config is None else config
