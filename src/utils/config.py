"""Configuration utilities for experiment management."""

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in place.

    For nested dicts, values are merged recursively.
    For all other types, override replaces base.

    Returns:
        The modified base dict.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_experiment_config(experiment_name: str) -> dict:
    """Load default config, then overlay experiment-specific config.

    Args:
        experiment_name: Name of the experiment. Must match a YAML file
            in configs/ (e.g., "prompt_ensemble" loads configs/prompt_ensemble.yaml).

    Returns:
        Merged configuration dict.
    """
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)

    exp_path = PROJECT_ROOT / "configs" / f"{experiment_name}.yaml"
    if exp_path.exists():
        with open(exp_path) as f:
            exp_config = yaml.safe_load(f)
        if exp_config:
            deep_merge(config, exp_config)

    return config
