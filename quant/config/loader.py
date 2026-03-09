"""
Configuration loading utilities for the trading system.
"""

import yaml
from pathlib import Path
from typing import Dict


def _load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _validate_configuration(config: Dict) -> None:
    required_top = ["data", "signals", "universe"]
    missing = [key for key in required_top if key not in config]
    if missing:
        raise ValueError(f"Configuration missing required sections: {missing}")

    if "tickers" not in config["universe"] or not config["universe"]["tickers"]:
        raise ValueError("Configuration requires at least one ticker in universe.tickers")

    cache_dir = config.get("data", {}).get("cache_dir")
    if not cache_dir:
        raise ValueError("Configuration requires data.cache_dir")


def load_configuration() -> Dict:
    """Load and validate merged system configuration from YAML files."""
    base_dir = Path("quant/config")
    config_path = base_dir / "settings.yaml"
    universe_path = base_dir / "universe.yaml"
    modules_path = base_dir / "modules.yaml"

    config = _load_yaml(config_path)
    universe = _load_yaml(universe_path)
    modules_cfg = _load_yaml(modules_path).get("modules", {})

    config['universe'] = universe
    config['modules'] = modules_cfg

    _validate_configuration(config)
    return config


def get_module_configuration(config: Dict) -> Dict:
    """Return module configuration from the unified config object."""
    return config.get("modules", {})
