"""
Configuration loading utilities for the trading system.
"""

import yaml
from pathlib import Path
from typing import Dict


def load_configuration() -> Dict:
    """Load system configuration from YAML files."""
    config_path = Path("quant/config/settings.yaml")
    universe_path = Path("quant/config/universe.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)

    config['universe'] = universe
    return config