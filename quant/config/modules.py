"""
Python module version of modules.yaml for easier importing
"""

from pathlib import Path

import yaml


def _load_modules_from_yaml() -> dict:
    path = Path("quant/config/modules.yaml")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        parsed = yaml.safe_load(f) or {}
    return parsed.get("modules", {})


# Kept for backwards compatibility with imports expecting `quant.config.modules.modules`.
modules = _load_modules_from_yaml()
