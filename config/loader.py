"""
Configuration loader module.
Loads settings from config/config.yaml with environment variable overrides and sensible fallbacks.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_config_cache: Dict[str, Any] = {}


def load_config(config_file: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file with caching."""
    global _config_cache
    if _config_cache:
        return _config_cache

    path = Path(config_file)
    if not path.exists():
        # Fallback to defaults
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _config_cache = data
    return _config_cache


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Retrieve a specific nested config value with optional default."""
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)
