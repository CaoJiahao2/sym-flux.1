from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _expand_env(value: Any) -> Any:
    """Recursively expand $VARS and ~ in string config values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a flat training config from JSON, YAML or YML.

    JSON is dependency-free and is the recommended format for this project.
    YAML is accepted only when PyYAML is installed in the environment.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")

    if suffix == ".json":
        cfg = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML config requires PyYAML. Use a .json config or install pyyaml."
            ) from exc
        cfg = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config suffix: {suffix}. Use .json, .yaml or .yml.")

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be an object/dict, got {type(cfg).__name__}")
    return _expand_env(cfg)


def validate_config_keys(config: dict[str, Any], valid_keys: set[str], config_path: str | None = None) -> None:
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        prefix = f" in {config_path}" if config_path else ""
        raise ValueError(
            f"Unknown config key(s){prefix}: {', '.join(unknown)}. "
            "Config keys must match the target script argument names."
        )
