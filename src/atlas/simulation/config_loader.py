"""YAML configuration loader with stdlib fallback when PyYAML is absent."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[+-]?\d+", val):
        return int(val)
    if re.fullmatch(r"[+-]?\d+\.\d+", val):
        return float(val)
    return val


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse nested key/value YAML used by simulation config files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        # Strip inline comment outside quotes
        if "#" in stripped and not stripped.startswith(('"', "'")):
            parts = stripped.split("#", 1)
            if parts[0].count('"') % 2 == 0 and parts[0].count("'") % 2 == 0:
                stripped = parts[0].strip()
        if ":" not in stripped:
            continue
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _simple_yaml_load(text)
