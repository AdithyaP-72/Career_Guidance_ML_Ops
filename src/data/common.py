"""Shared helpers: skill normalisation and config loading."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(rel: str):
    with open(ROOT / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_params() -> dict:
    return load_yaml("params.yaml")


def load_aliases() -> dict[str, str]:
    return {str(k).lower(): str(v).lower() for k, v in load_yaml("configs/skill_aliases.yaml").items()}


_WS = re.compile(r"\s+")


def normalise_skills(raw: str | None, sep: str, aliases: dict[str, str]) -> list[str]:
    """Split a raw skill string, lower-case, alias and de-duplicate (order preserved)."""
    if not isinstance(raw, str):
        return []
    out, seen = [], set()
    for tok in raw.split(sep):
        s = _WS.sub(" ", tok.strip().lower())
        if not s or len(s) > 60:
            continue
        s = aliases.get(s, s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
