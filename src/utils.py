"""Shared helpers: project paths, config loading, skill and experience parsing."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_params() -> dict:
    return load_yaml(ROOT / "params.yaml")


def _json_default(obj):
    # numpy scalars/arrays → plain Python
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── skills ─────────────────────────────────────────────────────────────────────

_SPACE_AROUND_PUNCT = re.compile(r"\s*([/+\-#.])\s*")  # "c + +" → "c++", "ci / cd" → "ci/cd"
_EDGE_JUNK = " \t\"'`,;:*•·|"


def normalise_skill(raw: str, aliases: dict[str, str] | None = None) -> str:
    """Lower-case, tidy spacing/punctuation, then map known synonyms to one name."""
    s = re.sub(r"\s+", " ", str(raw).lower()).strip()
    s = _SPACE_AROUND_PUNCT.sub(r"\1", s)
    s = s.strip(_EDGE_JUNK).rstrip(".").strip(_EDGE_JUNK)
    if aliases:
        s = aliases.get(s, s)
    return s


def load_aliases(path: str | Path = CONFIGS / "skill_aliases.yaml") -> dict[str, str]:
    """skill_aliases.yaml is {canonical: [variants]}; return {variant: canonical}."""
    lookup: dict[str, str] = {}
    for canonical, variants in (load_yaml(path) or {}).items():
        canon = normalise_skill(canonical)
        for v in variants or []:
            lookup[normalise_skill(v)] = canon
    return lookup


def parse_skills(raw, sep: str, aliases: dict[str, str] | None = None) -> list[str]:
    """Split a raw skills string on the regex `sep`; normalised, de-duplicated, order kept."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    skills = (normalise_skill(part, aliases) for part in re.split(sep, str(raw)))
    return list(dict.fromkeys(s for s in skills if s))


def as_list(skills) -> list[str]:
    """Parquet gives list columns back as numpy arrays; turn one cell into a list."""
    return [] if skills is None else list(skills)


# ── experience ─────────────────────────────────────────────────────────────────


def parse_experience(raw) -> tuple[float, float]:
    """'2 - 5 yrs' / '0-3 Yrs' → (2.0, 5.0). Anything without numbers → (nan, nan)."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))] if pd.notna(raw) else []
    if not nums:
        return math.nan, math.nan
    return nums[0], nums[1] if len(nums) > 1 else nums[0]
