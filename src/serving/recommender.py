"""Everything the API computes, independent of FastAPI (so it is easy to test).

    Recommender(bundle_dir)
        .clean(raw_skills)            normalise + alias-map like the training data; split known / unknown
        .search(query)                autocomplete over the model's skill vocabulary
        .recommend(skills, exp, k)    top-k roles + readiness + skill gap + learning path for each

Readiness (O2): share of the role's core-skill weight you already have, where each core skill is
weighted by how many of the role's postings ask for it.

Learning path (O3): the role's missing core skills, ordered by
    priority = share of role postings asking for it
             × (1 + affinity: how often it appears together with the skills you already have)
             × growth: its 2025 share of postings vs 2019 (clipped to 0.5..2, unknown = 1)
i.e. learn first what the role needs most, that builds on what you know, and that is growing.
Each step lists up to 3 courses from skill_courses.json (free NPTEL / SWAYAM first, then Coursera).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import skops.io

from src.features.build import EXPERIENCE, SKILLS
from src.utils import load_aliases, normalise_skill, read_json

GROWTH_CLIP = (0.5, 2.0)
COURSES_PER_SKILL = 3


@dataclass
class Cleaned:
    known: list[str]
    unknown: list[str]


class Recommender:
    def __init__(self, bundle_dir: str | Path, aliases: dict[str, str] | None = None):
        d = Path(bundle_dir)
        self.meta = read_json(d / "meta.json")
        self.model = skops.io.load(d / "model.skops", trusted=read_json(d / "trusted_types.json"))
        profiles = read_json(d / "skill_profiles.json")
        self.roles: dict = profiles["roles"]
        self.growth = {s: _growth(v) for s, v in profiles.get("skill_growth", {}).items()}
        self.co = sp.load_npz(d / "skill_cooccurrence.npz").tocsr()
        self.co_index = {s: i for i, s in enumerate(read_json(d / "skill_vocab.json"))}
        self.courses: dict[str, list[dict]] = read_json(d / "skill_courses.json")  # skill → [{title, provider, url}]
        self.aliases = load_aliases() if aliases is None else aliases
        self.vocab = self.model.named_steps["features"].named_transformers_["skills"].vocabulary_
        freq = self.co.diagonal()
        # Search list: every skill the model knows, most common first.
        self.search_list = sorted(self.vocab, key=lambda s: -freq[self.co_index[s]] if s in self.co_index else 0)

    # ── input ─────────────────────────────────────────────────────────────────

    def clean(self, raw_skills: list[str]) -> Cleaned:
        seen = list(dict.fromkeys(s for s in (normalise_skill(r, self.aliases) for r in raw_skills) if s))
        return Cleaned([s for s in seen if s in self.vocab], [s for s in seen if s not in self.vocab])

    def search(self, query: str, limit: int = 10) -> list[str]:
        q = normalise_skill(query, None)
        if not q:
            return self.search_list[:limit]
        prefix = [s for s in self.search_list if s.startswith(q)]
        inside = [s for s in self.search_list if q in s and not s.startswith(q)]
        return (prefix + inside)[:limit]

    # ── output ────────────────────────────────────────────────────────────────

    def predict(self, skills: list[str], experience: float | None, k: int) -> list[tuple[str, float]]:
        X = pd.DataFrame({SKILLS: [skills], EXPERIENCE: [np.nan if experience is None else float(experience)]})
        proba = self.model.predict_proba(X)[0]
        top = np.argsort(-proba)[:k]
        return [(str(self.model.classes_[i]), float(proba[i])) for i in top]

    def readiness(self, role: str, skills: list[str]) -> tuple[float, list[str], list[str]]:
        core: dict[str, float] = self.roles[role]["core_skills"]
        have = [s for s in core if s in skills]
        total = sum(core.values())
        score = sum(core[s] for s in have) / total if total else 0.0
        return round(score, 4), have, [s for s in core if s not in skills]

    def affinity(self, skill: str, skills: list[str]) -> float:
        """Average share of postings with `skill` that also list each of your skills (0..1)."""
        i = self.co_index.get(skill)
        own = [self.co_index[s] for s in skills if s in self.co_index]
        if i is None or not own or self.co[i, i] == 0:
            return 0.0
        return float(self.co[i, own].sum() / (self.co[i, i] * len(own)))

    def learning_path(self, role: str, skills: list[str], missing: list[str]) -> list[dict]:
        core = self.roles[role]["core_skills"]
        steps = []
        for s in missing:
            aff, growth = self.affinity(s, skills), self.growth.get(s, 1.0)
            steps.append(
                {
                    "skill": s,
                    "share_of_role_postings": core[s],
                    "affinity_with_your_skills": round(aff, 4),
                    "market_growth": round(growth, 3),
                    "priority": round(core[s] * (1 + aff) * growth, 4),
                    "courses": self.courses.get(s, [])[:COURSES_PER_SKILL],
                }
            )
        return sorted(steps, key=lambda x: -x["priority"])

    def recommend(self, skills: list[str], experience: float | None, k: int = 3) -> list[dict]:
        out = []
        for rank, (role, conf) in enumerate(self.predict(skills, experience, k), start=1):
            info = self.roles.get(role, {})
            score, have, missing = self.readiness(role, skills) if info else (0.0, [], [])
            out.append(
                {
                    "rank": rank,
                    "role": role,
                    "family": info.get("family"),
                    "confidence": round(conf, 4),
                    "readiness": score,
                    "have": have,
                    "missing": missing,
                    "learning_path": self.learning_path(role, skills, missing) if info else [],
                    "typical_experience_years": info.get("experience_years"),
                }
            )
        return out


def _growth(shares: dict[str, float]) -> float:
    """{'share_2019': a, 'share_2025': b} → b / a, clipped. Keys sorted so the older year comes first."""
    old, new = (shares[k] for k in sorted(shares))
    if old <= 0:
        return GROWTH_CLIP[1] if new > 0 else 1.0
    return float(np.clip(new / old, *GROWTH_CLIP))
