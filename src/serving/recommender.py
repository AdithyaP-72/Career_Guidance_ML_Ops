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
Each step lists up to 3 courses from skill_courses.json (free NPTEL / SWAYAM first, then Coursera), only those
whose title is about the skill or a synonym of it.

configs/skill_equivalents.yaml decides what counts as the same skill ("ci/cd" = "continuous integration",
"fastapi" = "fast api") and which recruiter tags are too vague to be a skill gap ("api", "development").
This only affects readiness / gap / learning path; the model's inputs are unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import skops.io
import yaml

from src.features.build import EXPERIENCE, SKILLS
from src.utils import CONFIGS, load_aliases, normalise_skill, read_json

GROWTH_CLIP = (0.5, 2.0)
COURSES_PER_SKILL = 3
EQUIVALENTS = CONFIGS / "skill_equivalents.yaml"
_WORD = re.compile(r"[a-z][a-z0-9+#]*")


def load_equivalents(path: Path = EQUIVALENTS) -> dict:
    """configs/skill_equivalents.yaml: same_skill groups, too_vague skills, skills with no_courses."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        "same_skill": [list(g) for g in cfg.get("same_skill", [])],
        "too_vague": set(cfg.get("too_vague", [])),
        "no_courses": set(cfg.get("no_courses", [])),
    }


def _stems(text: str) -> set[str]:
    """Word stems for loose title matching: 'Accounting' and 'accounts payable' share 'accou'."""
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) >= 3}


@dataclass
class Cleaned:
    known: list[str]
    unknown: list[str]


class Recommender:
    def __init__(
        self,
        bundle_dir: str | Path,
        aliases: dict[str, str] | None = None,
        equivalents: dict | None = None,
    ):
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
        eq = load_equivalents() if equivalents is None else equivalents
        self.vague, self.no_courses = set(eq.get("too_vague", ())), set(eq.get("no_courses", ()))
        self.canon = self._same_skill_map(eq.get("same_skill", []))
        self.members: dict[str, set[str]] = defaultdict(set)
        for name, c in self.canon.items():
            self.members[c].add(name)

    def _same_skill_map(self, groups: list[list[str]]) -> dict[str, str]:
        """skill name -> one canonical name, for names that mean the same skill (see skill_equivalents.yaml)."""
        names = set(self.co_index) | {s for r in self.roles.values() for s in r["core_skills"]}
        canon: dict[str, str] = {}
        for g in groups:
            for s in g:
                canon[s] = g[0]
        twins: dict[str, list[str]] = defaultdict(list)  # "fast api" / "fastapi"
        for s in sorted(names):
            twins[re.sub(r"[\s.\-/_]", "", s)].append(s)
        for key, group in twins.items():
            if len(group) > 1 and len(key) >= 4:  # not "b.a" (Bachelor of Arts) / "ba" (Business Analyst)
                head = next((canon[s] for s in group if s in canon), group[0])
                for s in group:
                    canon.setdefault(s, head)
        for s in names:  # "python development" means python
            m = re.fullmatch(r"(.+) (?:development|developer|programming)", s)
            if m and m.group(1) in names:
                canon.setdefault(s, canon.get(m.group(1), m.group(1)))
        return canon

    def same(self, skill: str) -> str:
        return self.canon.get(skill, skill)

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

    def core_skills(self, role: str) -> dict[str, float]:
        """The role's core skills without vague fragments, one entry per skill (synonyms merged, highest share)."""
        merged: dict[str, tuple[str, float]] = {}
        for s, w in self.roles[role]["core_skills"].items():
            if s in self.vague:
                continue
            c = self.same(s)
            if c not in merged or w > merged[c][1]:
                merged[c] = (merged[c][0] if c in merged else s, w)
        return dict(merged.values())

    def readiness(self, role: str, skills: list[str]) -> tuple[float, list[str], list[str]]:
        core = self.core_skills(role)
        mine = {self.same(s): s for s in skills}  # canonical -> the user's own spelling
        have = [mine[self.same(s)] for s in core if self.same(s) in mine]
        missing = [s for s in core if self.same(s) not in mine]
        total = sum(core.values())
        score = sum(w for s, w in core.items() if self.same(s) in mine) / total if total else 0.0
        return round(score, 4), have, missing

    def courses_for(self, skill: str) -> list[dict]:
        """Up to 3 courses for the skill or any of its synonyms, only those whose title is about it."""
        names = self.members.get(self.same(skill)) or {skill}
        if names & self.no_courses:
            return []
        stems = set().union(*(_stems(n) for n in names))
        out, seen = [], set()
        for name in [skill, *sorted(names - {skill})]:
            for c in self.courses.get(name, []):
                title = c["title"].lower()
                about = any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", title) for n in names)
                if c["url"] not in seen and (about or stems & _stems(title)):
                    seen.add(c["url"])
                    out.append(c)
        return out[:COURSES_PER_SKILL]

    def affinity(self, skill: str, skills: list[str]) -> float:
        """Average share of postings with `skill` that also list each of your skills (0..1)."""
        i = self.co_index.get(skill)
        own = [self.co_index[s] for s in skills if s in self.co_index]
        if i is None or not own or self.co[i, i] == 0:
            return 0.0
        return float(self.co[i, own].sum() / (self.co[i, i] * len(own)))

    def learning_path(self, role: str, skills: list[str], missing: list[str]) -> list[dict]:
        core = self.core_skills(role)
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
                    "courses": self.courses_for(s),
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
