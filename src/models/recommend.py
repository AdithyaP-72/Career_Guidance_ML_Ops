"""Inference: skills + experience (+ education) -> top-3 roles, why, readiness, have/missing core skills, learning path.

The API wraps this. Everything is loaded from local files; no network access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data.common import ROOT, load_aliases, load_params

MIN_SKILLS = 5
MAX_SKILLS = 10  # README: at least 5, up to ~10
LOW_CONFIDENCE = 0.25
EMERGING = 0.004  # skill's share of postings grew by >=0.4 points 2019 -> 2025
FADING = -0.004

MODELS = ROOT / "models"
CHAMPION = MODELS / "champion"


def serving_dir():
    """Promoted champion if there is one, else the latest pipeline output."""
    return CHAMPION if (CHAMPION / "model.pkl").exists() else MODELS


@dataclass
class Recommender:
    pipe: object
    profiles: dict
    growth: dict
    edu_lift: dict
    courses: dict
    edu_alpha: float = 0.35
    reference_share: dict | None = None

    @classmethod
    def load(cls) -> Recommender:
        d = serving_dir()
        pipe = joblib.load(d / "model.pkl")
        sp = json.loads((d / "skill_profiles.json").read_text(encoding="utf-8"))
        edu = json.loads((d / "education_prior.json").read_text(encoding="utf-8"))["lift"]
        courses = json.loads((d / "course_links.json").read_text(encoding="utf-8"))
        return cls(pipe, sp["profiles"], sp["skill_growth_2019_2025"], edu, courses,
                   load_params()["lookups"]["education_alpha"], sp.get("reference_skill_share"))

    # ---- helpers
    @property
    def vocabulary(self) -> list[str]:
        v = self.pipe["features"].named_transformers_["skills"].vocabulary_
        return sorted(v, key=v.get)

    @property
    def _vocab_index(self) -> dict:
        return self.pipe["features"].named_transformers_["skills"].vocabulary_

    def _normalise(self, skills: list[str]) -> list[str]:
        al = load_aliases()
        out = []
        for s in skills:
            s = " ".join(s.lower().split())
            s = al.get(s, s)
            if s and s not in out:
                out.append(s)
        return out

    def _proba(self, skills: list[str], experience: float, education: str | None) -> np.ndarray:
        X = pd.DataFrame({"skills": [skills], "experience": [experience]})
        proba = self.pipe.predict_proba(X)[0]
        lift = self.edu_lift.get(education or "")
        if lift:  # gentle nudge from what employers ask this degree for; never overrides the skills
            w = np.array([lift.get(self.profiles[c]["family"], 1.0) ** self.edu_alpha for c in self.pipe.classes_])
            proba = proba * w
            proba = proba / proba.sum()
        return proba

    def explain(self, role: str, skills: list[str], k: int = 4) -> list[dict]:
        """Which of the user's skills pushed the model towards this role (linear model coefficients)."""
        clf = self.pipe["clf"]
        if not hasattr(clf, "coef_"):
            return []
        ci = int(np.where(self.pipe.classes_ == role)[0][0])
        idx = self._vocab_index
        contrib = [(s, float(clf.coef_[ci][idx[s]])) for s in skills if s in idx]
        contrib.sort(key=lambda t: -t[1])
        top = [c for c in contrib if c[1] > 0][:k]
        total = sum(v for _, v in top) or 1.0
        return [{"skill": s, "weight": round(v / total, 3)} for s, v in top]

    # ---- main entry points
    def recommend(self, skills: list[str], experience: float = 0.0, education: str | None = None,
                  top_n: int = 3, path_len: int = 4) -> dict:
        skills = self._normalise(skills)
        if len(skills) < MIN_SKILLS:
            raise ValueError(f"Please provide at least {MIN_SKILLS} skills (got {len(skills)}).")
        if len(skills) > MAX_SKILLS:
            raise ValueError(f"Please provide at most {MAX_SKILLS} skills (got {len(skills)}).")
        known = set(self._vocab_index)
        unknown = [s for s in skills if s not in known]
        used = [s for s in skills if s in known]
        proba = self._proba(used, experience, education)
        order = proba.argsort()[::-1][:top_n]
        roles = [{"role": self.pipe.classes_[i], "family": self.profiles[self.pipe.classes_[i]]["family"],
                  "confidence": round(float(proba[i]), 4)} for i in order]
        best = roles[0]
        fam_p: dict[str, float] = {}
        for c, pr in zip(self.pipe.classes_, proba):
            f = self.profiles[c]["family"]
            fam_p[f] = fam_p.get(f, 0.0) + float(pr)
        families = [{"family": f, "confidence": round(v, 4)} for f, v in sorted(fam_p.items(), key=lambda kv: -kv[1])[:3]]
        out = {
            "top_families": families,
            "top_roles": roles,
            "low_confidence": best["confidence"] < LOW_CONFIDENCE,
            "unknown_skills": unknown,
            "education_note": f"Ranking nudged using what employers ask {education} graduates for." if self.edu_lift.get(education or "") else None,
            "details": {r["role"]: self.gap(r["role"], used, experience, education, path_len, base=r["confidence"]) for r in roles},
        }
        if out["low_confidence"]:
            out["note"] = (f"Low confidence: your skills fit several areas, so treat the roles below as a starting point. "
                           f"The closest field is {families[0]['family']} ({families[0]['confidence']:.0%}). "
                           "Add more specific skills for a sharper match.")
        return out

    def gap(self, role: str, skills: list[str], experience: float = 0.0, education: str | None = None,
            path_len: int = 4, base: float | None = None) -> dict:
        prof = self.profiles[role]
        have_set = set(skills)
        core = prof["core_skills"]
        total = sum(c["share"] for c in core) or 1.0
        have = [c["skill"] for c in core if c["skill"] in have_set]
        missing = [c for c in core if c["skill"] not in have_set]
        readiness = sum(c["share"] for c in core if c["skill"] in have_set) / total
        cooc = prof["cooccurrence"]
        ci = int(np.where(self.pipe.classes_ == role)[0][0])
        if base is None:
            base = float(self._proba(skills, experience, education)[ci])
        known = set(self._vocab_index)
        scored = []
        for m in missing:
            sk = m["skill"]
            link = [cooc.get(h, {}).get(sk, 0.0) for h in have if h in cooc] or [0.0]
            link_score = sum(link) / len(link)
            g = self.growth.get(sk, 0.0)
            boost = 1.0 + max(g, 0.0) * 10  # emerging skills first
            scored.append((m["share"] * (0.5 + link_score) * boost, sk, m["share"], g))
        scored.sort(reverse=True)
        path = []
        for _, sk, share, g in scored[:path_len]:
            gain = None
            if sk in known:  # what-if: how much would learning this skill lift the match?
                gain = round(float(self._proba(skills + [sk], experience, education)[ci]) - base, 4)
            path.append({"skill": sk, "asked_in": f"{share:.0%} of postings", "share": round(share, 4),
                         "match_gain": gain, "trend": "emerging" if g >= EMERGING else "fading" if g <= FADING else "steady",
                         "course": self.courses.get(sk)})
        return {
            "readiness": round(readiness, 3),
            "have": have,
            "missing": [m["skill"] for m in missing],
            "why": self.explain(role, skills),
            "learning_path": path,
            "typical_titles": prof["top_titles"],
            "experience_range": prof["experience"],
        }

    def role_profile(self, role: str) -> dict:
        p = self.profiles[role]
        return {
            "role": role, "family": p["family"], "n_postings": p["n_postings"], "top_titles": p["top_titles"],
            "experience": p["experience"],
            "core_skills": [{**c, "trend": "emerging" if p["growth"].get(c["skill"], 0) >= EMERGING else
                             "fading" if p["growth"].get(c["skill"], 0) <= FADING else "steady",
                             "course": self.courses.get(c["skill"])} for c in p["core_skills"]],
        }


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender.load()


if __name__ == "__main__":
    import sys

    skills = [s.strip() for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [
        "accounting", "tally", "excel", "microsoft office", "communication skills"]
    exp = float(sys.argv[2]) if len(sys.argv) > 2 else 1
    edu = sys.argv[3] if len(sys.argv) > 3 else None
    print(json.dumps(get_recommender().recommend(skills, exp, edu), indent=2))
