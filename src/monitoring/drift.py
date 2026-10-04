"""Skill-distribution drift: Jensen-Shannon distance and PSI between a reference and a live window."""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon


def skill_frequencies(postings: list[list[str]], vocab: list[str]) -> np.ndarray:
    """Share of postings mentioning each vocabulary skill."""
    idx = {s: i for i, s in enumerate(vocab)}
    c = Counter(s for sk in postings for s in set(sk) if s in idx)
    n = max(len(postings), 1)
    out = np.zeros(len(vocab))
    for s, v in c.items():
        out[idx[s]] = v / n
    return out


def js_distance(ref: np.ndarray, live: np.ndarray) -> float:
    if ref.sum() == 0 or live.sum() == 0:
        return 0.0
    return float(jensenshannon(ref / ref.sum(), live / live.sum(), base=2))


def psi(ref: np.ndarray, live: np.ndarray, eps: float = 1e-4) -> float:
    r = ref / max(ref.sum(), eps) + eps
    l = live / max(live.sum(), eps) + eps  # noqa: E741
    return float(np.sum((l - r) * np.log(l / r)))


def top_movers(ref: np.ndarray, live: np.ndarray, vocab: list[str], k: int = 10) -> list[dict]:
    d = live - ref
    order = np.argsort(-np.abs(d))[:k]
    return [{"skill": vocab[i], "reference": round(float(ref[i]), 4), "live": round(float(live[i]), 4)} for i in order]


def status(js: float) -> str:
    return "alert" if js >= 0.35 else "warning" if js >= 0.2 else "ok"


def binned(ref: np.ndarray, live: np.ndarray, k: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Collapse to the k most common reference skills + one 'other' bucket.

    Sparse 1,500-way comparisons are dominated by sampling noise on small live windows; binning keeps the
    test stable while still catching a real shift in which skills people bring.
    """
    top = np.argsort(-ref)[:k]
    mask = np.zeros(len(ref), bool)
    mask[top] = True
    r, l = ref / max(ref.sum(), 1e-9), live / max(live.sum(), 1e-9)  # noqa: E741
    return (np.append(r[top], r[~mask].sum()), np.append(l[top], l[~mask].sum()))
