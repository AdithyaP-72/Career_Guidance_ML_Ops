"""Precision / recall of the resume -> skills extractor on hand-labelled public resumes.

    uv run python -m scripts.eval_resume_extractor --data /path/to/resume-dataset

`--data` is the unzipped Kaggle "Resume Dataset" (snehaanbhawal/resume-dataset); the PDFs are found by ID.
Labels live in tests/data/resume_eval_gold.json (IDs + skills only, no resume text).
The extractor is run on the real PDFs, exactly as the upload endpoint would.

Reported per variant: micro and macro precision / recall / F1, plus precision@10 and recall@10 (the app lets the
user keep 5-10 skills). Variants switch single design choices off, to show what each one buys.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from src.serving.resume import SkillExtractor, extract_skills
from src.utils import CONFIGS, MODELS, REPORTS, load_aliases

GOLD = Path(__file__).resolve().parents[1] / "tests" / "data" / "resume_eval_gold.json"


class NoGating(SkillExtractor):
    """Ablation: trust every vocabulary match (no short-skill / generic-word / strict-word rules)."""

    def _grade(self, canon, key, surface, in_skills):  # noqa: ARG002
        return "solid"


class NoSections(SkillExtractor):
    """Ablation: ignore 'Skills' sections (nothing counts as being inside one)."""

    def _scan(self, raw, norm, in_skills, found):
        super()._scan(raw, norm, False, found)


class Naive(NoGating, NoSections):
    """Ablation: plain longest-phrase vocabulary match, nothing else."""


def score(pred: list[str], gold: set[str], k: int | None = None) -> tuple[int, int, int]:
    p = pred[:k] if k else pred
    tp = len(set(p) & gold)
    return tp, len(p) - tp, len(gold) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(extractor: SkillExtractor, files: dict[str, bytes], gold: dict) -> dict:
    micro, micro10, per = [0, 0, 0], [0, 0, 0], []
    t0 = time.perf_counter()
    for rid, data in files.items():
        res = extract_skills(data, f"{rid}.pdf", top_n=100, extractor=extractor)
        g = set(gold[rid]["skills"])
        tp, fp, fn = score(res.names, g)
        tp10, fp10, fn10 = score(res.names, g, 10)
        micro = [a + b for a, b in zip(micro, (tp, fp, fn), strict=True)]
        micro10 = [a + b for a, b in zip(micro10, (tp10, fp10, fn10), strict=True)]
        per.append((rid, gold[rid]["category"], *prf(tp, fp, fn), len(res.names), len(g)))
    secs = (time.perf_counter() - t0) / len(files)
    mp, mr, mf = prf(*micro)
    p10, r10, f10 = prf(*micro10)
    n = len(per)
    return {
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro": {
            "precision": sum(x[2] for x in per) / n,
            "recall": sum(x[3] for x in per) / n,
            "f1": sum(x[4] for x in per) / n,
        },
        "at10": {"precision": p10, "recall": r10, "f1": f10},
        "avg_predicted": sum(x[5] for x in per) / n,
        "seconds_per_resume": secs,
        "per_resume": [
            {"id": a, "category": b, "precision": c, "recall": d, "f1": e, "predicted": f, "gold": g}
            for a, b, c, d, e, f, g in per
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="unzipped resume dataset (PDFs are found by ID)")
    ap.add_argument("--out", default=str(REPORTS / "resume_extractor_eval.json"))
    a = ap.parse_args()

    doc = json.loads(GOLD.read_text(encoding="utf-8"))
    pdfs = {p.stem: p for p in Path(a.data).rglob("*.pdf")}  # the zip nests them as data/data/<CATEGORY>/<ID>.pdf
    sets = {
        "dev (used to tune thresholds)": doc["resumes"],
        "HOLDOUT (labelled after freezing the design)": doc["holdout"],
    }
    for name, g in sets.items():
        missing = sorted(set(g) - set(pdfs))
        if missing:
            raise SystemExit(f"PDFs not found under {a.data} for {name}: {missing}")

    vocab = json.loads((MODELS / "skill_vocab.json").read_text(encoding="utf-8"))
    aliases = load_aliases(CONFIGS / "skill_aliases.yaml")
    kp = MODELS / "skill_keyness.json"
    keyness = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else None
    variants = {
        "full (default)": SkillExtractor(vocab, aliases, keyness=keyness),
        "no keyness (rules only)": SkillExtractor(vocab, aliases),
        "no fuzzy matching": SkillExtractor(vocab, aliases, fuzzy=False, keyness=keyness),
        "+ compound expansion": SkillExtractor(vocab, aliases, expand_compounds=True, keyness=keyness),
        "no 'Skills' section awareness": NoSections(vocab, aliases, keyness=keyness),
        "no false-positive gating": NoGating(vocab, aliases, keyness=keyness),
        "naive vocabulary match (no gating, no sections)": Naive(vocab, aliases),
    }
    out = {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "sets": {}}
    for set_name, gold in sets.items():
        files = {rid: pdfs[rid].read_bytes() for rid in gold}
        results = {name: evaluate(ex, files, gold) for name, ex in variants.items()}
        out["sets"][set_name] = {
            "n_resumes": len(files),
            "n_labels": sum(len(g["skills"]) for g in gold.values()),
            "variants": results,
        }
        print(f"\n=== {set_name}: {len(files)} resumes, {sum(len(g['skills']) for g in gold.values())} labelled skills")
        print(f"{'variant':50s} {'P':>6s} {'R':>6s} {'F1':>6s} | {'P@10':>6s} {'R@10':>6s} | {'macroF1':>7s}  s/resume")
        for name, r in results.items():
            m, t, k = r["micro"], r["macro"], r["at10"]
            print(
                f"{name:50s} {m['precision']:6.1%} {m['recall']:6.1%} {m['f1']:6.1%} | "
                f"{k['precision']:6.1%} {k['recall']:6.1%} | {t['f1']:7.1%}  {r['seconds_per_resume']:.3f}"
            )
        print("Weakest resumes (default variant, by F1):")
        for x in sorted(results["full (default)"]["per_resume"], key=lambda x: x["f1"])[:3]:
            print(
                f"  {x['id']} {x['category']:24s} P={x['precision']:.0%} R={x['recall']:.0%} "
                f"predicted={x['predicted']} gold={x['gold']}"
            )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
