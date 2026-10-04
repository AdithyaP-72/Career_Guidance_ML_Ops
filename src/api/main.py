"""FastAPI service. Run: uv run uvicorn src.api.main:app --port 8000"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.data.common import ROOT
from src.models.recommend import MAX_SKILLS, MIN_SKILLS, Recommender, serving_dir
from src.monitoring import drift

LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
EDUCATION = ["12th", "Diploma", "B.Tech", "B.Com", "BBA/MBA", "B.Sc", "BA", "CA/CS", "MBBS/Pharma/Nursing", "LLB", "B.Ed", "Other"]
MIN_LOGS_FOR_DRIFT = 50
MIN_FEEDBACK = 20
LOW_HELPFUL = 0.6


class RecommendRequest(BaseModel):
    skills: list[str] = Field(..., description=f"At least {MIN_SKILLS} skills from /skills")
    experience: float = Field(0, ge=0, le=50)
    education: str | None = None

    @field_validator("skills")
    @classmethod
    def _skills(cls, v):
        v = [s.strip() for s in v if s and s.strip()]
        if len(v) > MAX_SKILLS:
            raise ValueError(f"at most {MAX_SKILLS} skills")
        return v

    @field_validator("education")
    @classmethod
    def _edu(cls, v):
        if v is not None and v not in EDUCATION:
            raise ValueError(f"education must be one of {EDUCATION}")
        return v


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    role: str | None = None
    comment: str | None = Field(None, max_length=500)


def _append(name: str, row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _read(name: str) -> list[dict]:
    p = LOG_DIR / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rec = Recommender.load()
    vocab = app.state.rec.vocabulary
    app.state.vocab = vocab
    app.state.vocab_set = set(vocab)
    share = app.state.rec.reference_share or {}
    app.state.ref_freq = np.array([share.get(v, 0.0) for v in vocab])
    yield


app = FastAPI(title="Career Guidance API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "roles": len(app.state.rec.pipe.classes_), "vocabulary": len(app.state.vocab)}


@app.get("/meta")
def meta():
    return {"min_skills": MIN_SKILLS, "max_skills": MAX_SKILLS, "education": EDUCATION, "families": sorted({p["family"] for p in app.state.rec.profiles.values()})}


@app.get("/skills")
def skills(q: str = Query("", max_length=50), limit: int = Query(15, ge=1, le=50)):
    """Search-as-you-type over the fixed vocabulary the model was trained on."""
    q = q.strip().lower()
    vocab = app.state.vocab
    if not q:
        return {"skills": vocab[:limit]}
    starts = [s for s in vocab if s.startswith(q)]
    contains = [s for s in vocab if q in s and not s.startswith(q)]
    return {"skills": (starts + contains)[:limit]}


@app.get("/roles/{role:path}")
def role_detail(role: str):
    if role not in app.state.rec.profiles:
        raise HTTPException(404, "Unknown role")
    return app.state.rec.role_profile(role)


@app.get("/roles")
def roles():
    return {"roles": [{"role": r, "family": p["family"], "n_postings": p["n_postings"]} for r, p in sorted(app.state.rec.profiles.items())]}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    rec: Recommender = app.state.rec
    try:
        out = rec.recommend(req.skills, req.experience, req.education)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if len(set(out["unknown_skills"])) == len(set(req.skills)):
        raise HTTPException(422, "None of these skills are in the model vocabulary; pick skills from /skills.")
    rid = uuid.uuid4().hex[:12]
    top = out["top_roles"][0]
    _append("predictions.jsonl", {"id": rid, "ts": time.time(), "skills": [s for s in req.skills if s.lower() in app.state.vocab_set],
                                  "experience": req.experience, "education": req.education, "top": top["role"], "confidence": top["confidence"]})
    return {"request_id": rid, **out}


@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    _append("feedback.jsonl", {**fb.model_dump(), "ts": time.time()})
    return {"ok": True}


@app.get("/drift")
def drift_status():
    """Compare skills users enter now against the training distribution."""
    logs = _read("predictions.jsonl")
    n = len(logs)
    res = {"n_live_requests": n, "min_required": MIN_LOGS_FOR_DRIFT}
    if n < MIN_LOGS_FOR_DRIFT:
        res["status"] = "insufficient_data"
        return res
    vocab = app.state.vocab
    live = drift.skill_frequencies([r["skills"] for r in logs], vocab)
    ref = app.state.ref_freq
    # users pick ~5 skills, postings carry ~8, so compare normalised mixes, not raw shares
    rb, lb = drift.binned(ref, live)
    js = drift.js_distance(rb, lb)
    conf = np.array([r["confidence"] for r in logs])
    res.update({"js_distance": round(js, 4), "psi": round(drift.psi(rb, lb), 4), "status": drift.status(js),
                "mean_confidence": round(float(conf.mean()), 4), "low_confidence_rate": round(float((conf < 0.25).mean()), 4),
                "top_movers": drift.top_movers(ref / max(ref.sum(), 1e-9), live / max(live.sum(), 1e-9), vocab, 8)})
    return res




@app.get("/stats")
def stats():
    """Usage + feedback summary from the local logs (shown on the Insights tab).

    Feedback is joined to the prediction it rated, so we can see which roles users find unhelpful and use
    that, together with drift, as a retraining signal.
    """
    preds, fbs = _read("predictions.jsonl"), _read("feedback.jsonl")
    by_id = {p["id"]: p for p in preds}
    helpful = [f["helpful"] for f in fbs]
    per_role: dict[str, list[bool]] = {}
    for f in fbs:
        p = by_id.get(f["request_id"])
        if p:
            per_role.setdefault(p["top"], []).append(bool(f["helpful"]))
    rated = sorted(({"role": r, "rated": len(v), "helpful_rate": round(sum(v) / len(v), 3)} for r, v in per_role.items()),
                   key=lambda x: (x["helpful_rate"], -x["rated"]))
    rate = round(sum(helpful) / len(helpful), 3) if helpful else None
    d = drift_status()
    reasons = []
    if rate is not None and len(helpful) >= MIN_FEEDBACK and rate < LOW_HELPFUL:
        reasons.append(f"only {rate:.0%} of {len(helpful)} ratings were helpful")
    if d.get("status") == "alert":
        reasons.append("live skill drift is in alert")
    return {"requests": len(preds), "feedback": len(helpful), "helpful_rate": rate,
            "least_helpful_roles": [r for r in rated if r["rated"] >= 3][:5],
            "retrain_recommended": bool(reasons), "retrain_reasons": reasons,
            "top_roles": pd_counts([p["top"] for p in preds])[:8],
            "recent": [{"top": p["top"], "confidence": p["confidence"], "n_skills": len(p["skills"])} for p in preds[-8:]][::-1]}


def pd_counts(items: list[str]) -> list[dict]:
    from collections import Counter

    return [{"role": k, "count": v} for k, v in Counter(items).most_common()]


def _report(name: str):
    for base in (ROOT / "reports", serving_dir() / "reports"):
        f = base / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


@app.get("/model-info")
def model_info():
    log_f = ROOT / "models/promotion_log.json"
    return {"comparison": _report("model_comparison.json") or [],
            "promotions": json.loads(log_f.read_text(encoding="utf-8"))[-5:] if log_f.exists() else [],
            "metrics": _report("metrics.json") or {}, "label_quality": _report("label_quality.json") or {},
            "drift_2019_2025": _report("drift_2019_2025.json") or {}}


STATIC = ROOT / "app" / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
