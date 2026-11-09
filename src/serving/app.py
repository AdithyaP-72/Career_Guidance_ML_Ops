"""The inference API.   uv run uvicorn src.serving.app:app --reload     (then open http://127.0.0.1:8000/docs)

Reads one model bundle (src/serving/bundle.py) at startup. Settings come from environment variables:

    BUNDLE_DIR       where the bundle is                      (default: <repo>/bundle)
    MIN_SKILLS       known skills required per request        (default: 5, as in the product spec)
    LOW_CONFIDENCE   top-1 confidence below this → low_confidence = true   (default: 0.2)
    CORS_ORIGINS     comma-separated origins allowed to call the API from a browser   (default: *)

Endpoints:
    GET  /health       liveness + which model is loaded + which features are on (Docker HEALTHCHECK, Render, CD)
    GET  /model        the bundle's metadata (registry version, MLflow run, ...)
    GET  /skills?q=    autocomplete for the skill search box
    GET  /roles        every role the model can recommend, with its family
    POST /predict      skills + experience → top-k roles, readiness, skill gap, learning path
    POST /extract-skills   resume file (PDF/DOCX/TXT, ≤ 2 MB) → skills in the model's spelling
                           (501 until src/serving/resume.py is implemented)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from src.serving import resume
from src.serving.recommender import Recommender
from src.serving.schemas import ModelInfo, PredictRequest, PredictResponse, SkillSearch
from src.utils import ROOT, get_logger

log = get_logger("api")

MIN_SKILLS = int(os.environ.get("MIN_SKILLS", "5"))
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE", "0.2"))
MAX_RESUME_BYTES = 2 * 1024 * 1024
RESUME_TYPES = {".pdf", ".docx", ".txt"}


def resume_upload_ready() -> bool:
    """False while src/serving/resume.py is still the NotImplementedError placeholder."""
    try:
        resume.resume_to_skills(b"", "probe.txt", set(), {})
    except NotImplementedError:
        return False
    except Exception:  # an implemented extractor may reject an empty file; that still means "ready"
        return True
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rec = Recommender(Path(os.environ.get("BUNDLE_DIR", ROOT / "bundle")))
    app.state.resume_ready = resume_upload_ready()
    log.info("loaded %s (resume upload: %s)", json.dumps(app.state.rec.meta), app.state.resume_ready)
    yield


app = FastAPI(
    title="Career Guidance API",
    description="Skills + experience → top job roles in India, with skill gap and learning path.",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],  # the browser frontend lives on another domain (Render static site)
    allow_headers=["*"],
)


def rec(request: Request) -> Recommender:
    return request.app.state.rec


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    m = rec(request).meta
    return {
        "status": "ok",
        "model": m["model_name"],
        "version": str(m["version"]),
        "algorithm": m.get("algorithm"),
        "features": {"resume_upload": request.app.state.resume_ready},
    }


@app.get("/model")
def model(request: Request):
    return rec(request).meta


@app.get("/skills", response_model=SkillSearch)
def skills(request: Request, q: str = Query("", max_length=60), limit: int = Query(10, ge=1, le=50)):
    return SkillSearch(query=q, matches=rec(request).search(q, limit))


@app.get("/roles")
def roles(request: Request):
    r = rec(request)
    return [{"role": str(c), "family": r.roles.get(str(c), {}).get("family")} for c in r.model.classes_]


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    t = time.perf_counter()
    r = rec(request)
    cleaned = r.clean(body.skills)
    if len(cleaned.known) < MIN_SKILLS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"need at least {MIN_SKILLS} skills the model knows, got {len(cleaned.known)}",
                "skills_used": cleaned.known,
                "unknown_skills": cleaned.unknown,
                "suggestions": {u: r.search(u, 3) for u in cleaned.unknown},
            },
        )
    recs = r.recommend(cleaned.known, body.experience_years, body.top_k)
    m = r.meta
    response = PredictResponse(
        model=ModelInfo(model_name=m["model_name"], version=str(m["version"]), algorithm=m.get("algorithm")),
        skills_used=cleaned.known,
        unknown_skills=cleaned.unknown,
        low_confidence=recs[0]["confidence"] < LOW_CONFIDENCE,
        recommendations=recs,
    )
    # One JSON line per prediction: the raw material for Phase 3 monitoring.
    log.info(
        "prediction %s",
        json.dumps(
            {
                "model_version": m["version"],
                "n_skills": len(cleaned.known),
                "n_unknown": len(cleaned.unknown),
                "top_role": recs[0]["role"],
                "confidence": recs[0]["confidence"],
                "latency_ms": round((time.perf_counter() - t) * 1000, 2),
            }
        ),
    )
    return response


@app.post("/extract-skills")
async def extract_skills(request: Request, file: Annotated[UploadFile, File()]):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in RESUME_TYPES:
        raise HTTPException(415, f"upload a {', '.join(sorted(RESUME_TYPES))} file")
    data = await file.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "file is larger than 2 MB")
    r = rec(request)
    vocab = set(r.vocab)
    try:
        # Parsing is CPU work: run it in a worker thread so the server keeps answering other requests.
        found = await run_in_threadpool(resume.resume_to_skills, data, file.filename, vocab, r.aliases)
    except NotImplementedError:
        raise HTTPException(501, "resume upload is coming soon; add your skills by hand for now") from None
    except resume.ResumeError as e:
        raise HTTPException(422, str(e)) from None
    skills = list(dict.fromkeys(s for s in found if s in vocab))
    # Never log resume content (personal data): counts only.
    log.info("resume %s", json.dumps({"type": suffix, "bytes": len(data), "skills_found": len(skills)}))
    return {"skills": skills, "ignored": len(found) - len(skills)}  # unknown to the model, or repeated
