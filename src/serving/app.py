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
    POST /extract-skills   resume file (PDF/DOCX, ≤ 5 MB) → skills in the model's spelling (src/serving/resume.py)
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
from src.serving.schemas import ModelInfo, PredictRequest, PredictResponse, ResumeSkills, SkillSearch
from src.utils import ROOT, get_logger

log = get_logger("api")

MIN_SKILLS = int(os.environ.get("MIN_SKILLS", "5"))
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE", "0.2"))
MAX_RESUME_BYTES = resume.MAX_BYTES
RESUME_TYPES = {".pdf", ".docx"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = Path(os.environ.get("BUNDLE_DIR", ROOT / "bundle"))
    app.state.rec = Recommender(bundle)
    # The resume extractor's index is built once here, from the bundle's own vocabulary + keyness lookup,
    # so it always matches the model being served.
    app.state.extractor = resume.get_extractor(
        vocab_path=str(bundle / "skill_vocab.json"), keyness_path=str(bundle / "skill_keyness.json")
    )
    log.info("loaded %s", json.dumps(app.state.rec.meta))
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
        "revision": os.environ.get("GIT_SHA", "dev"),  # git commit the image was built from
        "features": {"resume_upload": True, "course_links": bool(rec(request).courses)},
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


@app.post("/extract-skills", response_model=ResumeSkills)
async def extract_skills(request: Request, file: Annotated[UploadFile, File()]):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in RESUME_TYPES:
        raise HTTPException(415, "Please upload a PDF or a .docx file.")
    data = await file.read(MAX_RESUME_BYTES + 1)  # never read more than the limit into memory
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, f"The file is too large (limit {MAX_RESUME_BYTES // 1_000_000} MB).")
    try:
        # Parsing is CPU work: run it in a worker thread so the server keeps answering other requests.
        result = await run_in_threadpool(resume.extract_skills, data, file.filename, 30, request.app.state.extractor)
    except resume.ResumeError as e:  # corrupt, encrypted, wrong content: the message is safe to show
        raise HTTPException(422, str(e)) from None
    vocab = rec(request).vocab
    skills = [s for s in result.names if s in vocab]
    message = result.message
    if not skills and not message:
        message = "We could not find skills from our list in this resume. You can add them manually."
    # Never log resume content (personal data): counts only.
    log.info("resume %s", json.dumps({"type": suffix, "bytes": len(data), "code": result.code, "skills": len(skills)}))
    return ResumeSkills(
        skills=skills, ignored=len(result.names) - len(skills), message=message, warnings=result.warnings
    )
