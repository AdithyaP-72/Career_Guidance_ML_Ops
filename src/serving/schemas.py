"""Request / response shapes for the API. FastAPI validates requests against these and
publishes them in the OpenAPI docs at /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    skills: list[str] = Field(
        min_length=1,
        max_length=30,
        description="Your skills. Use /skills?q=... to find the spellings the model knows.",
        examples=[["accounting", "tally", "gst", "tds", "excel"]],
    )
    experience_years: float | None = Field(default=None, ge=0, le=60, examples=[1])
    top_k: int = Field(default=3, ge=1, le=10)


class LearningStep(BaseModel):
    skill: str
    share_of_role_postings: float
    affinity_with_your_skills: float
    market_growth: float
    priority: float


class Recommendation(BaseModel):
    rank: int
    role: str
    family: str | None
    confidence: float
    readiness: float
    have: list[str]
    missing: list[str]
    learning_path: list[LearningStep]
    typical_experience_years: dict[str, float] | None = None


class ModelInfo(BaseModel):
    model_name: str
    version: str
    algorithm: str | None = None


class PredictResponse(BaseModel):
    model: ModelInfo
    skills_used: list[str]
    unknown_skills: list[str]
    low_confidence: bool = Field(
        description="Top role's confidence is below the threshold: show the family and ask for more skills"
    )
    recommendations: list[Recommendation]


class SkillSearch(BaseModel):
    query: str
    matches: list[str]
