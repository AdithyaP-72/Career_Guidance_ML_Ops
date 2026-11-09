"""API + recommender tests, against the toy bundle (tests/toy_bundle.py). No DVC or MLflow server needed."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.serving.app import app
from src.serving.bundle import write_local
from src.serving.recommender import Recommender, _growth
from tests.toy_bundle import make_toy_bundle

ACCOUNTANT = ["Accounting", "TALLY", "gst", "tds", "Excel"]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_toy_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture(scope="module")
def client(bundle):
    mp = pytest.MonkeyPatch()
    mp.setenv("BUNDLE_DIR", str(bundle))
    with TestClient(app) as c:  # `with` runs the startup (lifespan) that loads the bundle
        yield c
    mp.undo()


# ── API ──────────────────────────────────────────────────────────────────────


def test_health_reports_the_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == "toy" and body["algorithm"] == "logreg"


def test_predict_recommends_the_obvious_role_with_gap_and_path(client):
    r = client.post("/predict", json={"skills": ACCOUNTANT, "experience_years": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills_used"] == ["accounting", "tally", "gst", "tds", "excel"]  # normalised like training data
    recs = body["recommendations"]
    assert [x["rank"] for x in recs] == [1, 2, 3]
    assert recs[0]["role"] == "Accountant" and recs[0]["family"] == "Accounting & Finance"
    conf = [x["confidence"] for x in recs]
    assert conf == sorted(conf, reverse=True) and sum(conf) <= 1.0 + 1e-3  # rounded to 4 dp
    top = recs[0]
    assert 0 <= top["readiness"] <= 1
    assert set(top["have"]).isdisjoint(top["missing"])
    assert {s["skill"] for s in top["learning_path"]} == set(top["missing"])
    prio = [s["priority"] for s in top["learning_path"]]
    assert prio == sorted(prio, reverse=True)


def test_unknown_skills_are_reported_not_fatal(client):
    r = client.post("/predict", json={"skills": [*ACCOUNTANT, "underwater basket weaving"]})
    assert r.status_code == 200
    assert r.json()["unknown_skills"] == ["underwater basket weaving"]


def test_too_few_known_skills_is_rejected_with_suggestions(client):
    r = client.post("/predict", json={"skills": ["accounting", "tally", "tallly", "zzz"]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["skills_used"] == ["accounting", "tally"]
    assert set(detail["unknown_skills"]) == {"tallly", "zzz"}
    assert "suggestions" in detail


@pytest.mark.parametrize(
    "payload",
    [
        {"skills": []},
        {"skills": ACCOUNTANT, "experience_years": -1},
        {"skills": ACCOUNTANT, "top_k": 11},
        {"skills": "accounting, tally"},
        {"experience_years": 2},
    ],
)
def test_invalid_requests_get_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_experience_is_optional_and_top_k_is_respected(client):
    r = client.post("/predict", json={"skills": ACCOUNTANT, "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 2


def test_skill_search_puts_prefix_matches_first(client):
    r = client.get("/skills", params={"q": "ta"})
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert matches[0] == "tally"
    assert all("ta" in m for m in matches)


def test_roles_lists_every_class(client):
    assert {r["role"] for r in client.get("/roles").json()} == {"Accountant", "Frontend Developer", "Staff Nurse"}


# ── recommender + bundle ─────────────────────────────────────────────────────


def test_readiness_is_weighted_share_of_core_skills(bundle):
    rec = Recommender(bundle)
    core = rec.roles["Accountant"]["core_skills"]
    score, have, missing = rec.readiness("Accountant", ["tally", "gst"])
    assert have == [s for s in core if s in ("tally", "gst")]
    assert score == pytest.approx((core["tally"] + core["gst"]) / sum(core.values()), abs=1e-4)
    assert set(have) | set(missing) == set(core)


def test_growth_is_a_clipped_ratio():
    assert _growth({"share_2019": 0.01, "share_2025": 0.02}) == pytest.approx(2.0)
    assert _growth({"share_2019": 0.01, "share_2025": 0.10}) == 2.0  # clipped
    assert _growth({"share_2019": 0.10, "share_2025": 0.01}) == 0.5  # clipped
    assert _growth({"share_2019": 0.0, "share_2025": 0.0}) == 1.0


class NotTrusted(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)


def test_bundle_refuses_a_model_with_untrusted_types(tmp_path, bundle):
    pipe = Pipeline([("step", NotTrusted())]).fit([[1]])
    with pytest.raises(SystemExit, match="not in TRUSTED_TYPES"):
        write_local(tmp_path / "b", pipe, bundle, {"model_name": "x", "version": "x", "algorithm": "x"})


# ── resume upload (the extractor itself is a teammate's module; here we test the endpoint around it) ──


def test_resume_upload_is_off_until_the_extractor_exists(client):
    assert client.get("/health").json()["features"] == {"resume_upload": False}
    r = client.post("/extract-skills", files={"file": ("cv.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert r.status_code == 501


def test_resume_upload_rejects_wrong_type_and_big_files(client):
    assert (
        client.post("/extract-skills", files={"file": ("cv.exe", b"MZ", "application/octet-stream")}).status_code == 415
    )
    big = b"x" * (2 * 1024 * 1024 + 1)
    assert client.post("/extract-skills", files={"file": ("cv.txt", big, "text/plain")}).status_code == 413


def test_resume_skills_are_filtered_to_the_model_vocabulary(client, monkeypatch):
    from src.serving import resume

    seen = {}

    def fake(data, filename, vocab, aliases):
        seen.update(data=data, filename=filename, vocab_size=len(vocab))
        return ["tally", "gst", "underwater basket weaving", "tally"]

    monkeypatch.setattr(resume, "resume_to_skills", fake)
    r = client.post("/extract-skills", files={"file": ("cv.txt", b"I know Tally and GST", "text/plain")})
    assert r.status_code == 200
    assert r.json() == {"skills": ["tally", "gst"], "ignored": 2}
    assert seen["data"] == b"I know Tally and GST" and seen["filename"] == "cv.txt" and seen["vocab_size"] > 0


def test_unreadable_resume_gives_the_extractor_message(client, monkeypatch):
    from src.serving import resume

    def scanned(*_):
        raise resume.ResumeError("this PDF is a scanned image; no text to read")

    monkeypatch.setattr(resume, "resume_to_skills", scanned)
    r = client.post("/extract-skills", files={"file": ("cv.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 422 and "scanned" in r.json()["detail"]
