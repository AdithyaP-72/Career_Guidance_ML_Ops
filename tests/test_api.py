import pytest

from src.data.common import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "models/model.pkl").exists(), reason="run `dvc repro` first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    import src.api.main as m

    m.LOG_DIR = tmp_path_factory.mktemp("logs")
    with TestClient(m.app) as c:
        yield c


GOOD = {"skills": ["accounting", "tally", "gst", "tds", "excel"], "experience": 2, "education": "B.Com"}


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "B.Com" in client.get("/meta").json()["education"]


def test_skill_search(client):
    r = client.get("/skills", params={"q": "pyth"}).json()["skills"]
    assert r and all("pyth" in s for s in r)


def test_recommend_ok(client):
    r = client.post("/recommend", json=GOOD)
    assert r.status_code == 200
    b = r.json()
    assert len(b["top_roles"]) == 3 and b["request_id"]
    assert b["top_roles"][0]["family"] == "Accounting & Finance"


def test_recommend_validation(client):
    assert client.post("/recommend", json={"skills": ["python"], "experience": 1}).status_code == 422
    assert client.post("/recommend", json={**GOOD, "experience": -1}).status_code == 422
    assert client.post("/recommend", json={**GOOD, "education": "PhD in Cooking"}).status_code == 422
    assert client.post("/recommend", json={"skills": ["zzqq1", "zzqq2", "zzqq3", "zzqq4", "zzqq5"]}).status_code == 422


def test_feedback_metrics_drift(client):
    rid = client.post("/recommend", json=GOOD).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "helpful": True}).json()["ok"]
    assert client.get("/stats").json()["requests"] >= 1
    assert client.get("/drift").json()["status"] == "insufficient_data"


def test_role_detail_and_roles(client):
    roles = client.get("/roles").json()["roles"]
    assert len(roles) > 50
    r = client.get("/roles/Tax / GST Executive")
    assert r.status_code == 200 and r.json()["core_skills"]
    assert client.get("/roles/Nope").status_code == 404


def test_explainability_and_path(client):
    b = client.post("/recommend", json=GOOD).json()
    d = b["details"][b["top_roles"][0]["role"]]
    assert d["why"] and 0 <= d["readiness"] <= 1
    assert all("course" in p and "trend" in p for p in d["learning_path"])


def test_ui_and_info(client):
    assert "Career Guide" in client.get("/").text
    assert "metrics" in client.get("/model-info").json()
    assert "requests" in client.get("/stats").json()


def test_max_skills_and_family_fallback(client):
    many = [f"s{i}" for i in range(11)]
    assert client.post("/recommend", json={"skills": many}).status_code == 422
    b = client.post("/recommend", json=GOOD).json()
    assert len(b["top_families"]) == 3


def test_education_prior_available_for_every_ui_option(client):
    import json

    from src.models.recommend import serving_dir

    lift = json.loads((serving_dir() / "education_prior.json").read_text(encoding="utf-8"))["lift"]
    for edu in client.get("/meta").json()["education"]:
        if edu != "Other":
            assert edu in lift, edu


def test_stats_has_retrain_signal(client):
    assert "retrain_recommended" in client.get("/stats").json()
