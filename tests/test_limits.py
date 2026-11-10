"""Abuse protection (src/serving/limits.py) and how the API uses it."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.limits import RateLimiter
from tests.toy_bundle import make_toy_bundle

ACCOUNTANT = ["accounting", "tally", "gst", "tds", "excel"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_rate_limiter_per_client_window():
    clock = Clock()
    rl = RateLimiter(per_client=2, total=100, window=60, clock=clock)
    assert rl.check("a") == 0 and rl.check("a") == 0
    assert rl.check("a") > 0  # third in the same minute: wait
    assert rl.check("b") == 0  # someone else is not affected
    clock.t += 61
    assert rl.check("a") == 0  # the window moved on


def test_rate_limiter_total_holds_even_with_a_new_address_every_time():
    rl = RateLimiter(per_client=100, total=3, window=60, clock=Clock())
    assert [rl.check(f"fake-{i}") for i in range(4)][-1] > 0


def test_rate_limiter_forgets_idle_visitors():
    clock = Clock()
    rl = RateLimiter(per_client=5, total=10**9, window=60, clock=clock)
    rl.MAX_TRACKED = 3
    for i in range(3):
        rl.check(f"old-{i}")
    clock.t += 120
    rl.check("new-1")
    rl.check("new-2")
    assert set(rl.hits) <= {"new-1", "new-2"}


@pytest.fixture
def strict_client(tmp_path, monkeypatch):
    monkeypatch.setenv("BUNDLE_DIR", str(make_toy_bundle(tmp_path / "bundle")))
    monkeypatch.setenv("RATE_PREDICT_PER_MIN", "2")
    monkeypatch.setenv("RATE_PREDICT_TOTAL_PER_MIN", "3")
    with TestClient(app) as c:
        yield c


def test_api_answers_429_with_retry_after(strict_client):
    def post(ip):
        return strict_client.post("/predict", json={"skills": ACCOUNTANT}, headers={"X-Forwarded-For": ip})

    assert post("1.1.1.1").status_code == 200 and post("1.1.1.1").status_code == 200
    r = post("1.1.1.1")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1 and "Too many" in r.json()["detail"]
    assert post("2.2.2.2").status_code == 200  # another visitor still gets through...
    assert post("3.3.3.3").status_code == 429  # ...until the total for everyone is used up
    assert strict_client.get("/health").status_code == 200  # never limited: CD and keep-warm rely on it


def test_oversized_requests_are_refused_before_being_read(strict_client):
    big_json = {"skills": ["x" * 70_000]}
    assert strict_client.post("/predict", json=big_json).status_code == 413
    huge = b"%PDF-" + b"x" * 6_500_000
    r = strict_client.post("/extract-skills", files={"file": ("cv.pdf", huge, "application/pdf")})
    assert r.status_code == 413 and "too large" in r.json()["detail"].lower()


def test_oversized_body_without_content_length_is_cut_off(strict_client):
    def chunks():  # a chunked upload declares no size up front
        for _ in range(20):
            yield b"x" * 10_000

    r = strict_client.post("/predict", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_skill_text_is_capped(strict_client):
    r = strict_client.post("/predict", json={"skills": [*ACCOUNTANT[:4], "y" * 101]})
    assert r.status_code == 422
