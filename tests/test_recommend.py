import pytest

from src.data.common import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "models/model.pkl").exists(), reason="run `dvc repro` first")


def test_recommend_shape_and_min_skills():
    from src.models.recommend import get_recommender

    r = get_recommender()
    with pytest.raises(ValueError):
        r.recommend(["python", "sql"], 1)
    out = r.recommend(["accounting", "tally", "excel", "gst", "tds"], 2)
    assert len(out["top_roles"]) == 3
    top = out["top_roles"][0]["role"]
    assert 0 <= out["details"][top]["readiness"] <= 1
    assert out["top_roles"][0]["family"] == "Accounting & Finance"
