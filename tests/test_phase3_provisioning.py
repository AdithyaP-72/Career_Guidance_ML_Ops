import json

import yaml

from src.serving.phase3_evidence import evidence_metrics
from src.utils import ROOT, write_json


def yaml_file(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_compose_scraper_and_dashboard_use_the_same_provisioned_sources():
    monitoring = ROOT / "monitoring"
    services = yaml_file(ROOT / "compose.yaml")["services"]
    scrape = yaml_file(monitoring / "prometheus.yml")["scrape_configs"][0]
    assert scrape["metrics_path"] == "/metrics"
    assert scrape["static_configs"][0]["targets"] == ["api:8000"]
    assert scrape["authorization"]["credentials_file"] == services["api"]["environment"]["MONITORING_TOKEN_FILE"]
    provisioning = monitoring / "grafana" / "provisioning"
    datasource = yaml_file(provisioning / "datasources/prometheus.yml")["datasources"][0]
    assert datasource["url"] == "http://prometheus:9090"
    dashboard = json.loads((monitoring / "grafana/dashboards/drift_dashboard.json").read_text(encoding="utf-8"))
    assert len({p["id"] for p in dashboard["panels"]}) == len(dashboard["panels"])
    assert all(p["datasource"]["uid"] == datasource["uid"] for p in dashboard["panels"] if p.get("targets"))
    assert sum(p["type"] == "row" for p in dashboard["panels"]) == 7
    for service in services.values():
        assert all(binding.startswith("127.0.0.1:") for binding in service["ports"])


def test_evidence_exports_known_fields_without_leaking_other_report_content(tmp_path):
    assert evidence_metrics(tmp_path) == ""
    write_json(
        {
            "algorithm": "test",
            "run_id": "run-01",
            "token": "secret-do-not-export",
            "validation": {"test_check": {"passed": False}},
            "pipeline": ["validate"],
        },
        tmp_path / "pipeline.json",
    )
    write_json(
        {"audit": [{"at": "2026-11-27T12:00:00+00:00", "action": "rolled_back", "approved_by": "Demo reviewer"}]},
        tmp_path / "retraining" / "latest.json",
    )
    write_json(
        {"windows": {"synthetic_2026_demo": {"drift": {"overall_psi": 0.63}, "top3_accepted": 0.83}}},
        tmp_path / "demo" / "latest.json",
    )
    write_json(
        {"real_2019_reference": {"top3": 0.70, "max_psi": 0.26, "top_drifted": [{"feature": "html"}]}},
        tmp_path / "real_drift" / "latest.json",
    )
    metrics = evidence_metrics(tmp_path)
    assert 'career_phase3_real_drift_score{split="real_2019_reference",metric="max_psi"} 0.26' in metrics
    assert 'split="real_2025_test"' not in metrics
    assert 'career_phase3_quality_pass{check="test_check",source="committed_report"} 0.0' in metrics
    assert 'stage="validate"' in metrics
    assert 'action="rolled_back",approved_by="Demo reviewer"' in metrics
    assert 'career_phase3_replay_score{window="synthetic_2026_demo",metric="max_psi"} 0.63' in metrics
    assert "secret-do-not-export" not in metrics
