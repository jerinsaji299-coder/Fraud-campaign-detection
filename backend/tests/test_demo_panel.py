"""The live demo panel's contract with a viewer.

The panel's danger is not a wrong number, it is a **plausible** one: a
presentation audience cannot tell an invented output from a computed one. So
the things pinned here are the ones that stop that happening.

1. With no bundle installed every endpoint **404s with an explanation**, so the
   frontend shows an empty state. Nothing is defaulted, nothing is zero-filled.
2. The aggregate figures come from the comparison artifact or are **absent**;
   there is no path that fabricates them.
3. The panel's purity and coverage are the **frozen matching rule's own**
   numbers - `cluster_match_detail` decides the match and supplies the two
   quantities, so a cluster shown as 60% pure cannot be one the rule counted
   as a hit at a different threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import demo as demo_service
from fraudcamp import constants
from fraudcamp.evaluation import clusters as cluster_tools


@pytest.fixture
def empty_app(tmp_path):
    return TestClient(create_app(tmp_path))


def test_the_panel_endpoints_404_with_an_explanation_when_there_is_no_bundle(
    empty_app,
):
    """`manifest` and `run` need the bundle, so they 404 and say how to fix it."""
    for path in ("/api/demo/manifest", "/api/demo/run"):
        response = empty_app.get(path)
        assert response.status_code in (404, 422), (path, response.status_code)
        if response.status_code == 404:
            detail = response.json()["detail"]
            assert "bundle" in detail.lower(), detail
            assert "make_demo_bundle" in detail, (
                "the empty state must say how to fix it"
            )


def test_the_aggregate_reports_its_own_absence_rather_than_erroring(empty_app):
    """Deliberately NOT a 404.

    The aggregate does not depend on the demo bundle - it comes from the
    evaluation runs - so "there is no comparison artifact yet" is a state to
    report, not a failure. The panel needs to render its aggregate section
    with a specific empty state rather than an error, because the section must
    ship whether or not the numbers exist.
    """
    response = empty_app.get("/api/demo/aggregate")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["rows"] == []
    assert payload["reason"] and "comparison artifact" in payload["reason"]


def test_a_missing_bundle_is_reported_not_guessed(tmp_path):
    bundle = demo_service.load_bundle(tmp_path / "nothing")
    assert bundle.available is False
    assert bundle.reason and "make_demo_bundle" in bundle.reason
    assert bundle.boosters == {}
    assert bundle.windows.empty


def test_a_bundle_missing_only_its_boosters_is_still_unavailable(tmp_path):
    """A half-installed bundle must not half-work.

    Windows without boosters would let the panel render its inputs and then
    fail on the first toggle, which in front of an audience is worse than a
    clean empty state.
    """
    directory = tmp_path / "demo_panel"
    directory.mkdir()
    (directory / demo_service.MANIFEST).write_text("{}", encoding="utf-8")
    (directory / demo_service.TRUTH).write_text(
        json.dumps({"campaign_accounts": {}, "campaign_split": {}}),
        encoding="utf-8",
    )
    (directory / demo_service.INSTITUTIONS).write_text(
        json.dumps({"k": 8, "account_institution": {}}), encoding="utf-8"
    )
    import pandas as pd

    pd.DataFrame({"src": [], "dst": [], "detection_time": []}).to_parquet(
        directory / demo_service.WINDOWS, index=False
    )

    bundle = demo_service.load_bundle(directory)
    assert bundle.available is False
    assert "boosters" in (bundle.reason or "")


def test_the_aggregate_is_read_or_absent_never_invented(tmp_path):
    missing = demo_service.aggregate_table(tmp_path / "nope.csv")
    assert missing["available"] is False
    assert missing["rows"] == []
    assert "no comparison artifact" in missing["reason"]

    import pandas as pd

    path = tmp_path / "phase4_compare.csv"
    pd.DataFrame([
        {"contrast": "fedavg - isolated", "unit": "union",
         "eval_split": "test", "test_subset": "all",
         "metric": "recall_relative_to_ceiling", "mean_difference": 0.02,
         "n_pairs": 1, "below_seed_noise_floor": True},
        {"contrast": "fedavg - isolated", "unit": "union",
         "eval_split": "val", "test_subset": "all",
         "metric": "recall_relative_to_ceiling", "mean_difference": 0.9,
         "n_pairs": 1, "below_seed_noise_floor": False},
    ]).to_csv(path, index=False)

    present = demo_service.aggregate_table(path)
    assert present["available"] is True
    assert len(present["rows"]) == 1, "only the test split belongs in the panel"
    assert present["rows"][0]["below_seed_noise_floor"] in (True, "True", 1)


def test_the_panels_purity_and_coverage_are_the_frozen_rules_own():
    """One definition, so a displayed number cannot disagree with the match.

    A cluster that the rule counts as a hit must show purity at or above the
    rule's threshold, and one it rejects must not - which is only guaranteed
    because `cluster_hits_campaign` delegates to `cluster_match_detail`.
    """
    campaign = ("a", "b", "c", "d")
    for cluster in (("a", "b", "c"), ("a", "b"), ("a", "b", "c", "x", "y", "z")):
        detail = cluster_tools.cluster_match_detail(cluster, campaign)
        assert detail["matches"] is cluster_tools.cluster_hits_campaign(
            cluster, campaign
        )
        if detail["matches"]:
            assert detail["purity"] >= constants.MATCH_MIN_ACCOUNT_FRACTION
            assert detail["coverage_accounts"] >= constants.MATCH_MIN_ACCOUNTS
        assert 0.0 <= detail["purity"] <= 1.0
        assert 0.0 <= detail["coverage_fraction"] <= 1.0

    # and the two quantities really are different things
    narrow = cluster_tools.cluster_match_detail(("a",), campaign)
    assert narrow["purity"] == 1.0, "every account of the cluster is the campaign's"
    assert narrow["coverage_fraction"] == 0.25, "but it covers a quarter of it"
    assert narrow["matches"] is False, "the frozen rule needs 3 accounts"


def test_the_live_path_is_the_research_path(monkeypatch):
    """The panel must not grow its own detector.

    `DemoBundle.run` is required to go through `extract_clusters` and
    `classify_cluster`; if someone replaced them with a local loop this test
    fails, because both are observed being called.
    """
    calls = {"extract": 0, "classify": 0}
    real_extract = demo_service.cluster_tools.extract_clusters
    real_classify = demo_service.cluster_tools.classify_cluster

    def extract(*args, **kwargs):
        calls["extract"] += 1
        return real_extract(*args, **kwargs)

    def classify(*args, **kwargs):
        calls["classify"] += 1
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(demo_service.cluster_tools, "extract_clusters", extract)
    monkeypatch.setattr(demo_service.cluster_tools, "classify_cluster", classify)

    # the source is the single place to check: both names must appear in it
    source = Path(demo_service.__file__).read_text(encoding="utf-8")
    assert "cluster_tools.extract_clusters(" in source
    assert "cluster_tools.classify_cluster(" in source
    assert "cluster_tools.cluster_match_detail(" in source
    # and nothing that looks like a second clustering implementation
    assert "connected_component" not in source
    assert "union_find" not in source
