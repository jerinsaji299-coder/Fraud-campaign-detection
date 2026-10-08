"""API tests against a tiny fixture artifact set built by the real export."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import artifacts as artifacts_service
from conftest import N_CAMPAIGNS, N_PATTERN_TXNS
from fraudcamp import constants, export, pipeline


@pytest.fixture
def artifacts_dir(mini_dataset, tmp_path):
    out = tmp_path / "api_artifacts"
    export.export_all(pipeline.build(mini_dataset), out)
    return out


@pytest.fixture
def client(artifacts_dir):
    return TestClient(create_app(artifacts_dir))


@pytest.fixture
def empty_client(tmp_path):
    """A server pointed at a directory with no artifacts at all."""
    return TestClient(create_app(tmp_path / "nothing"))


# --- health -----------------------------------------------------------------


def test_health_reports_artifacts(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["core_artifacts_available"] is True
    assert body["results_available"] is False
    assert body["artifacts"]["campaigns.parquet"] is True
    assert body["artifacts"]["results/runs.parquet"] is False


def test_health_works_without_artifacts(empty_client):
    body = empty_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["core_artifacts_available"] is False
    assert all(v is False for v in body["artifacts"].values())


def test_health_lists_what_is_missing_and_how_to_fix_it(empty_client):
    body = empty_client.get("/api/health").json()
    assert sorted(body["missing_artifacts"]) == sorted(artifacts_service.CORE_ARTIFACTS)
    assert body["nested_artifacts_dir"] is None
    assert "--export" in body["hint"]
    assert "restart" in body["hint"]


# --- the "extracted one level too deep" trap ----------------------------------


def test_nested_artifacts_are_detected(tmp_path, mini_dataset):
    """Windows' "Extract All" produces artifacts/artifacts/summary.json, which
    otherwise looks like every artifact being mysteriously absent."""
    root = tmp_path / "artifacts"
    export.export_all(pipeline.build(mini_dataset), root / "artifacts")

    found = artifacts_service.find_nested_artifacts(root)
    assert found == root / "artifacts"


def test_nested_artifacts_reported_by_health(tmp_path, mini_dataset):
    root = tmp_path / "artifacts"
    export.export_all(pipeline.build(mini_dataset), root / "artifacts")

    client = TestClient(create_app(root))
    body = client.get("/api/health").json()

    assert body["core_artifacts_available"] is False
    assert body["nested_artifacts_dir"] == str(root / "artifacts")
    assert "one directory too deep" in body["hint"]
    assert "restart" in body["hint"]


def test_nested_detection_ignores_the_real_subdirectories(artifacts_dir):
    """visibility/ and results/ belong inside the artifacts directory and
    must never be reported as an accidental nesting level."""
    assert artifacts_service.find_nested_artifacts(artifacts_dir) is None


def test_no_nesting_claimed_when_everything_is_in_place(client):
    body = client.get("/api/health").json()
    assert body["core_artifacts_available"] is True
    assert body["missing_artifacts"] == []
    assert body["nested_artifacts_dir"] is None
    assert body["hint"] is None


def test_nested_detection_on_an_absent_directory(tmp_path):
    assert artifacts_service.find_nested_artifacts(tmp_path / "nope") is None


def test_endpoints_explain_missing_artifacts(empty_client):
    r = empty_client.get("/api/summary")
    assert r.status_code == 503
    assert "has not been exported yet" in r.json()["detail"]


# --- summary ----------------------------------------------------------------


def test_summary_matches_the_artifact(client):
    body = client.get("/api/summary").json()
    assert body["campaigns"]["total"] == N_CAMPAIGNS
    assert body["campaigns"]["pattern_transactions"] == N_PATTERN_TXNS
    assert body["rows"]["post_cutoff"] == 1
    assert body["cutoff"]["timestamp"] == constants.CUTOFF.isoformat()
    assert body["by_split"]["test"]["eval_ok"] == 2
    assert body["institutions"]["k"] == constants.K_INSTITUTIONS


# --- campaigns --------------------------------------------------------------


def test_campaign_list_paginates(client):
    body = client.get("/api/campaigns", params={"page_size": 2}).json()
    assert body["total"] == N_CAMPAIGNS
    assert body["page_size"] == 2
    assert body["pages"] == 2
    assert len(body["items"]) == 2

    page2 = client.get("/api/campaigns", params={"page_size": 2, "page": 2}).json()
    assert [c["campaign_id"] for c in page2["items"]] == [2, 3]


def test_campaign_list_filters(client):
    hub = client.get("/api/campaigns", params={"group": "hub"}).json()
    assert [c["campaign_id"] for c in hub["items"]] == [1]

    evaluable = client.get("/api/campaigns", params={"eval_ok": True}).json()
    assert evaluable["total"] == 3

    shared = client.get("/api/campaigns", params={"shares_train_account": True}).json()
    assert [c["campaign_id"] for c in shared["items"]] == [3]

    by_type = client.get("/api/campaigns", params={"base_type": "CYCLE"}).json()
    assert by_type["total"] == 1

    by_split = client.get("/api/campaigns", params={"split": "test"}).json()
    assert by_split["total"] == 2


def test_campaign_list_sorts(client):
    desc = client.get("/api/campaigns", params={"sort_by": "n_txn", "sort_dir": "desc"}).json()
    counts = [c["n_txn"] for c in desc["items"]]
    assert counts == sorted(counts, reverse=True)


def test_campaign_list_rejects_bad_sort_key(client):
    assert client.get("/api/campaigns", params={"sort_by": "group"}).status_code == 422


def test_campaign_search_by_id(client):
    # a digits-only query is the campaign id, not "any account containing a 1"
    by_id = client.get("/api/campaigns", params={"search": "1"}).json()
    assert [c["campaign_id"] for c in by_id["items"]] == [1]


def test_campaign_search_by_account(client):
    # account 1_A appears in campaigns 0 and 3
    by_account = client.get("/api/campaigns", params={"search": "1_A"}).json()
    assert [c["campaign_id"] for c in by_account["items"]] == [0, 3]

    # case-insensitive
    assert client.get("/api/campaigns", params={"search": "1_a"}).json()["total"] == 2

    # the account part of an id is searchable on its own, digits included
    by_part = client.get("/api/campaigns", params={"search": "L1"}).json()
    assert [c["campaign_id"] for c in by_part["items"]] == [1]


def test_campaign_search_with_no_hits(client):
    assert client.get("/api/campaigns", params={"search": "nosuchaccount"}).json()["total"] == 0


def test_campaign_detail(client):
    body = client.get("/api/campaigns/0").json()
    assert body["campaign"]["campaign_id"] == 0
    assert body["campaign"]["base_type"] == "CYCLE"
    assert len(body["transactions"]) == 4

    # time-ordered, with stable indices
    assert [t["txn_index"] for t in body["transactions"]] == [0, 1, 2, 3]
    timestamps = [t["timestamp"] for t in body["transactions"]]
    assert timestamps == sorted(timestamps)

    accounts = {a["account"] for a in body["accounts"]}
    assert accounts == {"1_A", "2_B", "3_C", "4_D"}
    for account in body["accounts"]:
        assert 0 <= account["natural_institution"] < constants.K_INSTITUTIONS


def test_campaign_detail_404(client):
    r = client.get("/api/campaigns/999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# --- visibility -------------------------------------------------------------


def test_campaign_visibility(client):
    body = client.get(
        "/api/campaigns/0/visibility", params={"seed": 0, "target": 1.0}
    ).json()

    assert body["campaign_id"] == 0
    assert body["achieved_visibility"] == 1.0
    assert body["bin"] == "100"
    assert body["campaign_reassigned"] is True
    assert body["n_transactions"] == 4
    assert {a["account"] for a in body["accounts"]} == {"1_A", "2_B", "3_C", "4_D"}

    # at target 1.0 one institution sees the entire campaign
    best = max(inst["n_visible"] for inst in body["institutions"])
    assert best == body["n_transactions"]
    assert len(body["institutions"]) == constants.K_INSTITUTIONS


def test_visible_indices_are_consistent_with_detail(client):
    detail = client.get("/api/campaigns/0").json()
    vis = client.get("/api/campaigns/0/visibility", params={"seed": 0, "target": 0.5}).json()

    valid = {t["txn_index"] for t in detail["transactions"]}
    for inst in vis["institutions"]:
        assert set(inst["visible_txn_indices"]) <= valid
        assert inst["n_visible"] == len(inst["visible_txn_indices"])

    # achieved visibility is exactly the best institution's share
    best = max(inst["n_visible"] for inst in vis["institutions"])
    assert best / vis["n_transactions"] == pytest.approx(vis["achieved_visibility"])


def test_visibility_rejects_unknown_seed_and_target(client):
    assert client.get("/api/campaigns/0/visibility", params={"seed": 99}).status_code == 422
    assert (
        client.get("/api/campaigns/0/visibility", params={"seed": 0, "target": 0.3}).status_code
        == 422
    )


def test_visibility_distribution(client):
    body = client.get("/api/visibility/distribution", params={"seed": 0}).json()

    assert body["seed"] == 0
    assert body["targets"] == constants.VISIBILITY_TARGETS
    assert len(body["per_target"]) == len(constants.VISIBILITY_TARGETS)

    for entry in body["per_target"]:
        # every campaign appears, including hub and non-eval_ok ones
        assert len(entry["campaigns"]) == N_CAMPAIGNS
        assert sum(entry["bin_counts"].values()) == N_CAMPAIGNS
        assert set(entry["bin_counts"]) == set(constants.VISIBILITY_BIN_ORDER)
        for group_counts in entry["bin_counts_by_group"].values():
            assert set(group_counts) == set(constants.VISIBILITY_BIN_ORDER)

    hub_points = [
        p
        for entry in body["per_target"]
        for p in entry["campaigns"]
        if p["group"] == "hub"
    ]
    assert hub_points and all(p["achieved_visibility"] == 1.0 for p in hub_points)
    assert all(p["reassigned"] is False for p in hub_points)


def test_visibility_distribution_rejects_unknown_seed(client):
    assert client.get("/api/visibility/distribution", params={"seed": 7}).status_code == 422


# --- methodology ------------------------------------------------------------


def test_methodology_is_served_from_constants(client):
    body = client.get("/api/methodology").json()
    entries = {e["key"]: e for e in body["entries"]}

    assert body["n_entries"] == len(body["entries"])
    assert {"cutoff", "splits", "groups", "institutions", "visibility", "visibility_bins"} <= set(
        entries
    )
    for entry in body["entries"]:
        assert entry["definition"] and entry["reason"]

    assert entries["cutoff"]["value"]["cutoff"] == constants.CUTOFF.isoformat()
    assert entries["institutions"]["value"]["k"] == constants.K_INSTITUTIONS
    assert entries["visibility"]["value"]["targets"] == constants.VISIBILITY_TARGETS
    assert entries["visibility_bins"]["value"]["order"] == constants.VISIBILITY_BIN_ORDER


def test_methodology_works_without_artifacts(empty_client):
    assert empty_client.get("/api/methodology").status_code == 200


# --- results ----------------------------------------------------------------


def test_results_404_until_phase_5(client):
    for path in ("/api/results/summary", "/api/results/detections"):
        r = client.get(path)
        assert r.status_code == 404
        assert "not available yet" in r.json()["detail"]


def test_results_detections_404_with_campaign_filter(client):
    r = client.get("/api/results/detections", params={"campaign_id": 0})
    assert r.status_code == 404


# --- app plumbing -----------------------------------------------------------


def test_openapi_docs_are_served(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_cors_allows_the_vite_dev_server(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
