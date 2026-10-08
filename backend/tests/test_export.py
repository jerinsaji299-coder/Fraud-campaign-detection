"""Artifact export on the tiny synthetic dataset."""

import json

import pandas as pd
import pytest

from conftest import N_CAMPAIGNS, N_PATTERN_TXNS, N_TOTAL_ROWS
from fraudcamp import constants, export, load_data, pipeline


@pytest.fixture
def exported(mini_dataset, tmp_path):
    result = pipeline.build(mini_dataset)
    artifacts_dir = tmp_path / "artifacts"
    written = export.export_all(result, artifacts_dir)
    return result, artifacts_dir, written


def test_every_expected_artifact_is_written(exported):
    _, artifacts_dir, written = exported
    expected = {
        "summary.json",
        "campaigns.parquet",
        "campaign_transactions.parquet",
        "institutions.json",
    } | {f"visibility/seed_{s}.parquet" for s in constants.VISIBILITY_SEEDS}
    assert set(written) == expected
    for path in written.values():
        assert path.exists()


def test_results_dir_exists_but_is_empty(exported):
    _, artifacts_dir, _ = exported
    results = artifacts_dir / "results"
    assert results.is_dir()
    assert [p.name for p in results.iterdir()] == [".gitkeep"]


def test_summary_contents(exported):
    _, artifacts_dir, _ = exported
    summary = json.loads((artifacts_dir / "summary.json").read_text())

    assert summary["rows"]["total"] == N_TOTAL_ROWS
    assert summary["rows"]["post_cutoff"] == 1
    assert summary["rows"]["pre_cutoff"] == N_TOTAL_ROWS - 1
    assert summary["campaigns"]["total"] == N_CAMPAIGNS
    assert summary["campaigns"]["pattern_transactions"] == N_PATTERN_TXNS
    assert summary["campaigns"]["eval_ok"] == 3  # campaign 2 is too small
    assert summary["campaigns"]["reassignable"] == 2  # 0 and 3
    assert summary["laundering"]["unassigned"] == 1
    assert summary["cutoff"]["timestamp"] == constants.CUTOFF.isoformat()
    assert summary["by_base_type"] == {"CYCLE": 1, "FAN-OUT": 1, "STACK": 1, "BIPARTITE": 1}
    assert summary["by_split"]["test"]["eval_ok"] == 2
    assert summary["by_split"]["test"]["shares_train_account"] == 1
    assert summary["institutions"]["k"] == constants.K_INSTITUTIONS


def test_campaigns_parquet_columns(exported):
    _, artifacts_dir, _ = exported
    camp = pd.read_parquet(artifacts_dir / "campaigns.parquet")
    required = {
        "campaign_id", "base_type", "n_txn", "n_accounts", "n_banks", "start", "end",
        "duration_h", "crosses_cutoff", "deadline", "eval_ok", "split", "group",
        "shares_train_account", "reassignable",
    }
    assert required <= set(camp.columns)
    assert len(camp) == N_CAMPAIGNS


def test_campaign_transactions_parquet(exported):
    _, artifacts_dir, _ = exported
    txns = pd.read_parquet(artifacts_dir / "campaign_transactions.parquet")

    assert len(txns) == N_PATTERN_TXNS
    required = {
        "campaign_id", "timestamp", "from_bank", "to_bank", "src", "dst",
        "amount_paid", "payment_currency", "amount_received", "receiving_currency",
        "payment_format", "is_laundering",
        "src_natural_institution", "dst_natural_institution",
    }
    assert required <= set(txns.columns)
    # only campaign rows, never the -1 filler
    assert (txns["campaign_id"] >= 0).all()
    assert txns["src"].str.contains("_").all()
    assert txns["src_natural_institution"].between(0, constants.K_INSTITUTIONS - 1).all()


def test_institutions_json(exported):
    _, artifacts_dir, _ = exported
    inst = json.loads((artifacts_dir / "institutions.json").read_text())
    assert inst["k"] == constants.K_INSTITUTIONS
    assert len(inst["institutions"]) == constants.K_INSTITUTIONS
    assert {e["institution"] for e in inst["institutions"]} == set(range(constants.K_INSTITUTIONS))
    assert sum(e["n_banks"] for e in inst["institutions"]) == inst["total_banks"]


def test_visibility_parquet_shape_and_values(exported):
    result, artifacts_dir, _ = exported
    vis = pd.read_parquet(artifacts_dir / "visibility" / "seed_0.parquet")

    required = {
        "target", "campaign_id", "account", "institution", "account_reassigned",
        "achieved_visibility", "bin", "campaign_reassigned",
    }
    assert required <= set(vis.columns)

    assert sorted(vis["target"].unique()) == sorted(constants.VISIBILITY_TARGETS)
    # every campaign appears at every target, with all of its accounts
    assert set(vis["campaign_id"].unique()) == set(range(N_CAMPAIGNS))
    expected_rows = len(constants.VISIBILITY_TARGETS) * sum(
        len(a) for a in result.accounts_by_campaign.values()
    )
    assert len(vis) == expected_rows

    assert vis["achieved_visibility"].between(0.0, 1.0).all()
    assert set(vis["bin"]) <= set(constants.VISIBILITY_BIN_ORDER)
    assert vis["institution"].between(0, constants.K_INSTITUTIONS - 1).all()

    # one achieved visibility and bin per (target, campaign)
    per_campaign = vis.groupby(["target", "campaign_id"])["achieved_visibility"].nunique()
    assert (per_campaign == 1).all()


def test_visibility_target_one_is_fully_visible(exported):
    _, artifacts_dir, _ = exported
    vis = pd.read_parquet(artifacts_dir / "visibility" / "seed_0.parquet")
    reassigned = vis[(vis["target"] == 1.0) & vis["campaign_reassigned"]]
    assert (reassigned["achieved_visibility"] == 1.0).all()
    assert (reassigned["bin"] == "100").all()


def test_shared_account_has_one_institution_per_run(exported):
    _, artifacts_dir, _ = exported
    vis = pd.read_parquet(artifacts_dir / "visibility" / "seed_0.parquet")
    # 1_A belongs to campaigns 0 and 3; conflict resolution must give it a
    # single institution within each target run
    shared = vis[vis["account"] == "1_A"]
    assert set(shared["campaign_id"]) == {0, 3}
    assert (shared.groupby("target")["institution"].nunique() == 1).all()


def test_export_is_deterministic(mini_dataset, tmp_path):
    first = export.export_all(pipeline.build(mini_dataset), tmp_path / "a")
    second = export.export_all(pipeline.build(mini_dataset), tmp_path / "b")
    for seed in constants.VISIBILITY_SEEDS:
        key = f"visibility/seed_{seed}.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(first[key]), pd.read_parquet(second[key]))
    pd.testing.assert_frame_equal(
        pd.read_parquet(first["campaigns.parquet"]), pd.read_parquet(second["campaigns.parquet"])
    )


def test_seeds_are_actually_different(mini_dataset, tmp_path):
    written = export.export_all(pipeline.build(mini_dataset), tmp_path / "artifacts")
    tables = {
        s: pd.read_parquet(written[f"visibility/seed_{s}.parquet"])
        for s in constants.VISIBILITY_SEEDS
    }
    # at least one seed pair must differ somewhere, or the seeding is broken
    assignments = {s: tuple(t["institution"]) for s, t in tables.items()}
    assert len(set(assignments.values())) > 1


def test_artifacts_dir_comes_from_config(mini_dataset):
    config = load_data.load_config(mini_dataset)
    artifacts_dir = load_data.resolve_path(mini_dataset, config["artifacts_dir"])
    written = export.export_all(pipeline.build(mini_dataset), artifacts_dir)
    assert written["summary.json"].parent == artifacts_dir
