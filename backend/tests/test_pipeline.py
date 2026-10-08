"""End-to-end pipeline run on the tiny synthetic dataset."""

import pandas as pd

from conftest import N_CAMPAIGNS, N_PATTERN_TXNS, N_TOTAL_ROWS
from fraudcamp import constants, pipeline, splits


def test_build_loads_and_matches_everything(mini_dataset):
    result = pipeline.build(mini_dataset)
    assert len(result.full_df) == N_TOTAL_ROWS
    assert len(result.pattern_df) == N_PATTERN_TXNS
    assert len(result.camp) == N_CAMPAIGNS
    # every pattern transaction found its row in the transaction table
    assert int((result.full_df["campaign_id"] != -1).sum()) == N_PATTERN_TXNS


def test_cutoff_excludes_the_tail(mini_dataset):
    result = pipeline.build(mini_dataset)
    assert len(result.pre_cutoff_df) == N_TOTAL_ROWS - 1  # one Sept 12 row
    assert (result.pre_cutoff_df["ts"] < constants.CUTOFF).all()


def test_splits_groups_and_flags(mini_dataset):
    camp = pipeline.build(mini_dataset).camp.set_index("campaign_id")

    assert camp.loc[0, "split"] == "train"
    assert camp.loc[1, "split"] == "test"
    assert camp.loc[2, "split"] == "train"
    assert camp.loc[3, "split"] == "test"

    assert camp.loc[1, "group"] == "hub"  # FAN-OUT
    assert camp.loc[0, "group"] in ("fragmentable", "unfragmentable")

    assert bool(camp.loc[0, "eval_ok"]) is True
    assert bool(camp.loc[2, "eval_ok"]) is False  # only 2 transactions

    # campaign 3 is a test campaign sharing account 1_A with train campaign 0
    assert bool(camp.loc[3, "shares_train_account"]) is True
    assert bool(camp.loc[1, "shares_train_account"]) is False


def test_reassignable_is_topology_not_group(mini_dataset):
    camp = pipeline.build(mini_dataset).camp
    mask = splits.reassignable_mask(camp)
    reassignable = set(camp[mask]["campaign_id"])
    # 0 and 3 are eval_ok non-hub; 1 is hub; 2 is not eval_ok
    assert reassignable == {0, 3}

    # a campaign relabelled "unfragmentable" must stay reassignable
    relabelled = camp.copy()
    relabelled.loc[relabelled["campaign_id"] == 0, "group"] = "unfragmentable"
    assert set(relabelled[splits.reassignable_mask(relabelled)]["campaign_id"]) == {0, 3}


def test_every_bank_gets_an_institution(mini_dataset):
    result = pipeline.build(mini_dataset)
    all_banks = set(result.full_df["From Bank"]) | set(result.full_df["To Bank"])
    # bank 99 appears only in a post-cutoff row, so it has zero pre-cutoff
    # volume but must still be assigned
    assert 99 in all_banks
    assert all_banks <= set(result.bank2inst)
    assert set(result.bank2inst.values()) <= set(range(constants.K_INSTITUTIONS))


def test_build_is_deterministic(mini_dataset):
    a = pipeline.build(mini_dataset)
    b = pipeline.build(mini_dataset)
    assert a.bank2inst == b.bank2inst
    assert a.loads == b.loads
    assert a.floor_achieved == b.floor_achieved
    pd.testing.assert_frame_equal(a.camp, b.camp)


def test_run_visibility_covers_every_campaign(mini_dataset):
    result = pipeline.build(mini_dataset)
    ordered = pipeline.ordered_reassignable(
        result.camp, result.edges_by_campaign, result.accounts_by_campaign
    )
    _, achieved = pipeline.run_visibility(
        ordered, result.edges_by_campaign, result.bank2inst, target=0.5, seed=0
    )
    assert set(achieved) == set(range(N_CAMPAIGNS))
    assert all(0.0 <= v <= 1.0 for v in achieved.values())
    # a hub campaign is fully visible to the hub account's institution
    assert achieved[1] == 1.0


def test_shared_account_resolved_by_earliest_campaign(mini_dataset):
    result = pipeline.build(mini_dataset)
    ordered = pipeline.ordered_reassignable(
        result.camp, result.edges_by_campaign, result.accounts_by_campaign
    )
    # campaign 0 (Sept 1) must be processed before campaign 3 (Sept 7)
    assert [cid for cid, _, _ in ordered] == [0, 3]

    assignment, _ = pipeline.run_visibility(
        ordered, result.edges_by_campaign, result.bank2inst, target=0.5, seed=0
    )
    # the shared account has exactly one institution across both campaigns
    assert "1_A" in assignment
