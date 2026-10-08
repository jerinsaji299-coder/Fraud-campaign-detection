import pandas as pd

from fraudcamp import splits


def _camp(start_dates, base_types):
    return pd.DataFrame(
        {
            "campaign_id": list(range(len(start_dates))),
            "base_type": base_types,
            "start": [pd.Timestamp(d) for d in start_dates],
        }
    )


def test_assign_split_boundaries():
    camp = _camp(
        ["2022-09-04 23:59", "2022-09-05 00:00", "2022-09-05 23:59",
         "2022-09-07 23:59", "2022-09-08 00:00", "2022-09-10 23:59",
         "2022-09-11 00:00"],
        ["CYCLE"] * 7,
    )
    out = splits.assign_split(camp)
    assert list(out["split"]) == ["train", "val", "val", "test", "stress", "stress", None]


def test_assign_group_hub_vs_fragmentable():
    camp = _camp(["2022-09-01"] * 4, ["FAN-IN", "FAN-OUT", "GATHER-SCATTER", "CYCLE"])
    out = splits.assign_group(camp)
    assert list(out["group"]) == ["hub", "hub", "hub", "fragmentable"]


def test_mark_unfragmentable():
    camp = _camp(["2022-09-01"] * 2, ["CYCLE", "CYCLE"])
    camp = splits.assign_group(camp)
    out = splits.mark_unfragmentable(camp, {0: 0.95, 1: 0.5})
    assert out.loc[0, "group"] == "unfragmentable"
    assert out.loc[1, "group"] == "fragmentable"


def test_control_group_mask():
    camp = pd.DataFrame({"group": ["hub", "unfragmentable", "fragmentable"]})
    assert list(splits.control_group_mask(camp)) == [True, True, False]


def test_mark_shared_accounts():
    camp = pd.DataFrame(
        {
            "campaign_id": [0, 1, 2],
            "split": ["train", "test", "test"],
        }
    )
    pattern_df = pd.DataFrame(
        {
            "campaign_id": [0, 1, 2],
            "src": ["acc_shared", "acc_shared", "acc_other"],
            "dst": ["acc_x", "acc_y", "acc_z"],
        }
    )
    out = splits.mark_shared_accounts(camp, pattern_df)
    assert bool(out.loc[out.campaign_id == 1, "shares_train_account"].iloc[0]) is True
    assert bool(out.loc[out.campaign_id == 2, "shares_train_account"].iloc[0]) is False
