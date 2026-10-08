"""End-to-end orchestration of the research pipeline.

Shared by `scripts/run_pipeline.py` (which prints the regression numbers)
and `export.py` (which writes the artifacts), so the two can never drift
apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import campaigns, constants, institutions, load_data, splits, visibility


@dataclass
class PipelineResult:
    config_path: Path
    trans_csv: Path
    patterns_txt: Path
    full_df: pd.DataFrame
    pre_cutoff_df: pd.DataFrame
    pattern_df: pd.DataFrame
    camp: pd.DataFrame
    bank2inst: dict[int, int]
    loads: list[int]
    edges_by_campaign: dict[int, list[tuple[str, str]]]
    #: Sorted tuples, never sets: these feed the seeded visibility search.
    accounts_by_campaign: dict[int, tuple[str, ...]]
    floor_achieved: dict[int, float] = field(default_factory=dict)


def _edges_and_accounts(pattern_df: pd.DataFrame):
    """Per-campaign edge lists and account collections.

    Accounts come back as a **sorted tuple**, not a set: they feed the
    seeded visibility search, and Python's per-process string hashing makes
    set iteration order vary between runs. `groupby` is left at its default
    `sort=True`, so campaign ids are enumerated in order too, and each
    campaign's edge order follows the pattern file.
    """
    edges: dict[int, list[tuple[str, str]]] = {}
    accounts: dict[int, tuple[str, ...]] = {}
    for cid, g in pattern_df.groupby("campaign_id", sort=True):
        edges[int(cid)] = list(zip(g["src"], g["dst"]))
        accounts[int(cid)] = tuple(sorted(set(g["src"]) | set(g["dst"])))
    return edges, accounts


def ordered_reassignable(camp: pd.DataFrame, edges_by_campaign, accounts_by_campaign):
    """The campaigns whose accounts get reassigned, in start-time order
    (earliest first), which is what makes shared-account conflict
    resolution favour the earliest campaign.

    Reassignment is a pure topology test — eval_ok and a non-hub
    base_type. Campaigns later relabelled "unfragmentable" stay in this
    set and keep receiving the same treatment; the relabel is an analysis
    grouping only.
    """
    mask = splits.reassignable_mask(camp)
    ordered = camp[mask].sort_values(["start", "campaign_id"])
    return [
        (int(cid), edges_by_campaign[int(cid)], accounts_by_campaign[int(cid)])
        for cid in ordered["campaign_id"]
    ]


def run_visibility(
    ordered,
    edges_by_campaign: dict[int, list[tuple[str, str]]],
    bank2inst: dict[int, int],
    target: float,
    seed: int,
) -> tuple[dict[str, int], dict[int, float]]:
    """Build the global account -> institution assignment for one (seed,
    target), then recompute achieved visibility for EVERY campaign from
    that final assignment."""
    assignment = visibility.build_global_assignment(ordered, target, seed)
    achieved = visibility.recompute_achieved(
        [(cid, edges) for cid, edges in edges_by_campaign.items()], assignment, bank2inst
    )
    return assignment, achieved


def build(config_path: str | Path) -> PipelineResult:
    """Run every deterministic stage of the pipeline: load, parse, match,
    campaign table, splits, groups, institutions, the unfragmentable floor
    run, and shared-account flags."""
    config_path = Path(config_path)
    config = load_data.load_config(config_path)
    trans_csv = load_data.resolve_path(config_path, config["data"]["trans_csv"])
    patterns_txt = load_data.resolve_path(config_path, config["data"]["patterns_txt"])

    full_df = load_data.load_transactions(trans_csv)
    attempts = campaigns.parse_patterns(patterns_txt)
    pattern_df = campaigns.build_pattern_df(attempts)
    full_df = campaigns.match_patterns(pattern_df, full_df)
    pre_cutoff_df = load_data.apply_cutoff(full_df)

    camp = campaigns.build_campaign_table(pattern_df)
    camp = campaigns.add_eval_fields(camp)
    camp = splits.assign_split(camp)
    camp = splits.assign_group(camp)
    camp = splits.mark_shared_accounts(camp, pattern_df)

    all_banks = sorted(set(full_df["From Bank"]) | set(full_df["To Bank"]))
    vol = institutions.bank_volume(pre_cutoff_df, all_banks=all_banks)
    bank2inst, loads = institutions.assign_institutions(vol, k=constants.K_INSTITUTIONS)

    edges_by_campaign, accounts_by_campaign = _edges_and_accounts(pattern_df)
    ordered = ordered_reassignable(camp, edges_by_campaign, accounts_by_campaign)

    # Floor run: the one that decides which campaigns are "unfragmentable".
    _, floor_achieved = run_visibility(
        ordered,
        edges_by_campaign,
        bank2inst,
        constants.UNFRAGMENTABLE_TARGET,
        constants.UNFRAGMENTABLE_SEED,
    )
    # Only campaigns that were actually reassigned have a meaningful floor;
    # for anything else this number is just its natural visibility.
    reassigned_ids = {cid for cid, _, _ in ordered}
    camp = splits.mark_unfragmentable(
        camp, {cid: v for cid, v in floor_achieved.items() if cid in reassigned_ids}
    )
    camp["reassignable"] = splits.reassignable_mask(camp)

    return PipelineResult(
        config_path=config_path,
        trans_csv=Path(trans_csv),
        patterns_txt=Path(patterns_txt),
        full_df=full_df,
        pre_cutoff_df=pre_cutoff_df,
        pattern_df=pattern_df,
        camp=camp,
        bank2inst=bank2inst,
        loads=loads,
        edges_by_campaign=edges_by_campaign,
        accounts_by_campaign=accounts_by_campaign,
        floor_achieved=floor_achieved,
    )
