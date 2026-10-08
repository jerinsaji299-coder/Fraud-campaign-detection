"""Write the small artifacts the API serves, from a completed pipeline run.

The full transaction table is far too large to ship or serve, so
everything here is a derived summary, the campaign ground truth, or the
~3.2k transactions that actually belong to a campaign. See README
"Artifacts" for the exact contents of each file.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from . import constants, institutions, pipeline, visibility

CAMPAIGN_TRANSACTION_COLUMNS = {
    "campaign_id": "campaign_id",
    # the parsed datetime, not the raw "2022/09/01 00:10" string, so the
    # API and frontend get a real timestamp to sort and replay on
    "ts": "timestamp",
    "From Bank": "from_bank",
    "To Bank": "to_bank",
    "src": "src",
    "dst": "dst",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def build_summary(result: pipeline.PipelineResult) -> dict:
    full_df = result.full_df
    camp = result.camp

    post_cutoff = full_df[full_df["ts"] >= constants.CUTOFF]
    eval_ok = camp[camp["eval_ok"]]

    by_split = {}
    for split in constants.SPLIT_ORDER:
        rows = camp[camp["split"] == split]
        rows_eval = rows[rows["eval_ok"]]
        by_split[split] = {
            "campaigns": int(len(rows)),
            "eval_ok": int(len(rows_eval)),
            "crosses_cutoff": int(rows_eval["crosses_cutoff"].sum()),
            "shares_train_account": int(rows_eval["shares_train_account"].sum()),
        }

    laundering_rows = int(full_df["Is Laundering"].sum())
    in_campaign = int(((full_df["Is Laundering"] == 1) & (full_df["campaign_id"] != -1)).sum())

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "trans_csv": str(result.trans_csv),
            "patterns_txt": str(result.patterns_txt),
        },
        "rows": {
            "total": int(len(full_df)),
            "pre_cutoff": int(len(result.pre_cutoff_df)),
            "post_cutoff": int(len(post_cutoff)),
        },
        "date_range": {
            "min": full_df["ts"].min().isoformat(),
            "max": full_df["ts"].max().isoformat(),
        },
        "banks": {
            "total": int(len(set(full_df["From Bank"]) | set(full_df["To Bank"]))),
        },
        "laundering": {
            "rows": laundering_rows,
            "in_campaign": in_campaign,
            "unassigned": laundering_rows - in_campaign,
        },
        "campaigns": {
            "total": int(len(camp)),
            "eval_ok": int(len(eval_ok)),
            "pattern_transactions": int(len(result.pattern_df)),
            "reassignable": int(camp["reassignable"].sum()),
        },
        "by_base_type": _counts(camp["base_type"]),
        "by_split": by_split,
        "by_group": _counts(camp["group"]),
        "by_group_eval_ok": _counts(eval_ok["group"]),
        "cutoff": {
            "timestamp": constants.CUTOFF.isoformat(),
            "rows_excluded": int(len(post_cutoff)),
            "laundering_rows_excluded": int(post_cutoff["Is Laundering"].sum()),
            "laundering_share_excluded": (
                float(post_cutoff["Is Laundering"].mean()) if len(post_cutoff) else 0.0
            ),
            "laundering_share_overall": float(full_df["Is Laundering"].mean()),
        },
        "transactions_per_day": {
            str(day): int(n) for day, n in full_df["ts"].dt.date.value_counts().sort_index().items()
        },
        "campaigns_starting_per_day": {
            str(day): int(n) for day, n in camp["start"].dt.date.value_counts().sort_index().items()
        },
        "institutions": {
            "k": constants.K_INSTITUTIONS,
        },
        "visibility": {
            "targets": constants.VISIBILITY_TARGETS,
            "seeds": constants.VISIBILITY_SEEDS,
            "bins": constants.VISIBILITY_BIN_ORDER,
        },
    }


def build_campaign_transactions(result: pipeline.PipelineResult) -> pd.DataFrame:
    """Only the transactions that belong to a campaign, with each
    endpoint's natural (bank-based) institution attached."""
    rows = result.full_df[result.full_df["campaign_id"] != -1].copy()
    out = rows[list(CAMPAIGN_TRANSACTION_COLUMNS)].rename(columns=CAMPAIGN_TRANSACTION_COLUMNS)
    out["src_natural_institution"] = [
        institutions.natural_institution(a, result.bank2inst) for a in rows["src"]
    ]
    out["dst_natural_institution"] = [
        institutions.natural_institution(a, result.bank2inst) for a in rows["dst"]
    ]
    out = out.sort_values(["campaign_id", "timestamp"]).reset_index(drop=True)
    # stable per-campaign transaction id: 0-based position in the
    # campaign's time-ordered transaction list. Defined here, in the
    # pipeline, so the API and frontend can both refer to it.
    out["txn_index"] = out.groupby("campaign_id").cumcount()
    return out


def build_institutions(result: pipeline.PipelineResult) -> dict:
    banks_per_institution: dict[int, int] = {i: 0 for i in range(constants.K_INSTITUTIONS)}
    for inst in result.bank2inst.values():
        banks_per_institution[inst] += 1
    return {
        "k": constants.K_INSTITUTIONS,
        "institutions": [
            {
                "institution": i,
                "load": int(result.loads[i]),
                "n_banks": int(banks_per_institution[i]),
            }
            for i in range(constants.K_INSTITUTIONS)
        ],
        "total_banks": int(len(result.bank2inst)),
        "note": (
            "load = number of pre-cutoff transaction endpoints assigned to the "
            "institution; banks are assigned largest-volume-first to the "
            "currently lightest institution"
        ),
    }


def build_visibility_table(result: pipeline.PipelineResult, seed: int) -> pd.DataFrame:
    """One row per (target, campaign_id, account) for a single seed:
    the account's institution under that run, plus the campaign's achieved
    visibility and bin recomputed from the final global assignment."""
    ordered = pipeline.ordered_reassignable(
        result.camp, result.edges_by_campaign, result.accounts_by_campaign
    )
    reassigned_ids = {cid for cid, _, _ in ordered}

    rows = []
    for target in constants.VISIBILITY_TARGETS:
        assignment, achieved = pipeline.run_visibility(
            ordered, result.edges_by_campaign, result.bank2inst, target, seed
        )
        for cid, accounts in sorted(result.accounts_by_campaign.items()):
            v = achieved[cid]
            bin_label = visibility.bin_of(v)
            was_reassigned = cid in reassigned_ids
            for account in sorted(accounts):
                rows.append(
                    (
                        target,
                        cid,
                        account,
                        visibility.institution_of(account, assignment, result.bank2inst),
                        account in assignment,
                        v,
                        bin_label,
                        was_reassigned,
                    )
                )

    return pd.DataFrame(
        rows,
        columns=[
            "target",
            "campaign_id",
            "account",
            "institution",
            "account_reassigned",
            "achieved_visibility",
            "bin",
            "campaign_reassigned",
        ],
    )


def export_all(result: pipeline.PipelineResult, artifacts_dir: str | Path) -> dict[str, Path]:
    """Write every artifact. `results/` is created but deliberately left
    empty — it is filled by Phase 5, and the API 404s until then."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "visibility").mkdir(exist_ok=True)
    results_dir = artifacts_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / ".gitkeep").touch()

    written: dict[str, Path] = {}

    summary_path = artifacts_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(build_summary(result), f, indent=2)
    written["summary.json"] = summary_path

    campaigns_path = artifacts_dir / "campaigns.parquet"
    result.camp.to_parquet(campaigns_path, index=False)
    written["campaigns.parquet"] = campaigns_path

    ct_path = artifacts_dir / "campaign_transactions.parquet"
    build_campaign_transactions(result).to_parquet(ct_path, index=False)
    written["campaign_transactions.parquet"] = ct_path

    institutions_path = artifacts_dir / "institutions.json"
    with open(institutions_path, "w", encoding="utf-8") as f:
        json.dump(build_institutions(result), f, indent=2)
    written["institutions.json"] = institutions_path

    for seed in constants.VISIBILITY_SEEDS:
        path = artifacts_dir / "visibility" / f"seed_{seed}.parquet"
        build_visibility_table(result, seed).to_parquet(path, index=False)
        written[f"visibility/seed_{seed}.parquet"] = path

    return written
