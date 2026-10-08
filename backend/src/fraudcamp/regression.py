"""The full-data regression expectations, in one place.

These are the numbers in README "Verified numbers", taken from the reference
notebook's own cell outputs. Both the full-data test suite
(`tests/test_fulldata.py`) and the Kaggle runner
(`scripts/kaggle/kaggle_runner.py`) read them from here, so the pass/fail
table and the tests can never disagree about what is expected.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import constants, pipeline, splits


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    expected: int
    evidence: str


#: Every regression number, in the order the report prints them.
METRICS: tuple[Metric, ...] = (
    Metric("total_rows", "Total transaction rows", 5_078_345, "notebook cell b42159f4"),
    Metric(
        "rows_on_or_after_cutoff",
        f"Rows on/after CUTOFF ({constants.CUTOFF:%Y-%m-%d})",
        1_108,
        "notebook cell 280d9179",
    ),
    Metric("campaigns", "Campaigns in the pattern file", 370, "notebook cell cd70c51d"),
    Metric(
        "pattern_transactions",
        "Pattern-file transactions",
        3_209,
        "notebook cell cd70c51d",
    ),
    Metric(
        "matched_pattern_transactions",
        "Pattern transactions matched into the transaction table",
        3_209,
        "notebook cell cd70c51d (0 unmatched)",
    ),
    Metric("laundering_rows", "Laundering rows (Is Laundering == 1)", 5_177, "notebook cell b42159f4"),
    Metric(
        "unassigned_laundering_rows",
        "Unassigned laundering rows (label 1, no campaign)",
        1_968,
        "README decision log (5,177 - 3,209)",
    ),
    Metric("eval_ok_total", "eval_ok campaigns", 258, "notebook cell 280d9179"),
    Metric("eval_ok_train", "eval_ok campaigns in train", 100, "notebook cell b7e46dec"),
    Metric("eval_ok_val", "eval_ok campaigns in val", 26, "notebook cell b7e46dec"),
    Metric("eval_ok_test", "eval_ok campaigns in test", 56, "notebook cell b7e46dec"),
    Metric("eval_ok_stress", "eval_ok campaigns in stress", 76, "notebook cell b7e46dec"),
    Metric(
        "reassignable_total",
        "eval_ok fragmentable (reassignable) campaigns",
        152,
        "notebook cell b7e46dec",
    ),
    Metric(
        "reassignable_test",
        "eval_ok fragmentable campaigns in test",
        37,
        "notebook cell b7e46dec",
    ),
    Metric(
        "reassignable_test_shares_train_account",
        "Fragmentable test campaigns sharing a train account",
        7,
        "notebook cell cb0caa40",
    ),
    Metric("base_type_CYCLE", "base_type CYCLE", 54, "notebook cell 11875435"),
    Metric("base_type_GATHER-SCATTER", "base_type GATHER-SCATTER", 51, "notebook cell 11875435"),
    Metric("base_type_BIPARTITE", "base_type BIPARTITE", 49, "notebook cell 11875435"),
    Metric("base_type_FAN-OUT", "base_type FAN-OUT", 48, "notebook cell 11875435"),
    Metric("base_type_SCATTER-GATHER", "base_type SCATTER-GATHER", 44, "notebook cell 11875435"),
    Metric("base_type_STACK", "base_type STACK", 43, "notebook cell 11875435"),
    Metric("base_type_RANDOM", "base_type RANDOM", 41, "notebook cell 11875435"),
    Metric("base_type_FAN-IN", "base_type FAN-IN", 40, "notebook cell 11875435"),
)

EXPECTED: dict[str, int] = {metric.key: metric.expected for metric in METRICS}


def compute_actuals(result: pipeline.PipelineResult) -> dict[str, int]:
    """Every regression number, recomputed from a completed pipeline run."""
    full_df = result.full_df
    camp = result.camp
    reassignable = splits.reassignable_mask(camp)

    laundering = full_df["Is Laundering"] == 1
    in_campaign = full_df["campaign_id"] != -1

    actuals: dict[str, int] = {
        "total_rows": int(len(full_df)),
        "rows_on_or_after_cutoff": int((full_df["ts"] >= constants.CUTOFF).sum()),
        "campaigns": int(len(camp)),
        "pattern_transactions": int(len(result.pattern_df)),
        "matched_pattern_transactions": int(in_campaign.sum()),
        "laundering_rows": int(laundering.sum()),
        "unassigned_laundering_rows": int((laundering & ~in_campaign).sum()),
        "eval_ok_total": int(camp["eval_ok"].sum()),
        "reassignable_total": int(reassignable.sum()),
        "reassignable_test": int((reassignable & (camp["split"] == "test")).sum()),
        "reassignable_test_shares_train_account": int(
            (reassignable & (camp["split"] == "test") & camp["shares_train_account"]).sum()
        ),
    }

    for split in constants.SPLIT_ORDER:
        actuals[f"eval_ok_{split}"] = int(((camp["split"] == split) & camp["eval_ok"]).sum())

    counts = camp["base_type"].value_counts()
    for key in EXPECTED:
        if key.startswith("base_type_"):
            base_type = key.removeprefix("base_type_")
            actuals[key] = int(counts.get(base_type, 0))

    return actuals


@dataclass(frozen=True)
class Comparison:
    metric: Metric
    actual: int | None

    @property
    def passed(self) -> bool:
        return self.actual == self.metric.expected


def compare(actuals: dict[str, int]) -> list[Comparison]:
    return [Comparison(metric, actuals.get(metric.key)) for metric in METRICS]


def is_full_dataset(actuals: dict[str, int]) -> bool:
    """Whether these numbers came from the real HI-Small file.

    The regression table only means anything against the full dataset; on the
    1% sample every row would read FAIL for no useful reason.
    """
    return actuals.get("total_rows") == EXPECTED["total_rows"]


def format_table(comparisons: list[Comparison]) -> str:
    """A fixed-width PASS/FAIL table, for pasting back into a report."""
    label_width = max(len(c.metric.label) for c in comparisons)
    header = f"{'metric'.ljust(label_width)}  {'expected':>12}  {'actual':>12}  result"
    lines = [header, "-" * len(header)]

    for comparison in comparisons:
        actual = "missing" if comparison.actual is None else f"{comparison.actual:,}"
        lines.append(
            f"{comparison.metric.label.ljust(label_width)}  "
            f"{comparison.metric.expected:>12,}  "
            f"{actual:>12}  "
            f"{'PASS' if comparison.passed else 'FAIL'}"
        )

    failed = [c for c in comparisons if not c.passed]
    lines.append("-" * len(header))
    lines.append(
        f"{len(comparisons) - len(failed)} of {len(comparisons)} passed"
        + ("" if not failed else f"  ({len(failed)} FAILED)")
    )
    return "\n".join(lines)


def format_markdown(comparisons: list[Comparison]) -> str:
    rows = [
        "| Metric | Expected | Actual | Result | Evidence |",
        "|---|---:|---:|---|---|",
    ]
    for c in comparisons:
        actual = "missing" if c.actual is None else f"{c.actual:,}"
        rows.append(
            f"| {c.metric.label} | {c.metric.expected:,} | {actual} | "
            f"{'PASS' if c.passed else '**FAIL**'} | {c.metric.evidence} |"
        )
    return "\n".join(rows)


def summarize_run(result: pipeline.PipelineResult) -> dict[str, object]:
    """Context worth recording alongside the numbers themselves."""
    camp = result.camp
    return {
        "pandas_version": pd.__version__,
        "trans_csv": str(result.trans_csv),
        "patterns_txt": str(result.patterns_txt),
        "k_institutions": constants.K_INSTITUTIONS,
        "institution_loads": list(result.loads),
        "group_counts": {str(k): int(v) for k, v in camp["group"].value_counts().items()},
        "unfragmentable": int((camp["group"] == "unfragmentable").sum()),
        "date_range": [
            result.full_df["ts"].min().isoformat(),
            result.full_df["ts"].max().isoformat(),
        ],
    }
