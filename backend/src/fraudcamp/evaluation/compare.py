"""Direct comparison of two detectors from their saved per-campaign records.

**Rule 3.** Two detectors are compared *directly*, over the campaigns both
detected. An A-versus-B difference is never inferred from `A vs C` and
`B vs C`, because each of those is a median over its own shared set and the
two sets are different campaigns. This module exists so that the
XGBoost-versus-GNN comparison - and later every federated-versus-centralized
comparison - is made the one valid way, from the records themselves, rather
than by eye from two summary tables.

It deliberately holds no model code and reads no transactions. Two
`outcome_records` tables are enough, which means a comparison can be run
after the fact, in a CPU session, across runs that happened days apart on
different machines.

What is matched, and why. Rows are paired on **evaluation split, lookback,
seed and operating point**, and only keys present in both tables are
compared. A GNN at seed 3 is compared with an XGBoost at seed 3; the
F1-chosen operating point is compared with the F1-chosen operating point and
a 50-per-day budget with a 50-per-day budget. Comparing across seeds would
mix model variance into the difference, and comparing the F1 points of two
models that happen to alert at ten times the rate compares a cheap detector
with an expensive one, which is why the budget rows exist.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .evaluate import (
    describe_paired_comparison,
    outcomes_from_records,
    paired_lead_time_comparison,
)

#: What has to match before two rows are a pair. Every one of these changes
#: what the detector is or what it is being asked to do.
MATCH_KEYS: tuple[str, ...] = (
    "eval_split", "lookback_h", "seed", "operating_point", "ablation",
)


def _key(record: dict, keys: Sequence[str]) -> tuple:
    out = []
    for name in keys:
        value = record.get(name)
        if name in {"lookback_h", "seed"} and value is not None and value == value:
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                pass
        out.append(value)
    return tuple(out)


def group_records(
    records: Iterable[dict], keys: Sequence[str] = MATCH_KEYS
) -> dict[tuple, list[dict]]:
    """Saved outcome rows grouped by the key tuple they are compared on."""
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(_key(record, keys), []).append(record)
    return grouped


def compare_outcome_tables(
    a_records: Iterable[dict],
    b_records: Iterable[dict],
    label_a: str,
    label_b: str,
    keys: Sequence[str] = MATCH_KEYS,
    bootstrap_seed: int = 0,
    n_resamples: int = 1000,
) -> tuple[list[dict], list[str], list[tuple]]:
    """Paired comparisons of `b` against `a`, one per matched key.

    Returns the rows, a readable note per row, and the keys that appear in
    only one of the two tables. Those unmatched keys are returned rather than
    dropped silently: a missing seed or a missing budget means the comparison
    is narrower than it looks, and a report that does not say so invites the
    reader to assume the full grid.

    The sign convention is the one `paired_lead_time_comparison` fixes:
    `median_difference_h` is `b - a`, so **positive means `b` detected
    earlier**.
    """
    a_groups = group_records(a_records, keys)
    b_groups = group_records(b_records, keys)
    shared = sorted(
        set(a_groups) & set(b_groups), key=lambda k: tuple(str(v) for v in k)
    )
    unmatched = sorted(
        (set(a_groups) ^ set(b_groups)), key=lambda k: tuple(str(v) for v in k)
    )

    rows: list[dict] = []
    notes: list[str] = []
    for key in shared:
        # A group is one detector at one operating point, so a campaign may
        # appear once in it. A repeat means the keys do not separate what
        # they must - two variants collapsed into one group, say - and the
        # paired sets would silently be built from a mixture.
        for side, groups in (("a", a_groups), ("b", b_groups)):
            ids = [int(r["campaign_id"]) for r in groups[key]]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"table {side} repeats campaign ids under "
                    f"{dict(zip(keys, key))}; the match keys do not identify "
                    "one detector at one operating point"
                )
        outcomes_a = outcomes_from_records(a_groups[key])
        outcomes_b = outcomes_from_records(b_groups[key])
        labelled = dict(zip(keys, key))
        paired = paired_lead_time_comparison(
            outcomes_a,
            outcomes_b,
            label_a,
            label_b,
            split=str(labelled.get("eval_split", "")),
            bootstrap_seed=bootstrap_seed,
            n_resamples=n_resamples,
        )
        rows.append(
            {
                "comparison": f"{label_b}_vs_{label_a}",
                **labelled,
                "tau_a": a_groups[key][0].get("tau"),
                "tau_b": b_groups[key][0].get("tau"),
                **paired.as_dict(),
            }
        )
        detail = ", ".join(f"{name}={labelled[name]}" for name in keys)
        notes.append(f"[{detail}] " + describe_paired_comparison(paired))
    return rows, notes, unmatched


def coverage_rows(
    a_records: Iterable[dict],
    b_records: Iterable[dict],
    label_a: str,
    label_b: str,
    keys: Sequence[str] = MATCH_KEYS,
) -> list[dict]:
    """Coverage each way, reported separately from earliness and by campaign id.

    Rule 2: coverage and earliness are never folded into one number. A
    detector that finds more campaigns usually shows a *lower* median lead
    time, because the extra ones are the slow, hard campaigns - so the two
    have to be read side by side, and the discordant campaigns are named so
    they can be looked at rather than assumed random.
    """
    a_groups = group_records(a_records, keys)
    b_groups = group_records(b_records, keys)
    rows = []
    for key in sorted(
        set(a_groups) & set(b_groups), key=lambda k: tuple(str(v) for v in k)
    ):
        detected_a = {
            int(r["campaign_id"]) for r in a_groups[key] if _detected(r)
        }
        detected_b = {
            int(r["campaign_id"]) for r in b_groups[key] if _detected(r)
        }
        ids_a = {int(r["campaign_id"]) for r in a_groups[key]}
        ids_b = {int(r["campaign_id"]) for r in b_groups[key]}
        shared_ids = ids_a & ids_b
        rows.append(
            {
                **dict(zip(keys, key)),
                "label_a": label_a,
                "label_b": label_b,
                "n_campaigns": len(shared_ids),
                "n_detected_a": len(detected_a & shared_ids),
                "n_detected_b": len(detected_b & shared_ids),
                "recall_a": (
                    len(detected_a & shared_ids) / len(shared_ids)
                    if shared_ids else None
                ),
                "recall_b": (
                    len(detected_b & shared_ids) / len(shared_ids)
                    if shared_ids else None
                ),
                "n_both": len(detected_a & detected_b & shared_ids),
                "only_a": ",".join(
                    str(c) for c in sorted((detected_a - detected_b) & shared_ids)
                ),
                "only_b": ",".join(
                    str(c) for c in sorted((detected_b - detected_a) & shared_ids)
                ),
            }
        )
    return rows


def _detected(record: dict) -> bool:
    from .evaluate import _as_bool

    return _as_bool(record["detected"])
