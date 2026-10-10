#!/usr/bin/env python
"""Run this repository's heavy steps inside a Kaggle notebook.

Subcommands:
    verify     run the full-data regression tests and print a PASS/FAIL table
               of every number in README "Verified numbers"
    pipeline   run the full pipeline, export the artifacts, and zip them to
               <out>/artifacts.zip for download
    features-smoke
               build detection windows at L = 24/48/72 on the full data and
               report edges, accounts, laundering edges, build time, memory
               and tensor sizes, plus a full-batch GNN memory estimate
    evaluate-scorers
               run the oracle and random sanity scorers on the full data and
               report recall, precision, false alarms, lead times and CIs,
               with tau chosen on validation
    make-sample
               build the 1% development sample for download
    train      (Stage 3.3+) not implemented yet
    evaluate   (Stage 3.3+) not implemented yet

Everything it writes goes to --out (default /kaggle/working), which is the
only directory Kaggle persists. Each step prints its elapsed time and peak
memory.

See run_on_kaggle.md in this directory for the notebook cells to paste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent.parent
SRC = BACKEND / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_OUT = Path("/kaggle/working")


# --------------------------------------------------------------------------
# step timing / memory
# --------------------------------------------------------------------------


def memory_mb(include_children: bool = False) -> tuple[float | None, bool]:
    """Resident memory in MB, and whether it is a true high-water mark.

    Kaggle is Linux, where `resource` gives the real peak. Windows has no
    `resource`, so this falls back to the *current* RSS via psutil, which is
    reported as such rather than being passed off as a peak.
    """
    try:
        import resource

        usage = resource.getrusage(
            resource.RUSAGE_CHILDREN if include_children else resource.RUSAGE_SELF
        )
        peak = usage.ru_maxrss
        # Linux reports kilobytes; macOS reports bytes.
        return (peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)), True
    except ImportError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024), False
    except ImportError:
        return None, False


STEPS: list[dict[str, object]] = []


@contextmanager
def step(name: str, children: bool = False):
    print(f"\n=== {name} ===", flush=True)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        memory, is_peak = memory_mb(include_children=children)
        if memory is None:
            memory_text = "memory unavailable"
        elif is_peak:
            memory_text = f"peak memory {memory:,.0f} MB"
        else:
            memory_text = f"memory {memory:,.0f} MB (current, not peak - no `resource` module)"
        print(f"--- {name}: {elapsed:,.1f}s, {memory_text}", flush=True)
        STEPS.append(
            {
                "step": name,
                "seconds": round(elapsed, 1),
                "memory_mb": memory,
                "is_peak": is_peak,
            }
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visibility_hashes(artifacts_dir: Path) -> dict[str, str]:
    """SHA-256 of each visibility/seed_*.parquet.

    The point is comparing two independent Kaggle sessions: the visibility
    search is seeded, so identical seeds must produce identical bytes. If
    these digests differ between sessions, the build is not reproducible.
    """
    visibility = artifacts_dir / "visibility"
    if not visibility.is_dir():
        return {}
    return {path.name: sha256(path) for path in sorted(visibility.glob("seed_*.parquet"))}


def steps_markdown() -> str:
    lines = ["| Step | Seconds | Memory | True peak |", "|---|---:|---:|---|"]
    for entry in STEPS:
        memory = entry["memory_mb"]
        text = f"{memory:,.0f} MB" if isinstance(memory, (int, float)) else "unavailable"
        lines.append(
            f"| {entry['step']} | {entry['seconds']:,.1f} | {text} | "
            f"{'yes' if entry['is_peak'] else 'no (current RSS)'} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def command_verify(args: argparse.Namespace) -> int:
    from fraudcamp import regression

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    actuals_path = out / "regression_actuals.json"
    if actuals_path.exists():
        actuals_path.unlink()

    env = {**_base_env(), "FRAUDCAMP_REGRESSION_OUT": str(actuals_path)}

    with step("pytest -m fulldata", children=True):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "fulldata", "-v", "tests/"],
            cwd=BACKEND,
            env=env,
        )

    if not actuals_path.exists():
        print(
            "\nNo regression numbers were produced. The full-data tests were "
            "skipped, which almost always means the dataset is not where "
            f"{BACKEND / 'configs' / 'kaggle.yaml'} expects it.\n"
            "Check the path with:  !ls /kaggle/input/datasets/ealtman2019/"
            "ibm-transactions-for-anti-money-laundering-aml/",
            flush=True,
        )
        return 2

    payload = json.loads(actuals_path.read_text(encoding="utf-8"))
    comparisons = regression.compare(payload["actuals"])
    table = regression.format_table(comparisons)

    print("\n" + "=" * 72)
    print("PHASE 2 REGRESSION CHECK")
    print("=" * 72)
    print(table, flush=True)

    failed = [c for c in comparisons if c.is_failure]
    run_info = payload.get("run", {})

    report = out / "verify_report.md"
    report.write_text(
        "\n".join(
            [
                "# Phase 2 verification on the full dataset",
                "",
                f"- pytest exit code: `{completed.returncode}`",
                f"- pandas: `{run_info.get('pandas_version', 'unknown')}`",
                f"- transactions file: `{run_info.get('trans_csv', 'unknown')}`",
                f"- date range: `{run_info.get('date_range', 'unknown')}`",
                f"- institution loads (K={run_info.get('k_institutions')}): "
                f"`{run_info.get('institution_loads')}`",
                f"- group counts after the unfragmentable relabel: "
                f"`{run_info.get('group_counts')}`",
                f"- unfragmentable campaigns: **{run_info.get('unfragmentable')}**",
                "",
                "## Regression numbers",
                "",
                regression.format_markdown(comparisons),
                "",
                "## Timing",
                "",
                steps_markdown(),
                "",
                (
                    "**All regression numbers match.**"
                    if not failed
                    else f"**{len(failed)} regression number(s) FAILED — see the table.**"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {report}")
    print(f"Wrote {actuals_path}")

    if failed or completed.returncode != 0:
        print("\nVERIFY FAILED", flush=True)
        return 1
    print("\nVERIFY PASSED — every regression number matches.", flush=True)
    return 0


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC}{';' if sys.platform == 'win32' else ':'}{existing}".rstrip(
        ":;"
    )
    return env


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def command_pipeline(args: argparse.Namespace) -> int:
    from fraudcamp import export, load_data, pipeline, regression, splits

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    with step("build pipeline"):
        result = pipeline.build(config_path)

    camp = result.camp
    reassignable = splits.reassignable_mask(camp)
    print(f"  rows: {len(result.full_df):,}")
    print(f"  campaigns: {len(camp):,} ({int(camp['eval_ok'].sum())} eval_ok)")
    print(f"  reassignable: {int(reassignable.sum())}")
    print(f"  institution loads: {result.loads}")

    config = load_data.load_config(config_path)
    artifacts_dir = load_data.resolve_path(config_path, config["artifacts_dir"])

    with step("export artifacts"):
        written = export.export_all(result, artifacts_dir)

    total_kb = 0.0
    for name, path in written.items():
        size_kb = path.stat().st_size / 1024
        total_kb += size_kb
        print(f"  {name}: {size_kb:,.1f} KB")
    print(f"  total: {total_kb / 1024:,.2f} MB")

    with step("zip artifacts"):
        archive = shutil.make_archive(
            str(out / "artifacts"), "zip", root_dir=str(artifacts_dir)
        )
    print(f"  {archive}: {Path(archive).stat().st_size / (1024 * 1024):,.2f} MB")

    hashes = visibility_hashes(artifacts_dir)
    print("\n=== SHA-256 of the visibility files ===")
    print("Compare these across two independent sessions: the search is")
    print("seeded, so identical seeds must give identical bytes.")
    for name, digest in hashes.items():
        print(f"  {digest}  {name}")

    # The same numbers verify prints, so a pipeline run is self-checking --
    # but only against the real dataset. On the 1% sample every row would
    # read FAIL for no useful reason, so say that instead.
    actuals = regression.compute_actuals(result)
    full_dataset = regression.is_full_dataset(actuals)
    comparisons = regression.compare(actuals)
    failed = [c for c in comparisons if c.is_failure] if full_dataset else []

    if full_dataset:
        print("\n" + regression.format_table(comparisons), flush=True)
    else:
        print(
            f"\nSkipping the regression table: this run has "
            f"{actuals['total_rows']:,} rows, not the full dataset's "
            f"{regression.EXPECTED['total_rows']:,}. The table only means "
            "something against the real HI-Small file.",
            flush=True,
        )

    report = out / "pipeline_report.md"
    report.write_text(
        "\n".join(
            [
                "# Pipeline run",
                "",
                f"- artifacts: `{artifacts_dir}`",
                f"- archive: `{archive}`",
                f"- rows: {actuals['total_rows']:,}",
                "",
                "## Regression numbers",
                "",
                (
                    regression.format_markdown(comparisons)
                    if full_dataset
                    else f"Not checked: this run had {actuals['total_rows']:,} rows, not "
                    f"the full dataset's {regression.EXPECTED['total_rows']:,}."
                ),
                "",
                "## Visibility file digests (SHA-256)",
                "",
                "Compare these across two independent sessions; the seeded",
                "search must reproduce identical bytes.",
                "",
                "| File | SHA-256 |",
                "|---|---|",
                *(f"| `{name}` | `{digest}` |" for name, digest in hashes.items()),
                "",
                "## Timing",
                "",
                steps_markdown(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path = out / "visibility_hashes.json"
    json_path.write_text(
        json.dumps(
            {
                "unfragmentable_campaigns": actuals.get("unfragmentable_campaigns"),
                "visibility_sha256": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {report}")
    print(f"Wrote {json_path}")
    print(f"Download: {archive}")

    if failed:
        print(f"\nWARNING: {len(failed)} regression number(s) do not match.", flush=True)
        return 1
    return 0


# --------------------------------------------------------------------------
# features-smoke
# --------------------------------------------------------------------------

# September 2022: the 1st is a Thursday, so the 3rd/4th and 10th/11th are
# weekends. One weekday and one weekend detection time per split, except
# validation, which spans only Mon Sept 5 -> Tue Sept 6 and so has no
# weekend time to sample. That gap is reported rather than papered over.
SMOKE_PROBES: list[tuple[str, str, str]] = [
    ("train", "weekday (Fri)", "2022-09-02 12:00"),
    ("train", "weekend (Sun)", "2022-09-04 12:00"),
    ("val", "weekday (Mon)", "2022-09-05 12:00"),
    ("test", "weekday (Wed)", "2022-09-07 12:00"),
    ("test", "weekend (Sat)", "2022-09-10 12:00"),
]


def estimate_gine_memory_mb(
    n_nodes: int, n_edges: int, d_node: int, d_edge: int, hidden: int = 64, layers: int = 3
) -> dict[str, float]:
    """Rough full-batch GPU memory for a 3-layer GINEConv edge classifier.

    An estimate from tensor shapes, not a measurement. Counts the forward
    activations that autograd must retain, which dominate: per layer a node
    hidden state, the GINE MLP's inner layer (2x hidden), and two
    per-edge tensors (the projected edge features and the messages being
    aggregated). Then the edge readout MLP over [h_src, h_dst, edge_feats].
    PyTorch's allocator and the backward pass add overhead on top, so a
    multiplier is applied and reported separately.
    """
    f = 4  # float32

    inputs = (n_nodes * d_node + n_edges * d_edge) * f
    per_layer = (
        n_nodes * hidden  # node hidden state
        + n_nodes * 2 * hidden  # GINE MLP inner layer
        + n_edges * hidden  # edge features projected to hidden
        + n_edges * hidden  # messages awaiting aggregation
    ) * f
    activations = per_layer * layers
    readout = (n_edges * (2 * hidden + d_edge) + n_edges * hidden) * f

    forward_mb = (inputs + activations + readout) / 1e6
    # gradients roughly mirror the retained activations; the allocator
    # fragments and cuBLAS needs workspace, hence the headroom factor.
    with_backward_mb = forward_mb * 2.0
    return {
        "forward_activations_mb": round(forward_mb, 1),
        "with_backward_mb": round(with_backward_mb, 1),
        "recommended_headroom_mb": round(with_backward_mb * 1.3, 1),
    }


def _markdown_table(table) -> str:
    """Markdown table without pandas.to_markdown, which needs tabulate."""
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def scan_window_sizes(
    full_df,
    lookbacks: tuple[int, ...],
) -> list[dict]:
    """Edge and account counts for EVERY usable detection time, per (L, split).

    Deliberately cheap: no features are built. Timestamps are sorted once and
    each window is located with two binary searches, laundering counts come
    from a prefix sum, and only the distinct-account count needs real work.

    This exists because sampling a handful of detection times is not enough
    to find the largest window. The first version of this command sampled
    Sept 2 12:00, whose L=48 lookback reaches before the data starts, so it
    reported a clamped 1.5M-edge window as the "largest" and missed the
    genuinely biggest valid ones.
    """
    import numpy as np
    import pandas as pd

    from fraudcamp import constants, windows

    ordered = full_df.sort_values("ts", kind="stable")
    ts = ordered["ts"].to_numpy()
    src = ordered["src"].to_numpy()
    dst = ordered["dst"].to_numpy()
    laundering_prefix = np.concatenate(
        [[0], np.cumsum(ordered["Is Laundering"].to_numpy(dtype=np.int64))]
    )

    rows: list[dict] = []
    for lookback in lookbacks:
        # Training times are rule-3 filtered; evaluation horizons all start
        # days after DATA_START and so are always valid.
        per_split: dict[str, list] = {"train": windows.training_detection_times(lookback)}
        for split in constants.EVALUATION_HORIZONS:
            per_split[split] = windows.evaluation_detection_times(split)

        for split, times in per_split.items():
            for t in times:
                lower = t - pd.Timedelta(hours=lookback)
                lo = int(np.searchsorted(ts, np.datetime64(lower), side="left"))
                hi = int(np.searchsorted(ts, np.datetime64(t), side="left"))
                n_edges = hi - lo
                n_accounts = (
                    len(pd.unique(np.concatenate([src[lo:hi], dst[lo:hi]])))
                    if n_edges
                    else 0
                )
                rows.append(
                    {
                        "lookback_h": lookback,
                        "excluded": lookback in constants.EXCLUDED_LOOKBACKS_H,
                        "split": split,
                        "t": str(t),
                        "n_edges": n_edges,
                        "n_accounts": n_accounts,
                        "n_laundering": int(
                            laundering_prefix[hi] - laundering_prefix[lo]
                        ),
                    }
                )
    return rows


def summarise_scan(scan_rows: list[dict]):
    """min / median / max per (lookback, split), with the time of the max."""
    import pandas as pd

    frame = pd.DataFrame(scan_rows)
    summary = []
    for (lookback, split), group in frame.groupby(["lookback_h", "split"], sort=True):
        peak = group.loc[group["n_edges"].idxmax()]
        summary.append(
            {
                "lookback_h": int(lookback),
                "excluded": bool(group["excluded"].iloc[0]),
                "split": split,
                "n_times": int(len(group)),
                "edges_min": int(group["n_edges"].min()),
                "edges_median": int(group["n_edges"].median()),
                "edges_max": int(group["n_edges"].max()),
                "accounts_min": int(group["n_accounts"].min()),
                "accounts_median": int(group["n_accounts"].median()),
                "accounts_max": int(group["n_accounts"].max()),
                "laundering_max": int(group["n_laundering"].max()),
                "t_of_max_edges": str(peak["t"]),
                "accounts_at_max": int(peak["n_accounts"]),
            }
        )
    return pd.DataFrame(summary), frame


def command_features_smoke(args: argparse.Namespace) -> int:
    import gc

    import pandas as pd

    from fraudcamp import constants, features, pipeline, windows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with step("build pipeline"):
        result = pipeline.build(Path(args.config))
    full_df = result.full_df
    print(f"  rows: {len(full_df):,}")

    # Rule 3: a training detection time is usable only when its whole
    # lookback lies inside the data.
    usable = windows.usable_training_times_per_lookback()
    grid_total = len(windows.detection_times_for_split("train"))
    print("\n=== usable training detection times (rule 3: full lookback) ===")
    print(f"  unfiltered training grid: {grid_total}")
    for lookback, count in usable.items():
        times = windows.training_detection_times(lookback)
        span = f"{times[0]} .. {times[-1]}" if times else "none"
        tag = "  [EXCLUDED]" if lookback in constants.EXCLUDED_LOOKBACKS_H else ""
        print(
            f"  L={lookback:>2}h: {count:>2} usable, {grid_total - count:>2} dropped"
            f"   {span}{tag}"
        )

    # The spec is fitted on ONE training window, which must itself be
    # rule-3 valid. Fitting across all training windows would concatenate
    # overlapping windows into tens of millions of rows for no benefit here.
    fit_t = windows.training_detection_times(constants.DEFAULT_LOOKBACK_H)[0]
    with step(f"fit feature spec on the {fit_t} {constants.DEFAULT_LOOKBACK_H}h training window"):
        fit_window = windows.build_window(full_df, fit_t, lookback_h=constants.DEFAULT_LOOKBACK_H)
        spec = features.fit_feature_spec([fit_window], lookback_h=constants.DEFAULT_LOOKBACK_H)
    print(f"  edge dims: {len(spec.edge_feature_names)}")
    print(f"  node dims: {len(spec.node_feature_names)}")
    print(
        f"  payment currencies: {len(spec.payment_currencies)}, "
        f"receiving: {len(spec.receiving_currencies)}, formats: {len(spec.payment_formats)}"
    )
    del fit_window
    gc.collect()

    d_edge = len(spec.edge_feature_names)
    d_node = len(spec.node_feature_names)

    # --- cheap full scan of every usable detection time -------------------
    with step("scan every usable detection time (counts only, no features)"):
        scan_rows = scan_window_sizes(full_df, constants.MEASURED_LOOKBACKS_H)
        scan_summary, scan_frame = summarise_scan(scan_rows)

    print("\n=== window sizes across ALL usable detection times ===")
    print(
        f"  {'L':>3} {'split':<7} {'times':>5} {'edges min':>10} {'median':>10} "
        f"{'max':>10} {'accounts max':>13}  time of max"
    )
    for row in scan_summary.itertuples():
        tag = " [EXCL]" if row.excluded else ""
        print(
            f"  {row.lookback_h:>3} {row.split:<7} {row.n_times:>5} "
            f"{row.edges_min:>10,} {row.edges_median:>10,} {row.edges_max:>10,} "
            f"{row.accounts_max:>13,}  {row.t_of_max_edges}{tag}"
        )

    # --- detailed featurisation at the sampled probes ---------------------
    rows: list[dict] = []
    print("\n=== featurised sample windows ===")
    for split, day_kind, t_text in SMOKE_PROBES:
        t = pd.Timestamp(t_text)
        actual_split = windows.split_of_detection_time(t)
        if actual_split != split:
            print(
                f"  WARNING: {t} is in split {actual_split!r}, expected {split!r}; skipping.",
                flush=True,
            )
            continue

        for lookback in constants.MEASURED_LOOKBACKS_H:
            # Rule 3 applies to training windows only. Featurising an
            # invalid one would silently measure a clamped, short window -
            # which is exactly the bug this guard was added for.
            if split == "train" and not windows.has_full_lookback(t, lookback):
                lower = t - pd.Timedelta(hours=lookback)
                print(
                    f"  {split:<5} L={lookback:<2} {t_text}  SKIPPED: rule 3 - the "
                    f"lookback would start {lower}, before the data begins "
                    f"({constants.DATA_START}). First valid training time at this "
                    f"L is {windows.training_detection_times(lookback)[0]}.",
                    flush=True,
                )
                continue

            started = time.perf_counter()
            window = windows.build_window(full_df, t, lookback_h=lookback)
            slice_s = time.perf_counter() - started

            if window.n_transactions == 0:
                print(f"  {split:<5} L={lookback:<2} {t_text}  empty window", flush=True)
                continue

            started = time.perf_counter()
            wf = features.build_window_features(window, spec)
            build_s = time.perf_counter() - started

            tensors = (
                wf.node_features.nbytes
                + wf.edge_features.nbytes
                + wf.edge_index.nbytes
                + wf.labels.nbytes
            )
            memory, _ = memory_mb()
            excluded = lookback in constants.EXCLUDED_LOOKBACKS_H
            rows.append(
                {
                    "split": split,
                    "day": day_kind,
                    "t": t_text,
                    "lookback_h": lookback,
                    "excluded": excluded,
                    "n_edges": wf.n_edges,
                    "n_accounts": wf.n_nodes,
                    "n_laundering": int(wf.labels.sum()),
                    "laundering_rate": round(wf.positive_rate, 6),
                    "class_weight": round(wf.class_weight(), 1),
                    "slice_s": round(slice_s, 2),
                    "build_s": round(build_s, 2),
                    "tensors_mb": round(tensors / 1e6, 1),
                    "xgb_matrix_mb": round(wf.n_edges * (d_edge + 2 * d_node) * 4 / 1e6, 1),
                    "peak_rss_mb": round(memory, 0) if memory else None,
                }
            )
            print(
                f"  {split:<5} L={lookback:<2} {t_text}  "
                f"edges={wf.n_edges:>9,}  accounts={wf.n_nodes:>8,}  "
                f"laundering={int(wf.labels.sum()):>5,}  "
                f"build={build_s:>5.1f}s  tensors={tensors / 1e6:>7.1f} MB"
                f"{'  [EXCLUDED]' if excluded else ''}",
                flush=True,
            )
            del wf, window
            gc.collect()

    # --- GPU estimates from the TRUE maxima found by the scan -------------
    print("\n=== estimated full-batch GPU memory from the TRUE largest valid window ===")
    print("    (3-layer GINEConv, hidden 64; shapes, not measurements)")
    estimates: list[dict] = []
    for lookback in constants.MEASURED_LOOKBACKS_H:
        per_lookback = scan_summary[scan_summary["lookback_h"] == lookback]
        if per_lookback.empty:
            continue
        excluded = bool(per_lookback["excluded"].iloc[0])

        overall = per_lookback.loc[per_lookback["edges_max"].idxmax()]
        training = per_lookback[per_lookback["split"] == "train"]
        training_peak = (
            training.loc[training["edges_max"].idxmax()] if not training.empty else None
        )

        for label, row in (("inference (any split)", overall), ("training", training_peak)):
            if row is None:
                continue
            estimate = estimate_gine_memory_mb(
                n_nodes=int(row["accounts_at_max"]),
                n_edges=int(row["edges_max"]),
                d_node=d_node,
                d_edge=d_edge,
            )
            estimates.append(
                {
                    "lookback_h": int(lookback),
                    "excluded": excluded,
                    "scope": label,
                    "split_of_max": row["split"],
                    "t_of_max": row["t_of_max_edges"],
                    "n_edges": int(row["edges_max"]),
                    "n_accounts": int(row["accounts_at_max"]),
                    **estimate,
                }
            )
            print(
                f"  L={lookback:>2}h {label:<22} {int(row['edges_max']):>9,} edges "
                f"@ {row['t_of_max_edges']} ({row['split']}) -> "
                f"{estimate['recommended_headroom_mb']:>9,.0f} MB with headroom"
                f"{'  [EXCLUDED]' if excluded else ''}"
            )

    estimates_frame = pd.DataFrame(estimates)
    candidate_estimates = estimates_frame[~estimates_frame["excluded"]]
    worst = (
        candidate_estimates.loc[candidate_estimates["recommended_headroom_mb"].idxmax()]
        if not candidate_estimates.empty
        else None
    )
    if worst is not None:
        print(
            f"\n  worst candidate case: L={int(worst['lookback_h'])}h "
            f"{worst['scope']}, {int(worst['n_edges']):,} edges -> "
            f"{worst['recommended_headroom_mb']:,.0f} MB. Kaggle GPUs have "
            "16 GB (16,384 MB)."
        )
        if worst["recommended_headroom_mb"] > 16384:
            print(
                "  => full-batch does not fit; neighbour sampling is required, "
                "which the Phase 3 design permits."
            )

    # --- reports -----------------------------------------------------------
    sample_table = pd.DataFrame(rows)
    scan_summary.to_csv(out / "features_smoke_scan.csv", index=False)
    scan_frame.to_csv(out / "features_smoke_scan_full.csv", index=False)
    if not sample_table.empty:
        sample_table.to_csv(out / "features_smoke.csv", index=False)

    (out / "features_smoke.json").write_text(
        json.dumps(
            {
                "edge_dims": d_edge,
                "node_dims": d_node,
                "vocabularies": {
                    "payment_currencies": len(spec.payment_currencies),
                    "receiving_currencies": len(spec.receiving_currencies),
                    "payment_formats": len(spec.payment_formats),
                },
                "spec_fitted_on": spec.fitted_on,
                "lookback_candidates": list(constants.LOOKBACKS_H),
                "lookbacks_excluded": list(constants.EXCLUDED_LOOKBACKS_H),
                "usable_training_times_per_lookback": {str(k): v for k, v in usable.items()},
                "training_grid_total": grid_total,
                "scan_summary": scan_summary.to_dict(orient="records"),
                "sample_windows": rows,
                "gpu_estimates": estimates,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    (out / "features_smoke.md").write_text(
        "\n".join(
            [
                "# Feature smoke test on the full dataset",
                "",
                f"- edge features: {d_edge} dims "
                f"({len(spec.payment_currencies)} payment currencies, "
                f"{len(spec.receiving_currencies)} receiving, "
                f"{len(spec.payment_formats)} formats)",
                f"- node features: {d_node} dims",
                f"- lookback candidates: {list(constants.LOOKBACKS_H)}; "
                f"excluded but measured: {list(constants.EXCLUDED_LOOKBACKS_H)}",
                f"- spec fitted on: `{spec.fitted_on}`",
                "",
                "## Usable training detection times (rule 3: full lookback)",
                "",
                f"Unfiltered training grid: {grid_total}.",
                "",
                "| Lookback | Usable | Dropped |",
                "|---|---:|---:|",
                *(
                    f"| {lookback}h | {count} | {grid_total - count} |"
                    for lookback, count in usable.items()
                ),
                "",
                "## Window sizes across ALL usable detection times",
                "",
                "Counts only, no features. This is the authoritative source for",
                "the largest window: sampling a few detection times is not",
                "enough to find it.",
                "",
                _markdown_table(scan_summary),
                "",
                "## Featurised sample windows",
                "",
                "Training probes that would violate rule 3 are skipped, not",
                "silently clamped to a shorter window.",
                "",
                _markdown_table(sample_table) if not sample_table.empty else "_none_",
                "",
                "## Estimated full-batch GPU memory",
                "",
                "3-layer GINEConv, hidden 64, from the true largest valid window",
                "per lookback. Estimated from tensor shapes, not measured.",
                "",
                _markdown_table(estimates_frame) if not estimates_frame.empty else "_none_",
                "",
                "## Timing",
                "",
                steps_markdown(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'features_smoke.md'} (+ .json, .csv, scan csvs)")
    return 0


def command_evaluate_scorers(args: argparse.Namespace) -> int:
    """Validate the evaluation module on the FULL data with the oracle and
    random scorers, before any model exists to blame.

    The oracle's numbers are also the *ceiling*: the best lead time
    achievable under the frozen matching rule, since no detector can see a
    campaign earlier than the moment the rule's three-account requirement is
    satisfiable.
    """
    import numpy as np
    import pandas as pd

    from fraudcamp import constants, evaluation, pipeline
    from fraudcamp.evaluation import scorers

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    lookbacks = [int(v) for v in args.lookbacks.split(",")] if args.lookbacks else list(
        constants.LOOKBACKS_H
    )
    tau_grid = (
        tuple(float(v) for v in args.tau_grid.split(","))
        if args.tau_grid
        else evaluation.DEFAULT_TAU_GRID
    )

    with step("build pipeline"):
        result = pipeline.build(Path(args.config))
    print(f"  rows: {len(result.full_df):,}, campaigns: {len(result.camp):,}")

    def oracle(window, t):
        return scorers.oracle_scores(window["Is Laundering"].to_numpy())

    def make_random(seed: int):
        def scorer(window, t):
            # seeded per detection time so each window differs but the whole
            # run reproduces exactly
            return scorers.random_scores(len(window), seed + int(t.value % 100_000))

        return scorer

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    breakdown_rows: list[dict] = []
    tau_choices: list[dict] = []

    for lookback in lookbacks:
        print(f"\n{'=' * 78}\nL = {lookback}h\n{'=' * 78}")

        # --- tau, chosen on validation only -----------------------------
        # The oracle's scores are exactly 0 or 1, so every tau in (0, 1]
        # keeps precisely the laundering edges and the choice is vacuous.
        # It is still run through the same selection so the two scorers are
        # treated identically and the curve is on record.
        chosen: dict[str, float] = {}
        for name, scorer in (("oracle", oracle), ("random", make_random(args.seed))):
            with step(f"select tau on validation: {name}, L={lookback}h"):
                tau, curve = evaluation.select_tau(
                    result.full_df, result.camp, result.pattern_df,
                    result.accounts_by_campaign, scorer,
                    tau_grid=tau_grid, lookback_h=lookback, seed=args.seed,
                    scorer_name=name,
                )
            chosen[name] = tau
            for row in curve:
                curve_rows.append({"scorer": name, **row})
            tau_choices.append(
                {
                    "scorer": name,
                    "lookback_h": lookback,
                    "chosen_tau": tau,
                    "selected_on": "val",
                    "criterion": "max campaign-level F1, ties to the lower tau",
                    "grid": list(tau_grid),
                    "seed": args.seed,
                }
            )
            print(f"  chosen tau = {tau} (campaign-F1 on validation)")
            print(f"  {'tau':>6} {'precision':>10} {'recall':>8} {'F1':>8} {'hits':>7} {'FA':>8}")
            for row in curve:
                print(
                    f"  {row['tau']:>6} {row['campaign_precision']:>10.3f} "
                    f"{row['campaign_recall']:>8.3f} {row['campaign_f1']:>8.3f} "
                    f"{row['hits']:>7,} {row['false_alarms']:>8,}"
                )

        # --- evaluate each horizon at the frozen tau ---------------------
        for name, scorer in (("oracle", oracle), ("random", make_random(args.seed))):
            tau = chosen[name]
            for split in splits:
                with step(f"evaluate {name} on {split}, L={lookback}h, tau={tau}"):
                    evaluated = evaluation.evaluate_split(
                        result.full_df, result.camp, result.pattern_df,
                        result.accounts_by_campaign, split, scorer,
                        tau=tau, lookback_h=lookback, seed=args.seed,
                        scorer_name=name,
                    )
                for subset in constants.TEST_SUBSETS:
                    summary_rows.append(
                        {
                            "lookback_h": lookback,
                            **evaluation.summarize(
                                evaluated, subset=subset,
                                bootstrap_seed=args.seed, n_resamples=args.bootstrap,
                            ),
                        }
                    )
                if name == "oracle":
                    for attribute in ("base_type", "group"):
                        for row in evaluation.summarize_by(evaluated, attribute):
                            breakdown_rows.append(
                                {
                                    "lookback_h": lookback, "split": split,
                                    "attribute": attribute,
                                    "value": row[attribute],
                                    **{k: v for k, v in row.items() if k != attribute},
                                }
                            )

    # --- report ---------------------------------------------------------
    def fmt(value, spec=".2f"):
        return format(value, spec) if value is not None else "-"

    print("\n" + "=" * 126)
    print(
        f"{'L':>3} {'split':<7}{'scorer':<8}{'subset':<20}{'tau':>5}{'camp':>5}{'det':>5}"
        f"{'recall':>8}{'prec':>7}{'FA/day':>8}{'other':>7}{'ambig':>7}"
        f"{'lead_h':>9}{'norm':>7}{'frac':>7}"
    )
    print("-" * 126)
    for row in summary_rows:
        print(
            f"{row['lookback_h']:>3} {row['split']:<7}{row['scorer']:<8}"
            f"{row['test_subset']:<20}{row['tau']:>5}{row['n_campaigns']:>5}"
            f"{row['n_detected']:>5}{fmt(row['campaign_recall']):>8}"
            f"{fmt(row['campaign_precision']):>7}{row['false_alarms_per_day']:>8.1f}"
            f"{row['other_split_hits']:>7}{row['ambiguous']:>7}"
            f"{fmt(row['median_lead_time_h'], '.1f'):>9}"
            f"{fmt(row['median_normalized_lead_time']):>7}"
            f"{fmt(row['median_frac_observed_at_detection']):>7}"
        )
    print("=" * 126)

    print("\nbootstrap 95% CIs (subset = all):")
    for row in summary_rows:
        if row["test_subset"] != "all":
            continue
        rl, rh = row["recall_ci"]
        ll, lh = row["median_lead_time_h_ci"]
        print(
            f"  L={row['lookback_h']:>2} {row['split']:<7}{row['scorer']:<8}"
            f" recall {fmt(row['campaign_recall'], '.3f')} "
            f"[{fmt(rl, '.3f')}, {fmt(rh, '.3f')}]"
            f"   median lead {fmt(row['median_lead_time_h'], '.1f')}h "
            f"[{fmt(ll, '.1f')}, {fmt(lh, '.1f')}]"
        )

    # --- the oracle ceiling ---------------------------------------------
    ceiling = [
        r for r in summary_rows
        if r["scorer"] == "oracle" and r["test_subset"] == "all"
    ]
    print("\n=== ORACLE CEILING (best achievable under the frozen matching rule) ===")
    for row in ceiling:
        print(
            f"  L={row['lookback_h']:>2}h {row['split']:<7} "
            f"recall {fmt(row['campaign_recall'], '.3f')}  "
            f"median lead {fmt(row['median_lead_time_h'], '.1f')}h  "
            f"median normalized {fmt(row['median_normalized_lead_time'], '.3f')}  "
            f"median frac observed {fmt(row['median_frac_observed_at_detection'], '.3f')}"
        )
    print(
        "  No detector can beat these: the matching rule needs 3 of a "
        "campaign's accounts before a cluster can hit it."
    )

    print("\n=== sanity verdict ===")
    ok = True
    for lookback in lookbacks:
        for split in splits:
            o = next(
                (r for r in summary_rows if r["scorer"] == "oracle"
                 and r["split"] == split and r["lookback_h"] == lookback
                 and r["test_subset"] == "all"), None,
            )
            r = next(
                (x for x in summary_rows if x["scorer"] == "random"
                 and x["split"] == split and x["lookback_h"] == lookback
                 and x["test_subset"] == "all"), None,
            )
            if not o or not r:
                continue
            o_recall = o["campaign_recall"] or 0.0
            r_recall = r["campaign_recall"] or 0.0
            verdict = "OK" if o_recall > 0.5 and o_recall > r_recall + 0.2 else "SUSPICIOUS"
            ok &= verdict == "OK"
            print(
                f"  L={lookback:>2} {split:<7} oracle {o_recall:.2f} vs "
                f"random {r_recall:.2f}  -> {verdict}"
            )
    if not ok:
        print(
            "  SUSPICIOUS means the evaluation code needs a look before any "
            "model is trained on top of it."
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out / "evaluate_scorers.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "evaluate_scorers_tau_curve.csv", index=False)
    pd.DataFrame(breakdown_rows).to_csv(out / "evaluate_scorers_breakdown.csv", index=False)

    (out / "evaluate_scorers.json").write_text(
        json.dumps(
            {
                "lookbacks": lookbacks,
                "splits": splits,
                "seed": args.seed,
                "bootstrap_resamples": args.bootstrap,
                "tau_selection": tau_choices,
                "tau_curve": curve_rows,
                "summary": summary_rows,
                "oracle_breakdown": breakdown_rows,
                "oracle_ceiling": ceiling,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    (out / "evaluate_scorers.md").write_text(
        "\n".join(
            [
                "# Sanity scorers on the full dataset",
                "",
                "The oracle scores each edge with its true label; random scores",
                "uniformly. They validate the evaluation code before any model",
                "exists to blame for a bad number.",
                "",
                f"- seed: {args.seed}",
                f"- lookbacks: {lookbacks}",
                f"- bootstrap resamples: {args.bootstrap}",
                "",
                "## How tau was chosen",
                "",
                "On **validation only**, maximising campaign-level F1, ties broken",
                "toward the lower tau. The oracle's scores are 0/1, so any tau in",
                "(0, 1] is equivalent for it; it is selected the same way anyway so",
                "both scorers are treated identically.",
                "",
                _markdown_table(pd.DataFrame(tau_choices)),
                "",
                "## Precision / recall vs tau (validation)",
                "",
                _markdown_table(pd.DataFrame(curve_rows)),
                "",
                "## Results",
                "",
                _markdown_table(summary_frame),
                "",
                "## Oracle ceiling",
                "",
                "The best lead time achievable under the frozen matching rule:",
                "no detector can hit a campaign before three of its accounts are",
                "visible.",
                "",
                _markdown_table(pd.DataFrame(ceiling)),
                "",
                "## Oracle breakdown by base_type and group",
                "",
                _markdown_table(pd.DataFrame(breakdown_rows)),
                "",
                "## Timing",
                "",
                steps_markdown(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'evaluate_scorers.md'} (+ .json, .csv, tau curve, breakdown)")
    return 0 if ok else 1


def command_make_sample(args: argparse.Namespace) -> int:
    """Build the 1% development sample from the full data (runs on Kaggle).

    Keeps every transaction named in the pattern file plus a seeded random
    1% of the rest, and copies the pattern file unchanged. Download the two
    files into backend/data/sample/ afterwards.
    """
    sys.path.insert(0, str(BACKEND / "scripts"))
    from make_sample import build_sample

    from fraudcamp import load_data

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_data.load_config(Path(args.config))
    trans_csv = load_data.resolve_path(Path(args.config), config["data"]["trans_csv"])
    patterns_txt = load_data.resolve_path(Path(args.config), config["data"]["patterns_txt"])

    with step("build the 1% sample"):
        written = build_sample(trans_csv, patterns_txt, out)
    for path in written:
        print(f"  {path}: {path.stat().st_size / (1024 * 1024):,.2f} MB")
    print("\nDownload both files into backend/data/sample/.")
    return 0


# --------------------------------------------------------------------------
# not yet implemented
# --------------------------------------------------------------------------


def command_not_yet(args: argparse.Namespace) -> int:
    print(
        f"`{args.command}` is not implemented yet. It arrives with Phase 3 "
        "stage 3.3 (XGBoost baseline) and 3.4 (GNN). Nothing to run.",
        file=sys.stderr,
    )
    return 2


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command",
        choices=[
            "verify", "pipeline", "features-smoke", "evaluate-scorers",
            "make-sample", "train", "evaluate",
        ],
    )
    parser.add_argument(
        "--config",
        default=str(BACKEND / "configs" / "kaggle.yaml"),
        help="pipeline config YAML (default: configs/kaggle.yaml)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="directory for reports and archives (default: /kaggle/working)",
    )
    parser.add_argument(
        "--splits", default="val,test,stress",
        help="evaluate-scorers: horizons to evaluate (default: val,test,stress)",
    )
    parser.add_argument(
        "--lookbacks", default="",
        help="evaluate-scorers: comma-separated lookbacks (default: the candidates)",
    )
    parser.add_argument(
        "--tau-grid", default="",
        help="evaluate-scorers: comma-separated taus to sweep on validation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="bootstrap resamples for confidence intervals",
    )
    args = parser.parse_args()

    handlers = {
        "verify": command_verify,
        "pipeline": command_pipeline,
        "features-smoke": command_features_smoke,
        "evaluate-scorers": command_evaluate_scorers,
        "make-sample": command_make_sample,
        "train": command_not_yet,
        "evaluate": command_not_yet,
    }
    started = time.perf_counter()
    code = handlers[args.command](args)
    print(f"\nTotal elapsed: {time.perf_counter() - started:,.1f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
