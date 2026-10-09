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

    # The spec is fitted on ONE training window, not all of them: this
    # command measures sizes and timing, and fitting across all 16 training
    # windows would concatenate overlapping windows into tens of millions of
    # rows for no benefit here. Categories missing from one window land in
    # the __other__ column, which is exactly the designed behaviour.
    fit_t = pd.Timestamp(SMOKE_PROBES[0][2])
    with step(f"fit feature spec on the {fit_t} 24h training window"):
        fit_window = windows.build_window(full_df, fit_t, lookback_h=24)
        spec = features.fit_feature_spec([fit_window], lookback_h=24)
    print(f"  edge dims: {len(spec.edge_feature_names)}")
    print(f"  node dims: {len(spec.node_feature_names)}")
    print(f"  payment currencies: {len(spec.payment_currencies)}, "
          f"receiving: {len(spec.receiving_currencies)}, formats: {len(spec.payment_formats)}")
    del fit_window
    gc.collect()

    d_edge = len(spec.edge_feature_names)
    d_node = len(spec.node_feature_names)
    rows: list[dict] = []

    for split, day_kind, t_text in SMOKE_PROBES:
        t = pd.Timestamp(t_text)
        actual_split = windows.split_of_detection_time(t)
        if actual_split != split:
            print(
                f"\nWARNING: {t} is in split {actual_split!r}, expected {split!r}. "
                "Skipping so the table cannot mislabel a window.",
                flush=True,
            )
            continue

        for lookback in constants.LOOKBACKS_H:
            started = time.perf_counter()
            window = windows.build_window(full_df, t, lookback_h=lookback)
            slice_s = time.perf_counter() - started

            if window.n_transactions == 0:
                rows.append(
                    {
                        "split": split, "day": day_kind, "t": t_text,
                        "lookback_h": lookback, "n_edges": 0, "n_accounts": 0,
                        "n_laundering": 0, "slice_s": round(slice_s, 2),
                        "build_s": 0.0, "tensors_mb": 0.0, "xgb_matrix_mb": 0.0,
                        "peak_rss_mb": memory_mb()[0],
                    }
                )
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
            rows.append(
                {
                    "split": split,
                    "day": day_kind,
                    "t": t_text,
                    "lookback_h": lookback,
                    "n_edges": wf.n_edges,
                    "n_accounts": wf.n_nodes,
                    "n_laundering": int(wf.labels.sum()),
                    "laundering_rate": round(wf.positive_rate, 6),
                    "class_weight": round(wf.class_weight(), 1),
                    "slice_s": round(slice_s, 2),
                    "build_s": round(build_s, 2),
                    "tensors_mb": round(tensors / 1e6, 1),
                    # analytic: building it would double peak memory for no
                    # extra information
                    "xgb_matrix_mb": round(
                        wf.n_edges * (d_edge + 2 * d_node) * 4 / 1e6, 1
                    ),
                    "peak_rss_mb": round(memory, 0) if memory else None,
                }
            )
            print(
                f"  {split:<5} L={lookback:<2} {t_text}  "
                f"edges={wf.n_edges:>9,}  accounts={wf.n_nodes:>8,}  "
                f"laundering={int(wf.labels.sum()):>5,}  "
                f"build={build_s:>5.1f}s  tensors={tensors / 1e6:>7.1f} MB",
                flush=True,
            )
            del wf, window
            gc.collect()

    if not rows:
        print("No windows were built; nothing to report.", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows)
    largest = table.loc[table["n_edges"].idxmax()]
    gine = estimate_gine_memory_mb(
        n_nodes=int(largest["n_accounts"]),
        n_edges=int(largest["n_edges"]),
        d_node=d_node,
        d_edge=d_edge,
    )

    print("\n=== largest window ===")
    print(
        f"  {largest['t']} L={largest['lookback_h']}h ({largest['split']}): "
        f"{int(largest['n_edges']):,} edges, {int(largest['n_accounts']):,} accounts"
    )
    print("\n=== estimated full-batch GPU memory, 3-layer GINEConv, hidden 64 ===")
    for key, value in gine.items():
        print(f"  {key.replace('_', ' ')}: {value:,.1f} MB")
    print(
        "  (an estimate from tensor shapes, not a measurement; if this "
        "approaches the GPU's capacity, use neighbour sampling instead of "
        "full-batch, which the Phase 3 design already permits)"
    )

    csv_path = out / "features_smoke.csv"
    table.to_csv(csv_path, index=False)
    json_path = out / "features_smoke.json"
    json_path.write_text(
        json.dumps(
            {
                "edge_dims": d_edge,
                "node_dims": d_node,
                "spec_fitted_on": spec.fitted_on,
                "windows": rows,
                "largest_window": {k: (int(v) if isinstance(v, (int, float)) and k in ("n_edges", "n_accounts") else v) for k, v in largest.to_dict().items()},
                "gine_estimate_mb": gine,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    report = out / "features_smoke.md"
    report.write_text(
        "\n".join(
            [
                "# Feature smoke test on the full dataset",
                "",
                f"- edge features: {d_edge} dims",
                f"- node features: {d_node} dims",
                f"- spec fitted on: `{spec.fitted_on}`",
                "",
                "Validation spans only Mon Sept 5 to Tue Sept 6, so it has no",
                "weekend detection time to sample.",
                "",
                "## Windows",
                "",
                _markdown_table(table),
                "",
                "## Largest window",
                "",
                f"`{largest['t']}` at L={largest['lookback_h']}h "
                f"({largest['split']}): {int(largest['n_edges']):,} edges, "
                f"{int(largest['n_accounts']):,} accounts.",
                "",
                "## Estimated full-batch GPU memory (3-layer GINEConv, hidden 64)",
                "",
                *(f"- {k.replace('_', ' ')}: {v:,.1f} MB" for k, v in gine.items()),
                "",
                "Estimated from tensor shapes, not measured.",
                "",
                "## Timing",
                "",
                steps_markdown(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report}")
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
        choices=["verify", "pipeline", "features-smoke", "train", "evaluate"],
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
    args = parser.parse_args()

    handlers = {
        "verify": command_verify,
        "pipeline": command_pipeline,
        "features-smoke": command_features_smoke,
        "train": command_not_yet,
        "evaluate": command_not_yet,
    }
    started = time.perf_counter()
    code = handlers[args.command](args)
    print(f"\nTotal elapsed: {time.perf_counter() - started:,.1f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
