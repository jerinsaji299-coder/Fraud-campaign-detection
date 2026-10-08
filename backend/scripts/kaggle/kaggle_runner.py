#!/usr/bin/env python
"""Run this repository's heavy steps inside a Kaggle notebook.

Subcommands:
    verify     run the full-data regression tests and print a PASS/FAIL table
               of every number in README "Verified numbers"
    pipeline   run the full pipeline, export the artifacts, and zip them to
               <out>/artifacts.zip for download
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
    parser.add_argument("command", choices=["verify", "pipeline", "train", "evaluate"])
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
        "train": command_not_yet,
        "evaluate": command_not_yet,
    }
    started = time.perf_counter()
    code = handlers[args.command](args)
    print(f"\nTotal elapsed: {time.perf_counter() - started:,.1f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
