#!/usr/bin/env python
"""Run the research pipeline end to end and print every number from the
README's "Verified numbers" table, so a run can be checked against it.
With --export, also write the artifacts the API serves.

Usage:
    python scripts/run_pipeline.py --config configs/local.yaml
    python scripts/run_pipeline.py --config configs/kaggle.yaml --export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraudcamp import constants, export, load_data, pipeline, splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a config YAML (local.yaml or kaggle.yaml)")
    parser.add_argument("--export", action="store_true", help="Also write artifacts to the configured artifacts dir")
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"Running pipeline with {config_path} ...")
    result = pipeline.build(config_path)

    full_df = result.full_df
    camp = result.camp

    print("\n--- Verified numbers ---")
    print("Total transaction rows:", len(full_df))
    print(
        f"Rows on/after CUTOFF ({constants.CUTOFF}):",
        int((full_df["ts"] >= constants.CUTOFF).sum()),
    )
    print("Campaigns (pattern-file attempts):", len(camp))
    print("Pattern-file transactions:", len(result.pattern_df), "(all matched)")
    print("Laundering rows (Is Laundering == 1):", int(full_df["Is Laundering"].sum()))
    print(
        "Unassigned laundering rows (label=1, campaign_id=-1):",
        int(((full_df["Is Laundering"] == 1) & (full_df["campaign_id"] == -1)).sum()),
    )

    print("\neval_ok campaigns (total):", int(camp["eval_ok"].sum()))
    print("eval_ok by split:")
    for split in constants.SPLIT_ORDER:
        print(f"  {split}: {int(((camp['split'] == split) & camp['eval_ok']).sum())}")

    # The 152 / 37 figures are the topology counts (eval_ok, non-hub base
    # type), i.e. before the unfragmentable relabel — that is how the
    # reference notebook counted them, and it is also exactly the set of
    # campaigns that gets reassigned.
    reassignable = splits.reassignable_mask(camp)
    print("\neval_ok fragmentable (reassignable) campaigns:", int(reassignable.sum()))
    print(
        "eval_ok fragmentable test campaigns:",
        int((reassignable & (camp["split"] == "test")).sum()),
    )
    print(
        "Fragmentable test campaigns with shares_train_account:",
        int((reassignable & (camp["split"] == "test") & camp["shares_train_account"]).sum()),
    )

    print("\nbase_type counts:")
    for base_type, n in camp["base_type"].value_counts().items():
        print(f"  {base_type}: {n}")

    print("\n--- Groups and institutions ---")
    print("Group counts (after the unfragmentable relabel):")
    for group, n in camp["group"].value_counts().items():
        print(f"  {group}: {n}")
    print(f"Institution loads (K={constants.K_INSTITUTIONS}):", result.loads)

    if args.export:
        config = load_data.load_config(config_path)
        artifacts_dir = load_data.resolve_path(config_path, config["artifacts_dir"])
        print(f"\nExporting artifacts to {artifacts_dir} ...")
        written = export.export_all(result, artifacts_dir)
        for name, path in written.items():
            size_kb = path.stat().st_size / 1024
            print(f"  {name}: {size_kb:.1f} KB")

    print("\nDone.")


if __name__ == "__main__":
    main()
