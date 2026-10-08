"""Full-data regression tests against the README "Verified numbers" table.

Expectations live in `fraudcamp.regression` so this suite and the Kaggle
runner's PASS/FAIL table can never disagree. Skipped automatically when the
full dataset is not present.

When FRAUDCAMP_REGRESSION_OUT is set, the computed numbers are written there
as JSON, so the runner can print the comparison table without paying for a
second full pipeline run.
"""

import json
import os
from pathlib import Path

import pytest

from fraudcamp import load_data, pipeline, regression

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "kaggle.yaml"

pytestmark = pytest.mark.fulldata


def _data_available() -> bool:
    try:
        config = load_data.load_config(CONFIG_PATH)
        trans_csv = load_data.resolve_path(CONFIG_PATH, config["data"]["trans_csv"])
        patterns_txt = load_data.resolve_path(CONFIG_PATH, config["data"]["patterns_txt"])
        return Path(trans_csv).exists() and Path(patterns_txt).exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def actuals() -> dict[str, int]:
    if not _data_available():
        pytest.skip("full dataset not available locally")

    result = pipeline.build(CONFIG_PATH)
    computed = regression.compute_actuals(result)

    out = os.environ.get("FRAUDCAMP_REGRESSION_OUT")
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "actuals": computed,
                    "expected": regression.EXPECTED,
                    "run": regression.summarize_run(result),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return computed


@pytest.mark.parametrize("metric", regression.METRICS, ids=lambda m: m.key)
def test_regression_number(metric: regression.Metric, actuals: dict[str, int]):
    assert actuals[metric.key] == metric.expected, (
        f"{metric.label}: expected {metric.expected:,}, got {actuals[metric.key]:,} "
        f"(evidence: {metric.evidence})"
    )


def test_every_expected_metric_was_computed(actuals: dict[str, int]):
    assert set(actuals) >= set(regression.EXPECTED)
