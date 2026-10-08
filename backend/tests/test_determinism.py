"""The visibility build must be reproducible from its seed alone.

Account collections start life as set unions of transaction endpoints, and
Python randomises string hashing per process, so set iteration order differs
between runs. Before this was fixed, the seeded search followed a different
trajectory in every process: identical seed, different assignment. Two
Kaggle runs of the same pipeline disagreed on how many campaigns were
"unfragmentable" (3 vs 2) because of it.

In-process tests cannot catch this, because one process has one hash seed.
These tests therefore run the build in *subprocesses* with explicit,
differing PYTHONHASHSEED values.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import write_mini_dataset
from fraudcamp import pipeline

BACKEND = Path(__file__).resolve().parent.parent
SRC = BACKEND / "src"

HASH_SEEDS = ["1", "2", "3"]

# Runs one visibility build and prints the result as JSON. Kept as a string
# so it can be executed in a fresh interpreter with a chosen hash seed.
PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from fraudcamp import pipeline

result = pipeline.build(sys.argv[2])
ordered = pipeline.ordered_reassignable(
    result.camp, result.edges_by_campaign, result.accounts_by_campaign
)
assignment, achieved = pipeline.run_visibility(
    ordered, result.edges_by_campaign, result.bank2inst, float(sys.argv[3]), 42
)
print(json.dumps({
    "assignment": {k: int(v) for k, v in sorted(assignment.items())},
    "achieved": {str(k): round(float(v), 10) for k, v in sorted(achieved.items())},
    "accounts": {str(c): list(a) for c, a in sorted(result.accounts_by_campaign.items())},
    "unfragmentable": int((result.camp["group"] == "unfragmentable").sum()),
}, sort_keys=True))
"""


def _run_in_subprocess(config_path: Path, hash_seed: str, target: float = 0.25) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(SRC), str(config_path), str(target)],
        capture_output=True,
        text=True,
        env=env,
        cwd=BACKEND,
    )
    assert completed.returncode == 0, (
        f"probe failed under PYTHONHASHSEED={hash_seed}:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory) -> Path:
    """One dataset for the whole module, so the subprocess runs below can be
    shared instead of rebuilt per test."""
    return write_mini_dataset(tmp_path_factory.mktemp("determinism"))


@pytest.fixture(scope="module")
def _probe_results(shared_dataset) -> dict[str, dict]:
    """One subprocess run per hash seed, shared across the tests below so the
    pipeline is built three times rather than a dozen."""
    return {seed: _run_in_subprocess(shared_dataset, seed) for seed in HASH_SEEDS}


def test_account_order_is_identical_across_processes(_probe_results):
    """The root cause: account enumeration order must not depend on hashing."""
    reference = _probe_results[HASH_SEEDS[0]]["accounts"]
    for seed in HASH_SEEDS[1:]:
        assert _probe_results[seed]["accounts"] == reference, (
            f"account order differs under PYTHONHASHSEED={seed}"
        )


def test_assignment_is_identical_across_processes(_probe_results):
    reference = _probe_results[HASH_SEEDS[0]]["assignment"]
    for seed in HASH_SEEDS[1:]:
        assert _probe_results[seed]["assignment"] == reference, (
            f"account->institution map differs under PYTHONHASHSEED={seed}"
        )


def test_achieved_visibility_is_identical_across_processes(_probe_results):
    reference = _probe_results[HASH_SEEDS[0]]["achieved"]
    for seed in HASH_SEEDS[1:]:
        assert _probe_results[seed]["achieved"] == reference, (
            f"achieved visibility differs under PYTHONHASHSEED={seed}"
        )


def test_unfragmentable_count_is_identical_across_processes(_probe_results):
    """The symptom that exposed the bug: this count comes from the seed-42
    floor run, so it must not vary between processes."""
    counts = {seed: r["unfragmentable"] for seed, r in _probe_results.items()}
    assert len(set(counts.values())) == 1, f"unfragmentable count varies: {counts}"


def test_serialized_result_is_byte_identical_across_processes(_probe_results):
    """Strictest form: the whole JSON payload, byte for byte."""
    payloads = {
        seed: json.dumps(result, sort_keys=True)
        for seed, result in _probe_results.items()
    }
    assert len(set(payloads.values())) == 1, "serialized build differs across processes"


@pytest.mark.parametrize("target", [1.0, 0.75, 0.5, 0.25])
def test_in_process_runs_match(mini_dataset, target):
    """Two builds in the same process must agree at every target."""
    results = []
    for _ in range(2):
        result = pipeline.build(mini_dataset)
        ordered = pipeline.ordered_reassignable(
            result.camp, result.edges_by_campaign, result.accounts_by_campaign
        )
        results.append(
            pipeline.run_visibility(
                ordered, result.edges_by_campaign, result.bank2inst, target, 42
            )
        )
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_accounts_are_sorted_sequences_not_sets(mini_dataset):
    """Guards the invariant directly: if accounts ever become a set again,
    the subprocess tests above would be the only thing standing between the
    project and silently irreproducible experiments."""
    result = pipeline.build(mini_dataset)
    for campaign_id, accounts in result.accounts_by_campaign.items():
        assert not isinstance(accounts, (set, frozenset)), (
            f"campaign {campaign_id}: accounts must be an ordered sequence"
        )
        assert list(accounts) == sorted(accounts), f"campaign {campaign_id}: not sorted"
