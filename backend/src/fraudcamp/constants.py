"""
All frozen research definitions, in one place, so the pipeline and the
FastAPI `/methodology` endpoint can never disagree.

See README.md "Frozen research definitions" for the reasoning behind each
of these. Do not change any value here without explicit approval — see
CLAUDE.md.
"""

from __future__ import annotations

import datetime as _dt

# ---------------------------------------------------------------------------
# Raw data shape
# ---------------------------------------------------------------------------

TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

# ---------------------------------------------------------------------------
# Cutoff
# ---------------------------------------------------------------------------

CUTOFF = _dt.datetime(2022, 9, 11, 0, 0)

# ---------------------------------------------------------------------------
# Campaign evaluation
# ---------------------------------------------------------------------------

EVAL_OK_MIN_TXN = 3
EVAL_OK_MIN_ACCOUNTS = 3

# ---------------------------------------------------------------------------
# Splits, by campaign start date
# ---------------------------------------------------------------------------

TRAIN_END = _dt.datetime(2022, 9, 5, 0, 0)  # start < this -> train
VAL_END = _dt.datetime(2022, 9, 6, 0, 0)  # start < this -> val
TEST_END = _dt.datetime(2022, 9, 8, 0, 0)  # start < this -> test
# start >= TEST_END (and before/through the stress window, Sept 8-10) -> stress

SPLIT_ORDER = ["train", "val", "test", "stress"]

# ---------------------------------------------------------------------------
# Groups / topology
# ---------------------------------------------------------------------------

HUB_TYPES = frozenset({"FAN-IN", "FAN-OUT", "GATHER-SCATTER"})

# Seed and target used to determine, once, which fragmentable campaigns can
# never be pulled below 0.9 visibility ("unfragmentable").
UNFRAGMENTABLE_SEED = 42
UNFRAGMENTABLE_TARGET = 0.25
UNFRAGMENTABLE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Institutions
# ---------------------------------------------------------------------------

K_INSTITUTIONS = 8

# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

VISIBILITY_TARGETS = [1.0, 0.75, 0.5, 0.25]
VISIBILITY_SEEDS = [0, 1, 2, 3, 4]
LOCAL_SEARCH_ITERS = 400

# Bin thresholds: "100" >= 0.9; "75" in [0.65, 0.9); "50" in [0.45, 0.65);
# "low" < 0.45.
VISIBILITY_BIN_THRESHOLDS = {
    "100": 0.9,
    "75": 0.65,
    "50": 0.45,
}
VISIBILITY_BIN_ORDER = ["100", "75", "50", "low"]

# ---------------------------------------------------------------------------
# Planned evaluation rules (Phase 3 — documented here, not implemented yet)
# ---------------------------------------------------------------------------

DETECTION_WINDOW_STEP_H = 6
DETECTION_WINDOW_LOOKBACK_MIN_H = 24
DETECTION_WINDOW_LOOKBACK_MAX_H = 72
MATCH_MIN_ACCOUNT_FRACTION = 0.5
MATCH_MIN_ACCOUNTS = 3
TEST_SUBSETS = ["all", "fully_observed", "no_shared_accounts"]
CONDITIONS = ["isolated", "fedavg_only", "fedavg_embedding", "centralized"]

# ---------------------------------------------------------------------------
# Phase 3 detection windows
# ---------------------------------------------------------------------------
# A detection time t is a moment at which a model is asked "is a campaign
# forming?". At t it may see ONLY transactions in [t - L, t); never t itself,
# and never anything after it.

DATA_START = _dt.datetime(2022, 9, 1, 0, 0)

#: Lookback candidates to choose between on validation. Development uses 24h.
#: 72h was dropped on 2026-10-10 — see README decision log.
LOOKBACKS_H = (24, 48)

#: Dropped as a candidate, but still measured once by `features-smoke` so the
#: reason for dropping it is on the record rather than only in prose.
EXCLUDED_LOOKBACKS_H = (72,)

#: Everything that gets measured: the candidates plus the excluded ones.
MEASURED_LOOKBACKS_H = LOOKBACKS_H + EXCLUDED_LOOKBACKS_H

DEFAULT_LOOKBACK_H = 24

# Detection-time splits. These are about *when detection runs*, and are
# distinct from the campaign splits above, which are about when a campaign
# starts. Boundaries are inclusive of the upper end.
DETECTION_TRAIN_END = _dt.datetime(2022, 9, 5, 0, 0)  # t <= this -> train
DETECTION_VAL_END = _dt.datetime(2022, 9, 6, 0, 0)  # t <= this -> val
DETECTION_TEST_END = CUTOFF  # t <= this -> test
DETECTION_SPLIT_ORDER = ["train", "val", "test"]

# --- Rule 2: evaluation horizons -------------------------------------------
# Which detection times a split's campaigns are *evaluated* over. Separate
# from the training-data splits above: validation campaigns start on Sept 5
# but need until Sept 8 to run their course, so the validation horizon
# deliberately overlaps the test horizon. Only validation-campaign labels
# are used for tau selection, so the overlap leaks nothing.
# Bounds are (exclusive start, inclusive end], matching the grid.
EVALUATION_HORIZONS: dict[str, tuple[_dt.datetime, _dt.datetime]] = {
    "val": (_dt.datetime(2022, 9, 5, 0, 0), _dt.datetime(2022, 9, 8, 0, 0)),
    "test": (_dt.datetime(2022, 9, 6, 0, 0), CUTOFF),
    # Frozen 2026-10-10. Stress campaigns start Sept 8-10; the horizon runs
    # to the cutoff, capped per campaign by its own deadline. Other-split
    # hits apply here exactly as in rule 1.
    "stress": (_dt.datetime(2022, 9, 8, 0, 0), CUTOFF),
}

# --- Rule 3: training windows must have their full lookback ----------------
# A training detection time is usable only when t - L >= DATA_START, so no
# training window is silently short. Evaluation horizons are unaffected:
# they all start days after DATA_START.
REQUIRE_FULL_LOOKBACK_FOR_TRAINING = True

# --- Rule 1: what a detected cluster can turn out to be --------------------
HIT_KINDS = ["hit", "other_split_hit", "ambiguous", "false_alarm"]
