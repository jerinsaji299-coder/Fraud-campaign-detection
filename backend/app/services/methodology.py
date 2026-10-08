"""The frozen research definitions as structured JSON.

Every *value* here is read from `fraudcamp.constants` — the same module
the pipeline uses — so the published methodology can never drift from
what the code actually did. Only the prose reasons live here.
"""

from __future__ import annotations

from fraudcamp import constants


def build_methodology() -> dict:
    entries = [
        {
            "key": "cutoff",
            "title": "Cutoff",
            "definition": (
                f"Model input is every transaction before {constants.CUTOFF.isoformat()}. "
                "Data from on/after that moment is kept only to establish a campaign's "
                "true completion time."
            ),
            "value": {"cutoff": constants.CUTOFF.isoformat()},
            "reason": (
                "Only 1,108 rows remain after the cutoff and 59.1% of them are laundering, "
                "versus about 0.1% overall. Training on that tail would let a model learn "
                "'late in the file means laundering' instead of real behaviour."
            ),
        },
        {
            "key": "campaign",
            "title": "What counts as a campaign",
            "definition": (
                "A campaign is one BEGIN/END block of the pattern file. Its campaign_id is "
                "its order of appearance, and its base_type is the text after '- ' and "
                "before any ':'. Every pattern transaction must match a row in the "
                "transaction table on timestamp, both banks, both accounts, and amount paid."
            ),
            "value": {"base_types": sorted(set(constants.HUB_TYPES) | {
                "CYCLE", "SCATTER-GATHER", "BIPARTITE", "STACK", "RANDOM",
            })},
            "reason": (
                "The pattern file is the only ground truth for which transactions belong "
                "to the same laundering attempt. Matching it back onto the transaction "
                "table is what makes campaign-level evaluation possible at all."
            ),
        },
        {
            "key": "eval_ok",
            "title": "Which campaigns are evaluated",
            "definition": (
                f"A campaign is eval_ok when it has at least {constants.EVAL_OK_MIN_TXN} "
                f"transactions and at least {constants.EVAL_OK_MIN_ACCOUNTS} accounts."
            ),
            "value": {
                "min_transactions": constants.EVAL_OK_MIN_TXN,
                "min_accounts": constants.EVAL_OK_MIN_ACCOUNTS,
            },
            "reason": (
                "About a quarter of campaigns have only one or two transactions, where "
                "'how early did we discover it' has no meaningful answer."
            ),
        },
        {
            "key": "splits",
            "title": "Splits",
            "definition": (
                "Campaigns are split by start date: train = Sept 1-4, val = Sept 5, "
                "test = Sept 6-7, stress = Sept 8-10."
            ),
            "value": {
                "train_end": constants.TRAIN_END.isoformat(),
                "val_end": constants.VAL_END.isoformat(),
                "test_end": constants.TEST_END.isoformat(),
                "stress_end": constants.CUTOFF.isoformat(),
                "order": constants.SPLIT_ORDER,
            },
            "reason": (
                "An earlier split used Sept 8-10 as the test set, but 74 of its 76 "
                "evaluable campaigns were still unfinished at the cutoff. Sept 6-7 keeps "
                "most test campaigns fully observed; Sept 8-10 becomes a deliberate "
                "stress split of heavily censored campaigns."
            ),
        },
        {
            "key": "groups",
            "title": "Topology groups",
            "definition": (
                f"Campaigns of type {', '.join(sorted(constants.HUB_TYPES))} are 'hub'; all "
                "other types are 'fragmentable'. A fragmentable campaign whose lowest "
                f"achievable visibility (seed {constants.UNFRAGMENTABLE_SEED}, target "
                f"{constants.UNFRAGMENTABLE_TARGET}) is at least "
                f"{constants.UNFRAGMENTABLE_THRESHOLD} is relabelled 'unfragmentable'. "
                "The control group is hub plus unfragmentable."
            ),
            "value": {
                "hub_types": sorted(constants.HUB_TYPES),
                "unfragmentable_seed": constants.UNFRAGMENTABLE_SEED,
                "unfragmentable_target": constants.UNFRAGMENTABLE_TARGET,
                "unfragmentable_threshold": constants.UNFRAGMENTABLE_THRESHOLD,
            },
            "reason": (
                "In a hub campaign every transaction touches the same hub account, so that "
                "account's own bank always sees 100% of it — visibility cannot be reduced, "
                "which makes these campaigns a control group rather than a treatment group. "
                "A few small fragmentable campaigns behave the same way in practice. Note "
                "that unfragmentable campaigns are still reassigned like any other "
                "fragmentable campaign; the relabel only affects how results are grouped."
            ),
        },
        {
            "key": "institutions",
            "title": "Institutions",
            "definition": (
                f"Banks are grouped into K = {constants.K_INSTITUTIONS} institutions. Bank "
                "volume is its number of pre-cutoff transactions as sender or receiver; "
                "banks are taken largest first and each is given to the institution with "
                "the smallest load so far. Ties break by bank id, and banks with no "
                "pre-cutoff activity are assigned last. An institution sees a transaction "
                "if the sender's or the receiver's bank belongs to it."
            ),
            "value": {"k": constants.K_INSTITUTIONS},
            "reason": (
                "Every transaction is visible to both endpoint banks, so the lowest "
                "visibility any campaign can structurally reach is about 2/K. With K = 3 "
                "that floor is around 67%, far above the lowest target of 25%, so K was "
                "raised to 8. The assignment is fully deterministic so runs are reproducible."
            ),
        },
        {
            "key": "visibility",
            "title": "Visibility",
            "definition": (
                "A campaign's visibility is the largest fraction of its transactions that "
                "any single institution can see. For each (seed, target) only the accounts "
                f"of reassignable campaigns are moved; targets are {constants.VISIBILITY_TARGETS}. "
                "At target 1.0 a campaign's accounts all go to one random institution; "
                "otherwise a local search makes "
                f"{constants.LOCAL_SEARCH_ITERS} single-account moves, keeping a move only "
                "when it does not worsen the distance to the target. An account shared by "
                "several campaigns is assigned by its earliest campaign, and every "
                "campaign's achieved visibility is recomputed from the final assignment."
            ),
            "value": {
                "targets": constants.VISIBILITY_TARGETS,
                "seeds": constants.VISIBILITY_SEEDS,
                "local_search_iterations": constants.LOCAL_SEARCH_ITERS,
            },
            "reason": (
                "This is the independent variable of the whole study: it simulates each "
                "bank holding only a fragment of a campaign. Reassigning an account moves "
                "all of its transactions, so the resulting partition stays internally "
                "consistent rather than being faked per campaign."
            ),
        },
        {
            "key": "visibility_bins",
            "title": "Visibility bins",
            "definition": (
                "Analysis groups runs by achieved visibility, not by the requested target: "
                "'100' at 0.9 and above, '75' in [0.65, 0.9), '50' in [0.45, 0.65), 'low' "
                "below 0.45."
            ),
            "value": {
                "thresholds": constants.VISIBILITY_BIN_THRESHOLDS,
                "order": constants.VISIBILITY_BIN_ORDER,
            },
            "reason": (
                "Local search frequently misses its target — at target 0.25 the median "
                "campaign only reaches about 0.33, and small campaigns bottom out near "
                "0.5. Reporting by what was actually achieved is honest; reporting by the "
                "requested target would not be."
            ),
        },
        {
            "key": "shared_accounts",
            "title": "Shared accounts",
            "definition": (
                "A test campaign is flagged shares_train_account when it shares any account "
                "with a train-split campaign."
            ),
            "value": {},
            "reason": (
                "100 accounts appear in more than one campaign, 83 of them across splits, "
                "affecting 7 of 37 fragmentable test campaigns. A model could recognise the "
                "account rather than generalise, so results are reported with and without them."
            ),
        },
        {
            "key": "evaluation",
            "title": "Planned evaluation rules (Phase 3, not yet implemented)",
            "definition": (
                f"Detection runs in {constants.DETECTION_WINDOW_STEP_H}-hour steps with a "
                f"{constants.DETECTION_WINDOW_LOOKBACK_MIN_H}-"
                f"{constants.DETECTION_WINDOW_LOOKBACK_MAX_H} hour lookback and never sees "
                "future transactions. A detected cluster hits a campaign when at least "
                f"{int(constants.MATCH_MIN_ACCOUNT_FRACTION * 100)}% of the cluster's "
                f"accounts belong to it and it covers at least {constants.MATCH_MIN_ACCOUNTS} "
                "of the campaign's accounts. Lead time is measured against true completion "
                "in hours, normalised by duration, and as the fraction of the campaign "
                "observed at detection; a campaign not detected by its deadline is missed. "
                f"Test results are reported on these subsets: {', '.join(constants.TEST_SUBSETS)}."
            ),
            "value": {
                "window_step_h": constants.DETECTION_WINDOW_STEP_H,
                "lookback_min_h": constants.DETECTION_WINDOW_LOOKBACK_MIN_H,
                "lookback_max_h": constants.DETECTION_WINDOW_LOOKBACK_MAX_H,
                "match_min_account_fraction": constants.MATCH_MIN_ACCOUNT_FRACTION,
                "match_min_accounts": constants.MATCH_MIN_ACCOUNTS,
                "test_subsets": constants.TEST_SUBSETS,
                "conditions": constants.CONDITIONS,
            },
            "reason": (
                "Causal windows are what make 'discovery lead time' a fair measurement: a "
                "detector must commit using only what was knowable at the time. The "
                "matching rule stops a cluster that merely brushes a campaign from counting "
                "as having found it."
            ),
        },
    ]
    return {"entries": entries, "n_entries": len(entries)}
