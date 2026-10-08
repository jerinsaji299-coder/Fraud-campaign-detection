"""Visibility targets, local search, conflict resolution, and bins.

Visibility of a campaign = max over institutions of the fraction of the
campaign's transactions that institution sees. For a given (seed, target),
only accounts belonging to eval_ok fragmentable campaigns are ever moved
off their natural (bank-based) institution; everything else keeps its
natural institution. Conflicts (an account shared by several campaigns)
are resolved in favor of the account's earliest campaign; achieved
visibility is then recomputed for every campaign from the final global
assignment. See README.md "Frozen research definitions".

Determinism
-----------
Every random step is seeded, but a seed only reproduces a run if the RNG is
asked for things in the same order. Account collections arrive here as
Python sets (a campaign's accounts are a set union of its endpoints), and
string hashing is randomised per process, so set iteration order differs
between runs. That made the seeded draws follow a different trajectory each
process: same seed, different assignment.

So every collection is sorted before anything random touches it. The search
itself is unchanged -- same targets, same iteration count, same seeds; only
the enumeration order is pinned.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import pandas as pd

from . import constants, institutions


def visibility_of(edges: list[tuple[str, str]], assignment: dict[str, int], k: int) -> float:
    """max over institutions of the fraction of edges that institution sees."""
    if not edges:
        return 0.0
    best = 0.0
    for inst in range(k):
        seen = sum(1 for s, d in edges if assignment.get(s) == inst or assignment.get(d) == inst)
        frac = seen / len(edges)
        if frac > best:
            best = frac
    return best


def _local_search_free(
    edges: list[tuple[str, str]],
    free: Iterable[str],
    locked: dict[str, int],
    target: float,
    rng: random.Random,
    k: int = constants.K_INSTITUTIONS,
    iters: int = constants.LOCAL_SEARCH_ITERS,
) -> tuple[dict[str, int], float]:
    """Randomly initialize `free` accounts, then hill-climb toward target
    visibility by moving one free account at a time, keeping a move only
    if it does not increase |achieved - target|. `locked` accounts are
    held fixed throughout. Returns (assignment restricted to free
    accounts, achieved visibility over the whole campaign)."""
    free = sorted(free)  # see "Determinism" in this module's docstring
    assignment = dict(locked)
    for a in free:
        assignment[a] = rng.randrange(k)

    achieved = visibility_of(edges, assignment, k)
    if not free:
        return {}, achieved

    for _ in range(iters):
        a = rng.choice(free)
        old = assignment[a]
        assignment[a] = rng.randrange(k)
        new_achieved = visibility_of(edges, assignment, k)
        if abs(new_achieved - target) <= abs(achieved - target):
            achieved = new_achieved
        else:
            assignment[a] = old

    return {a: assignment[a] for a in free}, achieved


def assign_campaign(
    edges: list[tuple[str, str]],
    free: Iterable[str],
    locked: dict[str, int],
    target: float,
    rng: random.Random,
    k: int = constants.K_INSTITUTIONS,
) -> dict[str, int]:
    """Assign institutions to `free` accounts of one campaign given
    already-`locked` accounts (from earlier-starting campaigns that share
    an account), aiming for `target` visibility. Returns only the
    assignment for `free` accounts."""
    free = sorted(free)  # see "Determinism" in this module's docstring
    if target >= 0.999:
        if not free:
            return {}
        inst = rng.randrange(k)
        return {a: inst for a in free}
    new_assignment, _ = _local_search_free(edges, free, locked, target, rng, k)
    return new_assignment


def build_global_assignment(
    campaigns_ordered: Sequence[tuple[int, list[tuple[str, str]], Iterable[str]]],
    target: float,
    seed: int,
    k: int = constants.K_INSTITUTIONS,
) -> dict[str, int]:
    """Process eval_ok fragmentable campaigns in start-time order, assigning
    each campaign's not-yet-locked accounts toward `target` visibility.
    `campaigns_ordered` is a list of (campaign_id, edges, accounts) already
    sorted by campaign start time ascending. Returns the global account ->
    institution map (only for accounts that were reassigned).

    Accounts are sorted before anything random happens, so the result
    depends only on `seed` — see "Determinism" in this module's docstring.
    """
    rng = random.Random(seed)
    global_assignment: dict[str, int] = {}

    for _cid, edges, accounts in campaigns_ordered:
        ordered_accounts = sorted(accounts)
        locked = {
            a: global_assignment[a] for a in ordered_accounts if a in global_assignment
        }
        free = [a for a in ordered_accounts if a not in global_assignment]
        new = assign_campaign(edges, free, locked, target, rng, k)
        global_assignment.update(new)

    return global_assignment


def institution_of(account_id: str, global_assignment: dict[str, int], bank2inst: dict[int, int]) -> int:
    if account_id in global_assignment:
        return global_assignment[account_id]
    return institutions.natural_institution(account_id, bank2inst)


def recompute_achieved(
    campaigns: list[tuple[int, list[tuple[str, str]]]],
    global_assignment: dict[str, int],
    bank2inst: dict[int, int],
    k: int = constants.K_INSTITUTIONS,
) -> dict[int, float]:
    """Recompute achieved visibility for each campaign from the final
    global assignment (falling back to natural institution for any
    account not reassigned)."""
    result: dict[int, float] = {}
    for cid, edges in campaigns:
        assignment = {}
        for s, d in edges:
            assignment[s] = institution_of(s, global_assignment, bank2inst)
            assignment[d] = institution_of(d, global_assignment, bank2inst)
        result[cid] = visibility_of(edges, assignment, k)
    return result


def bin_of(achieved: float) -> str:
    t = constants.VISIBILITY_BIN_THRESHOLDS
    if achieved >= t["100"]:
        return "100"
    if achieved >= t["75"]:
        return "75"
    if achieved >= t["50"]:
        return "50"
    return "low"


def visible_transactions(
    trans_df: pd.DataFrame,
    institution: int,
    global_assignment: dict[str, int],
    bank2inst: dict[int, int],
) -> pd.DataFrame:
    """Return the subset of trans_df visible to `institution` under the
    given global account assignment (institution sees a transaction if
    the sender's or receiver's institution is `institution`)."""
    src_inst = trans_df["src"].map(lambda a: institution_of(a, global_assignment, bank2inst))
    dst_inst = trans_df["dst"].map(lambda a: institution_of(a, global_assignment, bank2inst))
    mask = (src_inst == institution) | (dst_inst == institution)
    return trans_df[mask]
