import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

BACKEND = Path(__file__).resolve().parent.parent
SRC = BACKEND / "src"
for path in (SRC, BACKEND):  # fraudcamp lives in src/, the `app` package in backend/
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CSV_COLUMNS = [
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

# (campaign_id, type line, timestamp, from bank, account, to bank, account.1, amount)
#
# campaign 0: CYCLE, starts Sept 1  -> train, fragmentable, eval_ok
# campaign 1: FAN-OUT, starts Sept 6 -> test, hub, eval_ok
# campaign 2: STACK, starts Sept 2   -> train, fragmentable, only 2 txns -> NOT eval_ok
# campaign 3: BIPARTITE, starts Sept 7 -> test, fragmentable, eval_ok,
#             shares account 1_A with campaign 0 (a train campaign)
CAMPAIGN_TXNS = [
    (0, "CYCLE:  Max 4 hops", "2022/09/01 00:10", 1, "A", 2, "B", 100.00),
    (0, "CYCLE:  Max 4 hops", "2022/09/01 02:00", 2, "B", 3, "C", 200.00),
    (0, "CYCLE:  Max 4 hops", "2022/09/01 05:00", 3, "C", 4, "D", 300.00),
    (0, "CYCLE:  Max 4 hops", "2022/09/02 05:00", 4, "D", 1, "A", 400.00),
    (1, "FAN-OUT:  Max 3-degree Fan-Out", "2022/09/06 01:00", 5, "H", 6, "L1", 500.00),
    (1, "FAN-OUT:  Max 3-degree Fan-Out", "2022/09/06 03:00", 5, "H", 7, "L2", 600.00),
    (1, "FAN-OUT:  Max 3-degree Fan-Out", "2022/09/06 09:00", 5, "H", 8, "L3", 700.00),
    (2, "STACK", "2022/09/02 01:00", 9, "X", 10, "Y", 800.00),
    (2, "STACK", "2022/09/02 02:00", 10, "Y", 11, "Z", 900.00),
    (3, "BIPARTITE", "2022/09/07 01:00", 1, "A", 12, "P", 1000.00),
    (3, "BIPARTITE", "2022/09/07 02:00", 1, "A", 13, "Q", 1100.00),
    (3, "BIPARTITE", "2022/09/07 03:00", 14, "R", 12, "P", 1200.00),
]

# (timestamp, from bank, account, to bank, account.1, amount, is_laundering)
# includes an unassigned laundering row (label 1, in no campaign), and a
# post-cutoff row whose bank (99) appears nowhere before the cutoff.
NON_CAMPAIGN_TXNS = [
    ("2022/09/01 00:00", 1, "N1", 2, "N2", 10.0, 0),
    ("2022/09/03 12:00", 2, "N2", 3, "N3", 20.0, 0),
    ("2022/09/05 12:00", 3, "N3", 4, "N4", 30.0, 0),
    ("2022/09/06 12:00", 4, "N4", 5, "N5", 40.0, 0),
    ("2022/09/09 12:00", 5, "N5", 6, "N6", 50.0, 1),
    ("2022/09/12 12:00", 99, "N99", 1, "N1", 60.0, 0),
]

N_CAMPAIGNS = 4
N_PATTERN_TXNS = len(CAMPAIGN_TXNS)
N_TOTAL_ROWS = len(CAMPAIGN_TXNS) + len(NON_CAMPAIGN_TXNS)


def _csv_row(ts, from_bank, acct, to_bank, acct1, amount, is_laundering):
    return [ts, from_bank, acct, to_bank, acct1, amount, "US Dollar", amount, "US Dollar", "ACH", is_laundering]


def _pattern_line(ts, from_bank, acct, to_bank, acct1, amount):
    # banks are written with leading zeros, as the real pattern file does
    return (
        f"{ts},{from_bank:05d},{acct},{to_bank:05d},{acct1},"
        f"{amount:.2f},US Dollar,{amount:.2f},US Dollar,ACH,1"
    )


@pytest.fixture
def mini_dataset(tmp_path):
    """A tiny but complete dataset: writes HI-Small-shaped CSV + pattern
    files and a config YAML pointing at them. Returns the config path."""
    rows = [_csv_row(ts, fb, a, tb, a1, amt, 1) for _, _, ts, fb, a, tb, a1, amt in CAMPAIGN_TXNS]
    rows += [_csv_row(*t) for t in NON_CAMPAIGN_TXNS]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    trans_csv = data_dir / "HI-Small_Trans.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(trans_csv, index=False)

    lines = []
    for cid in range(N_CAMPAIGNS):
        txns = [t for t in CAMPAIGN_TXNS if t[0] == cid]
        type_line = txns[0][1]
        base_type = type_line.split(":")[0].strip()
        lines.append(f"BEGIN LAUNDERING ATTEMPT - {type_line}")
        lines += [_pattern_line(ts, fb, a, tb, a1, amt) for _, _, ts, fb, a, tb, a1, amt in txns]
        lines.append(f"END LAUNDERING ATTEMPT - {base_type}")
        lines.append("")
    patterns_txt = data_dir / "HI-Small_Patterns.txt"
    patterns_txt.write_text("\n".join(lines))

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "trans_csv": str(trans_csv),
                    "patterns_txt": str(patterns_txt),
                },
                "artifacts_dir": str(tmp_path / "artifacts"),
            }
        )
    )
    return config_path
