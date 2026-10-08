import pandas as pd

from fraudcamp import load_data


def _write_trans_csv(path, rows):
    columns = [
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
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_leading_zero_banks_and_account_ids(tmp_path):
    csv = tmp_path / "trans.csv"
    _write_trans_csv(
        csv,
        [
            ["2022/09/01 00:06", "021174", "800737690", "012", "80011F990", 100.0, "US Dollar", 100.0, "US Dollar", "ACH", 0],
        ],
    )
    df = load_data.load_transactions(csv)
    assert df.loc[0, "From Bank"] == 21174
    assert df.loc[0, "To Bank"] == 12
    assert df.loc[0, "src"] == "21174_800737690"
    assert df.loc[0, "dst"] == "12_80011F990"


def test_timestamp_parsed(tmp_path):
    csv = tmp_path / "trans.csv"
    _write_trans_csv(
        csv,
        [["2022/09/01 00:06", "1", "A", "1", "B", 1.0, "US Dollar", 1.0, "US Dollar", "ACH", 0]],
    )
    df = load_data.load_transactions(csv)
    assert df.loc[0, "ts"] == pd.Timestamp("2022-09-01 00:06")


def test_apply_cutoff():
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2022-09-10 23:59"), pd.Timestamp("2022-09-11 00:00"), pd.Timestamp("2022-09-11 00:01")],
        }
    )
    import datetime as dt

    out = load_data.apply_cutoff(df, cutoff=dt.datetime(2022, 9, 11, 0, 0))
    assert len(out) == 1


def test_account_bank():
    assert load_data.account_bank("021174_800737690") == 21174
