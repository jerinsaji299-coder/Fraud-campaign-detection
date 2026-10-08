import pandas as pd

from fraudcamp import campaigns, load_data

PATTERNS_TXT = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1
2022/09/01 09:14,021174,800737690,020,80006A5E0,35642.49,Yuan,35642.49,Yuan,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1
2022/09/01 02:52,020,80BC62F10,0240229,80F025640,7332.87,Swiss Franc,7332.87,Swiss Franc,ACH,1
END LAUNDERING ATTEMPT - CYCLE
"""


def _write_patterns(tmp_path):
    p = tmp_path / "patterns.txt"
    p.write_text(PATTERNS_TXT)
    return p


def test_parse_patterns_count_and_order(tmp_path):
    p = _write_patterns(tmp_path)
    attempts = campaigns.parse_patterns(p)
    assert len(attempts) == 2
    assert attempts[0]["type"].startswith("FAN-OUT")
    assert attempts[1]["type"].startswith("CYCLE")
    assert len(attempts[0]["rows"]) == 3
    assert len(attempts[1]["rows"]) == 2


def test_base_type_normalization(tmp_path):
    p = _write_patterns(tmp_path)
    attempts = campaigns.parse_patterns(p)
    pat = campaigns.build_pattern_df(attempts)
    assert set(pat["base_type"].unique()) == {"FAN-OUT", "CYCLE"}
    assert pat.loc[pat["campaign_id"] == 0, "base_type"].iloc[0] == "FAN-OUT"


def test_leading_zero_bank_match(tmp_path):
    p = _write_patterns(tmp_path)
    attempts = campaigns.parse_patterns(p)
    pat = campaigns.build_pattern_df(attempts)
    # "021174" in the pattern file must compare equal to an int bank id of 21174
    assert pat.loc[0, "From Bank"] == 21174


def test_match_patterns_all_match_and_tags_campaign_id(tmp_path):
    p = _write_patterns(tmp_path)
    attempts = campaigns.parse_patterns(p)
    pat = campaigns.build_pattern_df(attempts)

    # build a full transaction table containing exactly the pattern rows
    # plus one unrelated non-laundering row
    rows = []
    for _, r in pat.iterrows():
        rows.append(
            [r["Timestamp"], r["From Bank"], r["Account"], r["To Bank"], r["Account.1"], r["Amount Paid"], "US Dollar", r["Amount Paid"], "US Dollar", "ACH", 1]
        )
    rows.append(["2022/09/01 00:00", 99, "ZZZ", 99, "ZZZ", 1.0, "US Dollar", 1.0, "US Dollar", "ACH", 0])

    full_csv = tmp_path / "full.csv"
    pd.DataFrame(
        rows,
        columns=[
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
            "Payment Format", "Is Laundering",
        ],
    ).to_csv(full_csv, index=False)

    full_df = load_data.load_transactions(full_csv)
    tagged = campaigns.match_patterns(pat, full_df)

    assert (tagged["campaign_id"] != -1).sum() == len(pat)
    assert (tagged["campaign_id"] == -1).sum() == 1


def test_eval_ok_filter():
    camp = pd.DataFrame(
        {
            "campaign_id": [0, 1],
            "base_type": ["FAN-OUT", "CYCLE"],
            "n_txn": [3, 2],
            "n_accounts": [4, 2],
            "n_banks": [3, 2],
            "start": [pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-01")],
            "end": [pd.Timestamp("2022-09-02"), pd.Timestamp("2022-09-02")],
            "duration_h": [24.0, 24.0],
        }
    )
    out = campaigns.add_eval_fields(camp)
    assert out.loc[0, "eval_ok"] is True or out.loc[0, "eval_ok"] == True  # noqa: E712
    assert out.loc[1, "eval_ok"] == False  # noqa: E712
