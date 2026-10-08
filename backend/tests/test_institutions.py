import pandas as pd

from fraudcamp import institutions


def test_bank_volume_counts_sender_and_receiver():
    df = pd.DataFrame(
        {
            "From Bank": [1, 1, 2],
            "To Bank": [2, 3, 3],
        }
    )
    vol = institutions.bank_volume(df)
    assert vol[1] == 2
    assert vol[2] == 2
    assert vol[3] == 2


def test_assign_institutions_is_balanced_and_deterministic():
    vol = pd.Series({10: 100, 20: 50, 30: 50})
    bank2inst_a, loads_a = institutions.assign_institutions(vol, k=2)
    bank2inst_b, loads_b = institutions.assign_institutions(vol, k=2)
    assert bank2inst_a == bank2inst_b
    assert bank2inst_a[10] == 0  # largest bank goes to institution 0 first
    assert loads_a[0] == 100
    assert sorted(loads_a) == sorted(loads_b)


def test_natural_institution_lookup():
    bank2inst = {21174: 3}
    assert institutions.natural_institution("21174_800737690", bank2inst) == 3
