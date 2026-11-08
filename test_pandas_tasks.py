import pytest
import pandas as pd

from pandas_tasks import analyze_titanic
from grader_contracts.pandas_tasks import TitanicInput

def write_csv(tmp_path, rows, cols):
    path = tmp_path / "titanic.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return TitanicInput(csv_path=str(path))


COLS = ["PassengerId", "Age", "Pclass", "Survived", "Fare"]

@pytest.mark.parametrize(
    "rows,expected_count",
    [
        ([], 0),
        ([[1, 20.0, 1, 1, 5.0]], 1),
        ([[1, 20.0, 1, 1, 5.0], [2, 40.0, 2, 0, 10.0]], 2),
        ([[i, 25.0, 1, 1, 10.0] for i in range(1, 11)], 10),
    ],
)
def test_row_count(tmp_path, rows, expected_count):
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.row_count == expected_count

@pytest.mark.parametrize(
    "ages,expected",
    [
        ([], 0),
        ([10, 20, 29], 0),
        ([31, 50, 100], 3),
        ([10, 30, 31, 40], 2),  
        ([30.0, 30.1, 29.9], 1),
        ([None, 35, None], 1),
    ],
)
def test_adults_over_30(tmp_path, ages, expected):
    rows = [[i + 1, a if a is None else float(a), 1, 1, 10.0] for i, a in enumerate(ages)]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.adults_over_30_count == expected


def test_nan_age_not_zero(tmp_path):
    rows = [
        [1, 40.0, 1, 1, 10.0],
        [2, None, 1, 0, 20.0],
        [3, None, 1, 1, 30.0],
    ]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.mean_age_by_pclass[1] == pytest.approx(40.0)
    assert s.missing_by_column["Age"] == 2


def test_empty_dataframe(tmp_path):
    s = analyze_titanic(write_csv(tmp_path, [], COLS))
    assert s.row_count == 0
    assert s.adults_over_30_count == 0
    assert s.mean_age_by_pclass == {}
    assert s.survival_rate_by_pclass == {}
    assert s.highest_fares == []


def test_top5_fewer_rows(tmp_path):
    rows = [[i, 25.0, 1, 1, float(100 - i)] for i in range(3)]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert len(s.highest_fares) == 3
    assert s.highest_fares == sorted(s.highest_fares, reverse=True)


def test_top5_duplicates(tmp_path):
    rows = [[i, 25.0, 1, 1, 100.0] for i in range(7)]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert len(s.highest_fares) == 5
    assert all(f == 100.0 for f in s.highest_fares)


def test_top5_with_nan_and_strings(tmp_path):
    rows = [
        [1, 25.0, 1, 1, 50.0],
        [2, 25.0, 1, 1, None],
        [3, 25.0, 1, 1, "invalid"],
        [4, 25.0, 1, 1, 200.0],
        [5, 25.0, 1, 1, 150.0],
        [6, 25.0, 1, 1, 80.0],
    ]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.highest_fares == [200.0, 150.0, 80.0, 50.0]


def test_pclass_keys_are_int(tmp_path):
    rows = [[i, 20.0 + i * 10, c, 1, 10.0] for i, c in enumerate([1, 2, 3])]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert all(isinstance(k, int) for k in s.mean_age_by_pclass)
    assert all(isinstance(k, int) for k in s.survival_rate_by_pclass)


def test_mean_age_by_pclass(tmp_path):
    rows = [
        [1, 20.0, 1, 1, 10.0],
        [2, 40.0, 1, 0, 10.0],
        [3, 30.0, 2, 1, 10.0],
        [4, None, 2, 0, 10.0], 
        [5, 50.0, 3, 1, 10.0],
    ]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.mean_age_by_pclass[1] == pytest.approx(30.0)
    assert s.mean_age_by_pclass[2] == pytest.approx(30.0)
    assert s.mean_age_by_pclass[3] == pytest.approx(50.0)


def test_survival_rate(tmp_path):
    rows = [
        [1, 20.0, 1, 1, 10.0],
        [2, 25.0, 1, 0, 10.0],
        [3, 30.0, 2, 1, 10.0],
        [4, 35.0, 2, 1, 10.0],
        [5, 40.0, 3, 0, 10.0],
        [6, 45.0, 3, 0, 10.0],
        [7, 50.0, 3, 1, 10.0],
    ]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.survival_rate_by_pclass[1] == pytest.approx(0.5)
    assert s.survival_rate_by_pclass[2] == pytest.approx(1.0)
    assert s.survival_rate_by_pclass[3] == pytest.approx(1 / 3)


def test_missing_by_column(tmp_path):
    rows = [
        [1, 20.0, 1, 1, 10.0],
        [2, None, 2, 0, None],
        [3, 30.0, None, 1, 15.0],
    ]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.missing_by_column["Age"] == 1
    assert s.missing_by_column["Fare"] == 1
    assert s.missing_by_column["Pclass"] == 1
    assert s.missing_by_column["PassengerId"] == 0


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"X": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        analyze_titanic(TitanicInput(csv_path=str(path)))


def test_highest_fares_sorted_descending(tmp_path):
    rows = [[i, 25.0, 1, 1, float(v)] for i, v in enumerate([3, 100, 50, 200, 10, 80])]
    s = analyze_titanic(write_csv(tmp_path, rows, COLS))
    assert s.highest_fares == [200.0, 100.0, 80.0, 50.0, 10.0]


def test_real_titanic_smoke(tmp_path):
    import os
    real_path = os.path.join(os.path.dirname(__file__), "titanic.csv")
    if not os.path.exists(real_path):
        pytest.skip("titanic.csv не найден")
    s = analyze_titanic(TitanicInput(csv_path=real_path))
    assert s.row_count == 891
    assert s.adults_over_30_count > 0
    assert set(s.mean_age_by_pclass.keys()) == {1, 2, 3}
    assert len(s.highest_fares) == 5
    assert s.highest_fares == sorted(s.highest_fares, reverse=True)
