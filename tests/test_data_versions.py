import pandas as pd
import pytest

from scripts.synthetic_data import make_synthetic
from src.make_data_version import slice_by_end_date
from src.validate_data import validate


@pytest.fixture(scope="module")
def full():
    return make_synthetic()


def test_versions_are_nested_and_growing(full):
    v1 = slice_by_end_date(full, "2011-12-31 23:59:59")
    v2 = slice_by_end_date(full, "2012-06-30 23:59:59")
    v3 = slice_by_end_date(full, None)

    assert 0 < len(v1) < len(v2) < len(v3) == len(full)
    # each version is an exact prefix of the next (history is never rewritten)
    assert v1.equals(v2.iloc[: len(v1)])
    assert v2.equals(v3.iloc[: len(v2)])


def test_v1_only_contains_2011(full):
    v1 = slice_by_end_date(full, "2011-12-31 23:59:59")
    assert pd.to_datetime(v1["datetime"]).dt.year.unique().tolist() == [2011]


def test_validate_returns_stats(full):
    stats = validate(full)
    assert stats["rows"] == len(full)
    assert stats["date_start"].startswith("2011-01-01")


def test_validate_rejects_missing_column(full):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(full.drop(columns=["temp"]))


def test_validate_rejects_nulls_and_leakage_mismatch(full):
    bad = full.copy()
    bad.loc[0, "temp"] = None
    bad.loc[1, "count"] = bad.loc[1, "count"] + 5
    with pytest.raises(ValueError) as err:
        validate(bad)
    assert "null" in str(err.value)
    assert "casual + registered" in str(err.value)


def test_validate_rejects_duplicate_timestamps(full):
    bad = pd.concat([full, full.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        validate(bad)
