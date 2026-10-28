import pandas as pd
from src.prepare_data import prepare_features


def sample_data():
    return pd.DataFrame({
        "datetime": ["2011-01-01 00:00:00", "2011-01-01 01:00:00"],
        "season": [1, 1],
        "holiday": [0, 0],
        "workingday": [0, 0],
        "weather": [1, 1],
        "temp": [9.84, 9.02],
        "atemp": [14.395, 13.635],
        "humidity": [81, 80],
        "windspeed": [0.0, 0.0],
        "casual": [3, 8],
        "registered": [13, 32],
        "count": [16, 40],
    })


def test_feature_engineering():
    X, y = prepare_features(sample_data())

    expected = {
        "season", "holiday", "workingday", "weather", "temp", "atemp",
        "humidity", "windspeed", "hour", "day", "month", "year", "dayofweek",
    }

    assert set(X.columns) == expected
    assert len(X) == 2
    assert len(y) == 2


def test_no_target_leakage():
    X, y = prepare_features(sample_data())

    assert "count" not in X.columns
    assert "casual" not in X.columns
    assert "registered" not in X.columns
    assert y.tolist() == [16, 40]


def test_datetime_features():
    X, _ = prepare_features(sample_data())

    assert X["hour"].tolist() == [0, 1]
    assert X["day"].tolist() == [1, 1]
    assert X["month"].tolist() == [1, 1]
    assert X["year"].tolist() == [2011, 2011]
    assert X["dayofweek"].tolist() == [5, 5]
