import pandas as pd


def test_training_data_exists():
    X = pd.read_csv("data/processed/X_train.csv")
    y = pd.read_csv("data/processed/y_train.csv")

    assert len(X) > 0
    assert len(y) > 0
    assert len(X) == len(y)


def test_no_target_leakage():
    X = pd.read_csv("data/processed/X_train.csv")

    assert "count" not in X.columns
    assert "casual" not in X.columns
    assert "registered" not in X.columns


def test_expected_features_exist():
    X = pd.read_csv("data/processed/X_train.csv")

    expected = {
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "hour",
        "day",
        "month",
        "year",
        "dayofweek",
    }

    assert expected.issubset(set(X.columns))
