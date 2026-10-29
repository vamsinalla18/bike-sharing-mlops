from src.common import load_params
from src.prepare_data import prepare_features, split_chronological
from src.train import build_models
from scripts.synthetic_data import make_synthetic


def test_models_built_from_params():
    params = load_params()
    models = build_models(params["train"])
    assert set(models) == {"LinearRegression", "RandomForest", "XGBoost"}
    assert models["RandomForest"].n_estimators == params["train"]["models"]["RandomForest"]["n_estimators"]


def test_split_is_chronological_and_complete():
    X, y = prepare_features(make_synthetic("2011-01-01", "2011-03-31"))
    X_tr, X_va, y_tr, y_va = split_chronological(X, y, 0.8)
    assert len(X_tr) + len(X_va) == len(X)
    assert len(y_tr) == len(X_tr) and len(y_va) == len(X_va)
    # validation comes strictly after training in time
    assert (X_tr["month"].max(), X_tr["day"].max()) <= (X_va["month"].max(), X_va["day"].max())
