"""
Train the next-day iceberg position model (XGBoost).

The model predicts the DISPLACEMENT over the next HORIZON days (dx east, dy north,
in km). predict.py adds that displacement to the current position.

Usage:
    python train.py                      # uses training_data.csv
    python train.py path/to/data.csv

Outputs (in ./model/):
    model_dx.json, model_dy.json, meta.json (feature list + horizon)

predict.py imports build_dataset / to_latlon from this file, so training and
prediction always build features the same way.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

CSV_PATH = "training_data.csv"
MODEL_DIR = Path("model")
HORIZON = 1                 # days ahead to predict
KM_PER_DEG = 111.32
SPLIT = "time"              # "time"  -> train on earlier dates, test on later dates
                            # "group" -> hold out whole icebergs
TEST_FRACTION = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def to_km(lat, lon, lat_next, lon_next):
    """Displacement between two lat/lon points in km (dx east, dy north)."""
    dy = (lat_next - lat) * KM_PER_DEG
    dlon = (lon_next - lon + 180) % 360 - 180          # handles the +-180 wrap
    dx = dlon * KM_PER_DEG * np.cos(np.radians((lat + lat_next) / 2))
    return dx, dy


def to_latlon(lat, lon, dx, dy):
    """Inverse of to_km: apply a km displacement to a lat/lon position."""
    lat_new = lat + dy / KM_PER_DEG
    lon_new = lon + dx / (KM_PER_DEG * np.cos(np.radians(lat)))
    lon_new = (lon_new + 180) % 360 - 180
    return lat_new, lon_new


def add_vector_components(df, speed_col, dir_col, prefix):
    """(speed, direction in degrees) -> east/north components. Angles are circular
    (359 deg ~ 1 deg), so trees need u/v rather than raw angles."""
    rad = np.radians(df[dir_col])
    df[f"{prefix}_u"] = df[speed_col] * np.sin(rad)      # east
    df[f"{prefix}_v"] = df[speed_col] * np.cos(rad)      # north
    return df


# ---------------------------------------------------------------------------
# Data preparation (shared with predict.py)
# ---------------------------------------------------------------------------
def build_dataset(csv_path=CSV_PATH, horizon=HORIZON):
    if isinstance(csv_path, pd.DataFrame):               # already-loaded data (used by predict.py)
        df = csv_path.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("iceberg_id", sort=False)

    # ---- targets: displacement `horizon` days ahead -----------------------
    lat_next = g["latitude"].shift(-horizon)
    lon_next = g["longitude"].shift(-horizon)
    date_next = g["timestamp"].shift(-horizon)
    df["target_dx_km"], df["target_dy_km"] = to_km(
        df["latitude"], df["longitude"], lat_next, lon_next
    )
    # Keep only rows whose target really is `horizon` days later
    # (a few tracks have 2-7 day gaps; those would be mislabeled).
    good_target = (date_next - df["timestamp"]).dt.days == horizon
    df.loc[~good_target, ["target_dx_km", "target_dy_km"]] = np.nan

    # ---- features (past information only, so nothing leaks) ---------------
    df["lon_sin"] = np.sin(np.radians(df["longitude"]))
    df["lon_cos"] = np.cos(np.radians(df["longitude"]))
    df["doy"] = df["timestamp"].dt.dayofyear

    # ~66% of size_km2 values are exactly 0 (probably "unknown"): keep a flag
    df["size_known"] = (df["size_km2"] > 0).astype(int)
    df["log_size"] = np.log1p(df["size_km2"])

    df = add_vector_components(df, "current_speed", "current_direction", "cur")
    df = add_vector_components(df, "wind_speed", "wind_direction", "wind")

    df["prev1_speed"] = np.hypot(df["prev1_dx_km"], df["prev1_dy_km"])
    df["prev3_speed"] = np.hypot(df["prev3_dx_km"], df["prev3_dy_km"])
    g = df.groupby("iceberg_id", sort=False)
    df["prev2_dx_km"] = g["prev1_dx_km"].shift(1)        # displacement 2 days ago
    df["prev2_dy_km"] = g["prev1_dy_km"].shift(1)
    for col in ["prev1_dx_km", "prev1_dy_km"]:            # 7-day mean motion
        df[col.replace("prev1", "mean7")] = (
            g[col].transform(lambda s: s.rolling(7, min_periods=3).mean())
        )
    for col in ["cur_u", "cur_v", "wind_u", "wind_v"]:    # 1-day change in environment
        df[f"{col}_d1"] = g[col].diff()

    feature_cols = [
        "latitude", "longitude", "lon_sin", "lon_cos", "doy",
        "log_size", "size_known", "mask",
        "prev1_dx_km", "prev1_dy_km", "prev2_dx_km", "prev2_dy_km",
        "prev3_dx_km", "prev3_dy_km", "prev1_speed", "prev3_speed",
        "mean7_dx_km", "mean7_dy_km",
        "current_speed", "cur_u", "cur_v", "wind_speed", "wind_u", "wind_v",
        "cur_u_d1", "cur_v_d1", "wind_u_d1", "wind_v_d1",
        "temperature", "sea_ice_concentration",
    ]
    return df, feature_cols


# ---------------------------------------------------------------------------
# Split (never random rows: consecutive days of one iceberg are near-identical)
# ---------------------------------------------------------------------------
def split_data(df):
    labeled = df.dropna(subset=["target_dx_km", "target_dy_km"]).copy()

    if SPLIT == "time":
        cutoff = labeled["timestamp"].quantile(1 - TEST_FRACTION)
        train = labeled[labeled["timestamp"] < cutoff]
        test = labeled[labeled["timestamp"] >= cutoff]
    else:
        rng = np.random.default_rng(SEED)
        ids = labeled["iceberg_id"].unique()
        test_ids = rng.choice(ids, size=int(len(ids) * TEST_FRACTION), replace=False)
        test = labeled[labeled["iceberg_id"].isin(test_ids)]
        train = labeled[~labeled["iceberg_id"].isin(test_ids)]

    # last 15% of the training period (by time) is used for early stopping
    val_cut = train["timestamp"].quantile(0.85)
    fit = train[train["timestamp"] < val_cut]
    val = train[train["timestamp"] >= val_cut]
    return fit, val, test


# ---------------------------------------------------------------------------
# Training / evaluation / saving
# ---------------------------------------------------------------------------
def make_model():
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="reg:squarederror",   # try "reg:pseudohubererror" if outliers dominate
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )


def train_models(fit, val, feature_cols):
    models = {}
    for target in ["target_dx_km", "target_dy_km"]:
        model = make_model()
        model.fit(
            fit[feature_cols], fit[target],
            eval_set=[(val[feature_cols], val[target])],
            verbose=False,
        )
        models[target] = model
        print(f"{target}: best iteration = {model.best_iteration}")
    return models


def evaluate(models, test, feature_cols):
    """Next-day position error in km vs. two simple baselines."""
    true_dx, true_dy = test["target_dx_km"].values, test["target_dy_km"].values
    dx = models["target_dx_km"].predict(test[feature_cols])
    dy = models["target_dy_km"].predict(test[feature_cols])

    def err(px, py):
        return np.hypot(px - true_dx, py - true_dy)

    zero = err(0 * true_dx, 0 * true_dy)                                    # "stays put"
    persist = err(test["prev1_dx_km"].fillna(0).values,
                  test["prev1_dy_km"].fillna(0).values)                     # "same as yesterday"
    xgb = err(dx, dy)

    print("\nMean / median next-day position error (km):")
    print(f"  stays put         : {zero.mean():6.2f} / {np.median(zero):6.2f}")
    print(f"  same as yesterday : {persist.mean():6.2f} / {np.median(persist):6.2f}")
    print(f"  XGBoost           : {xgb.mean():6.2f} / {np.median(xgb):6.2f}")


def save_models(models, feature_cols, horizon=HORIZON):
    MODEL_DIR.mkdir(exist_ok=True)
    models["target_dx_km"].save_model(MODEL_DIR / "model_dx.json")
    models["target_dy_km"].save_model(MODEL_DIR / "model_dy.json")
    (MODEL_DIR / "meta.json").write_text(
        json.dumps({"feature_cols": feature_cols, "horizon": horizon}, indent=2)
    )
    print(f"\nSaved model to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df, feature_cols = build_dataset(csv)
    fit, val, test = split_data(df)
    print(f"train={len(fit)}  val={len(val)}  test={len(test)}  features={len(feature_cols)}")

    # 1) train on the train split and report honest test-set error
    models = train_models(fit, val, feature_cols)
    evaluate(models, test, feature_cols)

    imp = pd.Series(models["target_dx_km"].feature_importances_, index=feature_cols)
    print("\nTop features (dx model):")
    print(imp.sort_values(ascending=False).head(10).round(3))

    # 2) save the model that was validated above
    save_models(models, feature_cols)
