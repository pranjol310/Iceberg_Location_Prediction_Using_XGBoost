import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from train import MODEL_DIR, build_dataset, to_km, to_latlon  


def load_model(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model_dx, model_dy = XGBRegressor(), XGBRegressor()
    model_dx.load_model(model_dir / "model_dx.json")
    model_dy.load_model(model_dir / "model_dy.json")
    return model_dx, model_dy, meta["feature_cols"], meta["horizon"]


def compute_prev_columns(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("iceberg_id", sort=False)
    df["prev1_dx_km"], df["prev1_dy_km"] = to_km(
        g["latitude"].shift(1), g["longitude"].shift(1), df["latitude"], df["longitude"]
    )
    gap = g["timestamp"].diff().dt.days
    df.loc[gap != 1, ["prev1_dx_km", "prev1_dy_km"]] = float("nan")
    g = df.groupby("iceberg_id", sort=False)
    for axis in ["dx", "dy"]:
        df[f"prev3_{axis}_km"] = g[f"prev1_{axis}_km"].transform(
            lambda s: s.rolling(3, min_periods=3).mean()
        )
    return df


def predict_positions(csv_path, model_dir=MODEL_DIR, iceberg_id=None, raw=False):
    model_dx, model_dy, feature_cols, horizon = load_model(model_dir)

    data = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if raw:
        data = compute_prev_columns(data)
    df, _ = build_dataset(data, horizon=horizon)
    latest = df.groupby("iceberg_id", sort=False).tail(1)     # last known row per iceberg
    if iceberg_id is not None:
        latest = latest[latest["iceberg_id"] == iceberg_id]
        if latest.empty:
            raise ValueError(f"iceberg_id '{iceberg_id}' not found in {csv_path}")

    dx = model_dx.predict(latest[feature_cols])
    dy = model_dy.predict(latest[feature_cols])
    lat_new, lon_new = to_latlon(latest["latitude"].values, latest["longitude"].values, dx, dy)

    return pd.DataFrame({
        "iceberg_id": latest["iceberg_id"].values,
        "last_observed": latest["timestamp"].dt.date.values,
        "predicted_for": (latest["timestamp"] + pd.Timedelta(days=horizon)).dt.date.values,
        "current_lat": latest["latitude"].values,
        "current_lon": latest["longitude"].values,
        "pred_dx_km": dx,
        "pred_dy_km": dy,
        "pred_lat": lat_new,
        "pred_lon": lon_new,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data.csv")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--raw", action="store_true",
                        help="input has no prev1_*/prev3_* columns; compute them from lat/lon")
    parser.add_argument("--iceberg", default=None, help="predict only this iceberg_id")
    parser.add_argument("--output", default=None, help="optional path to save predictions as CSV")
    args = parser.parse_args()

    preds = predict_positions(args.input, args.model_dir, args.iceberg, args.raw)
    pd.set_option("display.width", 200)
    print(preds.round(4).to_string(index=False))
    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
