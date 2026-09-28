import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from train import MODEL_DIR, build_dataset, to_latlon  


def load_model(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model_dx, model_dy = XGBRegressor(), XGBRegressor()
    model_dx.load_model(model_dir / "model_dx.json")
    model_dy.load_model(model_dir / "model_dy.json")
    return model_dx, model_dy, meta["feature_cols"], meta["horizon"]


def predict_positions(csv_path, model_dir=MODEL_DIR, iceberg_id=None):
    model_dx, model_dy, feature_cols, horizon = load_model(model_dir)

    df, _ = build_dataset(csv_path, horizon=horizon)
    latest = df.groupby("iceberg_id", sort=False).tail(1)     
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
    parser.add_argument("--iceberg", default=None, help="predict only this iceberg_id")
    parser.add_argument("--output", default=None, help="optional path to save predictions as CSV")
    args = parser.parse_args()

    preds = predict_positions(args.input, args.model_dir, args.iceberg)
    pd.set_option("display.width", 200)
    print(preds.round(4).to_string(index=False))
    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
