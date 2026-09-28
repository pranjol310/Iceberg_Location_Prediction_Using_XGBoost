import glob
import os
import sys
import numpy as np
import pandas as pd

MAX_DAILY_JUMP_KM = 60.0   # bigger jumps are tracking errors / breakups, not drift
MAX_GAP_DAYS = 3           # a longer hole in the record starts a new segment
MIN_SEGMENT_DAYS = 5       # drop very short segments
VEL_COLS = ["prev1_dx_km", "prev1_dy_km", "prev3_dx_km", "prev3_dy_km"]


def _disp_km(lat0, lon0, lat1, lon1):
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    dx = dlon * 111.32 * np.cos(np.radians((lat0 + lat1) / 2.0))
    dy = (lat1 - lat0) * 111.32
    return dx, dy


def add_recent_velocity(df):
    """Adds prev1_* and prev3_* columns (NaN where the needed earlier day is missing)."""
    df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    out = {c: np.full(len(df), np.nan) for c in VEL_COLS}
    day = pd.Timedelta(days=1)
    for _, g in df.groupby("iceberg_id"):
        pos = {t: (la, lo) for t, la, lo in zip(g["timestamp"], g["latitude"], g["longitude"])}
        for idx, t, la, lo in zip(g.index, g["timestamp"], g["latitude"], g["longitude"]):
            p1 = pos.get(t - day)
            if p1 is not None:
                dx, dy = _disp_km(p1[0], p1[1], la, lo)
                out["prev1_dx_km"][idx], out["prev1_dy_km"][idx] = dx, dy
            p3 = pos.get(t - 3 * day)
            if p3 is not None:
                dx, dy = _disp_km(p3[0], p3[1], la, lo)
                out["prev3_dx_km"][idx], out["prev3_dy_km"][idx] = dx / 3.0, dy / 3.0
    for c, v in out.items():
        df[c] = v
    return df


def build(folder, out_path, year):
    frames = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        d = pd.read_csv(f)
        d["name"] = os.path.basename(f)[:-4].upper()
        frames.append(d)
    if not frames:
        raise SystemExit(f"No CSV files found in {folder}")
    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["date"].astype(str), format="%Y%j")
    df = df[df["timestamp"].dt.year == year].copy()
    df = df.sort_values(["name", "timestamp"]).reset_index(drop=True)

    gap = df.groupby("name")["timestamp"].diff().dt.days
    jump = df["disp"] > MAX_DAILY_JUMP_KM
    new_seg = (gap.isna()) | (gap > MAX_GAP_DAYS) | jump
    df["seg"] = new_seg.astype(int).groupby(df["name"]).cumsum()
    df["iceberg_id"] = df["name"] + "_" + df["seg"].astype(str)

    df["longitude"] = ((df["lon"] + 180) % 360) - 180
    df["size_km2"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = df.rename(columns={"lat": "latitude"})[
        ["iceberg_id", "timestamp", "latitude", "longitude", "size_km2", "mask"]]
    out = out[(out["latitude"] >= -80) & (out["latitude"] <= -40)]

    counts = out["iceberg_id"].value_counts()
    out = out[out["iceberg_id"].isin(counts[counts >= MIN_SEGMENT_DAYS].index)]
    out = add_recent_velocity(out)

    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows, {out['iceberg_id'].nunique()} segments "
          f"from {df['name'].nunique()} icebergs")
    print(f"Latitude range {out['latitude'].min():.1f} to {out['latitude'].max():.1f}, "
          f"longitude range {out['longitude'].min():.1f} to {out['longitude'].max():.1f}")
    print(f"Rows with size known: {(out['size_km2'] > 0).mean() * 100:.0f}%, "
          f"rows with 3-day history: {out['prev3_dx_km'].notna().mean() * 100:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2019)
