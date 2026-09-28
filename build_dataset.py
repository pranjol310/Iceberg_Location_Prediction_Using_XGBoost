
import glob
import sys
import numpy as np
import pandas as pd

# ---- NAME CONFIG:
ERA5_TIME, ERA5_LAT, ERA5_LON = "valid_time", "latitude", "longitude"
GLO_TIME, GLO_LAT, GLO_LON = "time", "latitude", "longitude"
ERA5_U, ERA5_V, ERA5_SST = "u10", "v10", "sst"
GLO_U, GLO_V, GLO_ICE = "uo", "vo", "siconc"


def speed_dir(u, v):
    return np.sqrt(u ** 2 + v ** 2), np.degrees(np.arctan2(u, v)) % 360.0  # direction flow heads TOWARD


def assemble_output(tracks, w_u, w_v, sst_k, c_u, c_v, ice):
    """Turns the looked-up arrays into the final table (pure pandas/numpy, no xarray)."""
    ws, wd = speed_dir(w_u, w_v)
    cs, cd = speed_dir(c_u, c_v)

    out = tracks.drop(columns=[c for c in ["day"] if c in tracks.columns]).copy()
    out["current_speed"], out["current_direction"] = cs, cd
    out["wind_speed"], out["wind_direction"] = ws, wd
    out["temperature"] = sst_k - 273.15

    # Where currents and temperature exist (i.e. it is ocean) but sea ice is missing,
    # the ocean model just has no ice there -> that means 0% ice, not "no data".
    ocean_ok = ~(np.isnan(cs) | np.isnan(sst_k))
    ice = np.where(np.isnan(ice) & ocean_ok, 0.0, ice)
    out["sea_ice_concentration"] = np.clip(ice, 0.0, 1.0)

    env_cols = ["current_speed", "current_direction", "wind_speed", "wind_direction",
                "temperature", "sea_ice_concentration"]
    before = len(out)
    out = out.dropna(subset=env_cols).reset_index(drop=True)   # velocity columns may stay NaN
    print(f"Dropped {before - len(out)} rows with missing environmental data "
          f"(land / outside grid); kept {len(out)}")
    return out


def _open_all(pattern, lat_name, time_name):
    import xarray as xr
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?") else [pattern]
    if not files:
        raise SystemExit(f"No files match: {pattern}")
    res = []
    for f in files:
        ds = xr.open_dataset(f).sortby(lat_name)
        t = ds[time_name].values
        res.append((t.min(), t.max(), ds))
    return res


def _pick(datasets, day):
    pad = np.timedelta64(1, "D")
    for tmin, tmax, ds in datasets:
        if tmin - pad <= day <= tmax + pad:
            return ds
    return None


def build(tracks_csv, era5_path, glorys_path, out_csv):
    import xarray as xr
    tracks = pd.read_csv(tracks_csv)
    tracks["timestamp"] = pd.to_datetime(tracks["timestamp"])
    tracks["day"] = tracks["timestamp"].dt.floor("D")
    n = len(tracks)

    era5 = _open_all(era5_path, ERA5_LAT, ERA5_TIME)
    glo = _open_all(glorys_path, GLO_LAT, GLO_TIME)

    cols = {k: np.full(n, np.nan) for k in ["w_u", "w_v", "sst", "c_u", "c_v", "ice"]}
    done = 0
    for day, g in tracks.groupby("day"):
        day64 = np.datetime64(day)
        idx = g.index.values
        lat = xr.DataArray(g["latitude"].values, dims="p")
        lon = xr.DataArray(g["longitude"].values, dims="p")

        e = _pick(era5, day64)
        if e is not None:
            sl = e[[ERA5_U, ERA5_V, ERA5_SST]].sel({ERA5_TIME: day64}, method="nearest").load()
            pts = sl.sel({ERA5_LAT: lat, ERA5_LON: lon}, method="nearest")
            cols["w_u"][idx] = pts[ERA5_U].values
            cols["w_v"][idx] = pts[ERA5_V].values
            cols["sst"][idx] = pts[ERA5_SST].values

        o = _pick(glo, day64)
        if o is not None:
            sl = o[[GLO_U, GLO_V, GLO_ICE]].sel({GLO_TIME: day64}, method="nearest")
            for dim in ("depth", "level"):
                if dim in sl.dims:
                    sl = sl.isel({dim: 0})
            sl = sl.load()
            pts = sl.sel({GLO_LAT: lat, GLO_LON: lon}, method="nearest")
            cols["c_u"][idx] = pts[GLO_U].values
            cols["c_v"][idx] = pts[GLO_V].values
            cols["ice"][idx] = pts[GLO_ICE].values

        done += 1
        if done % 30 == 0:
            print(f"  processed {done} days...")

    out = assemble_output(tracks, cols["w_u"], cols["w_v"], cols["sst"],
                          cols["c_u"], cols["c_v"], cols["ice"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    build(*sys.argv[1:5])
