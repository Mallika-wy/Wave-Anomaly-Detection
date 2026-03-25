from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .cache import cache_path_for_year, compute_stats, write_train_ready_cache
from .config import load_config, resolve_path
from .utils import save_json, seed_everything


DOMAIN = {
    "latitude": (0.0, 60.0),
    "longitude": (100.0, 180.0),
}


def _lower_map(names: list[str]) -> dict[str, str]:
    return {name.lower(): name for name in names}


def pick_name(dataset: xr.Dataset, candidates: list[str], kind: str) -> str:
    all_names = list(dataset.variables)
    lookup = _lower_map(all_names)
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
        lowered = candidate.lower()
        if lowered in lookup:
            return lookup[lowered]
    raise KeyError(f"Unable to find {kind}. Candidates={candidates}, available={all_names}")


def canonicalize_dataset(ds: xr.Dataset, config: dict[str, Any]) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    time_name = pick_name(ds, config["io"]["time_candidates"], "time coordinate")
    lat_name = pick_name(ds, config["io"]["latitude_candidates"], "latitude coordinate")
    lon_name = pick_name(ds, config["io"]["longitude_candidates"], "longitude coordinate")
    rename_map[time_name] = "time"
    rename_map[lat_name] = "latitude"
    rename_map[lon_name] = "longitude"
    ds = ds.rename(rename_map)
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.sel(
        latitude=slice(*DOMAIN["latitude"]),
        longitude=slice(*DOMAIN["longitude"]),
    )
    return ds


def sanitize_array(arr: xr.DataArray) -> xr.DataArray:
    fill_value = arr.attrs.get("_FillValue")
    if fill_value is not None:
        arr = arr.where(arr != fill_value)
    missing_value = arr.attrs.get("missing_value")
    if missing_value is not None:
        arr = arr.where(arr != missing_value)
    return arr.astype(np.float32)


def fill_missing(arr: xr.DataArray, channel_dim: str) -> xr.DataArray:
    arr = arr.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
    spatial_mean = arr.mean(dim=("latitude", "longitude"), skipna=True)
    arr = arr.fillna(spatial_mean)
    global_mean = arr.mean(dim=("time", "latitude", "longitude"), skipna=True)
    arr = arr.fillna(global_mean)
    return arr.fillna(0.0).transpose("time", channel_dim, "latitude", "longitude")


def _extract_variable(ds: xr.Dataset, candidates: list[str], kind: str) -> xr.DataArray:
    name = pick_name(ds, candidates, kind)
    arr = sanitize_array(ds[name])
    return arr.transpose("time", "latitude", "longitude")


def build_month_dataset(month_dir: Path, config: dict[str, Any]) -> xr.Dataset:
    print(2)
    wind_path = month_dir / "data_stream-oper_stepType-instant.nc"
    wave_path = month_dir / "data_stream-wave_stepType-instant.nc"
    if not wind_path.exists() or not wave_path.exists():
        raise FileNotFoundError(f"Missing monthly pair under {month_dir}")

    with xr.open_dataset(wind_path) as wind_ds_raw, xr.open_dataset(wave_path) as wave_ds_raw:
        wind_ds = canonicalize_dataset(wind_ds_raw, config)
        wave_ds = canonicalize_dataset(wave_ds_raw, config)

        u10 = _extract_variable(wind_ds, config["io"]["wind_u_candidates"], "wind u")
        v10 = _extract_variable(wind_ds, config["io"]["wind_v_candidates"], "wind v")

        wind_channels = [u10, v10]
        wind_names = ["u10", "v10"]
        if config["features"]["use_wind_speed"]:
            wind_speed = np.sqrt((u10**2) + (v10**2)).astype(np.float32)
            wind_channels.append(wind_speed)
            wind_names.append("ws")

        wave_vars = xr.Dataset(
            {
                "mwd": _extract_variable(wave_ds, config["io"]["wave_mwd_candidates"], "wave mwd"),
                "mwp": _extract_variable(wave_ds, config["io"]["wave_mwp_candidates"], "wave mwp"),
                "swh": _extract_variable(wave_ds, config["io"]["wave_swh_candidates"], "wave swh"),
            }
        )
        wave_vars = wave_vars.interp(
            time=wind_ds["time"],
            latitude=wind_ds["latitude"],
            longitude=wind_ds["longitude"],
            method=config["features"]["wave_interp_method"],
        )

        wave_channels: list[xr.DataArray] = []
        wave_names: list[str] = []
        if config["features"]["use_mwd_sincos"]:
            angle = np.deg2rad(wave_vars["mwd"] % 360.0)
            wave_channels.extend([np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)])
            wave_names.extend(["mwd_sin", "mwd_cos"])
        else:
            wave_channels.append(wave_vars["mwd"].astype(np.float32))
            wave_names.append("mwd")
        wave_channels.extend([wave_vars["mwp"].astype(np.float32), wave_vars["swh"].astype(np.float32)])
        wave_names.extend(["mwp", "swh"])

        wind = xr.concat(wind_channels, dim="wind_channel").assign_coords(wind_channel=wind_names)
        wave = xr.concat(wave_channels, dim="wave_channel").assign_coords(wave_channel=wave_names)
        quality_mask = (
            np.isfinite(wind).all(dim="wind_channel") & np.isfinite(wave).all(dim="wave_channel")
        ).astype(np.int8)

        wind = fill_missing(wind, "wind_channel")
        wave = fill_missing(wave, "wave_channel")

        ds = xr.Dataset(
            {
                "wind": wind.astype(np.float32),
                "wave": wave.astype(np.float32),
                "quality_mask": quality_mask.transpose("time", "latitude", "longitude"),
            }
        )
        ds = ds.assign_coords(
            time=wind_ds["time"],
            latitude=wind_ds["latitude"],
            longitude=wind_ds["longitude"],
        )
        return ds


def load_label_dataset(
    year: int,
    time_coord: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    config: dict[str, Any],
) -> xr.DataArray:
    label_dir = resolve_path(config["data"]["label_dir"])
    label_path = label_dir / config["data"]["label_filename_template"].format(year=year)
    if not label_path.exists():
        warnings.warn(f"Label file missing for {year}: {label_path}")
        shape = (time_coord.size, latitude.size, longitude.size)
        return xr.DataArray(
            np.full(shape, -1, dtype=np.int8),
            coords={"time": time_coord, "latitude": latitude, "longitude": longitude},
            dims=("time", "latitude", "longitude"),
            name="label",
        )

    with xr.open_dataset(label_path) as label_ds_raw:
        label_ds = canonicalize_dataset(label_ds_raw, config)
        label_name = pick_name(label_ds, config["io"]["label_candidates"], "label variable")
        label = sanitize_array(label_ds[label_name]).interp(
            time=time_coord,
            latitude=latitude,
            longitude=longitude,
            method="nearest",
        )
    label = xr.where(np.isfinite(label), label, -1)
    label = xr.where(label > 0.5, 1, xr.where(label < 0, -1, 0)).astype(np.int8)
    label.name = "label"
    label.attrs["source_path"] = str(label_path)
    return label


def iter_year_months(data_root: Path, year: int) -> list[Path]:
    year_dir = data_root / str(year)
    if not year_dir.exists():
        return []
    return sorted(path for path in year_dir.iterdir() if path.is_dir())


def preprocess_year(year: int, config: dict[str, Any]) -> Path:
    print(1)
    data_root = resolve_path(config["data"]["root_dir"])
    month_dirs = iter_year_months(data_root, year)
    if not month_dirs:
        raise FileNotFoundError(f"No monthly folders found for year {year}")

    monthly = [build_month_dataset(month_dir, config) for month_dir in month_dirs]
    year_ds = xr.concat(monthly, dim="time").sortby("time")
    year_ds = year_ds.assign(
        label=load_label_dataset(
            year,
            year_ds["time"],
            year_ds["latitude"],
            year_ds["longitude"],
            config,
        )
    )
    return write_train_ready_cache(cache_path_for_year(year, config), year_ds, year, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw wind/wave NetCDF files into yearly caches.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    years = args.years
    if not years:
        years = sorted(
            set(config["data"]["train_years"] + config["data"]["test_years"] + config["data"]["val_years"])
        )

    cache_paths: list[Path] = []
    for year in years:
        cache_path = preprocess_year(year, config)
        cache_paths.append(cache_path)
        print(f"[preprocess] saved {cache_path}")

    train_years = set(config["data"]["train_years"])
    stats_sources = [path for path in cache_paths if int(path.stem.split("_")[-1]) in train_years]
    if not stats_sources:
        cache_dir = resolve_path(config["data"]["cache_dir"])
        stats_sources = [
            cache_dir / config["data"]["cache_filename_template"].format(year=year)
            for year in config["data"]["train_years"]
            if (cache_dir / config["data"]["cache_filename_template"].format(year=year)).exists()
        ]
    stats = compute_stats(stats_sources)
    save_json(resolve_path(config["data"]["stats_path"]), stats)
    print(f"[preprocess] saved stats -> {resolve_path(config['data']['stats_path'])}")


if __name__ == "__main__":
    main()
