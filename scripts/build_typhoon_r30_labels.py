#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


EARTH_RADIUS_KM = 6371.0
NMILE_TO_KM = 1.852


@dataclass(frozen=True)
class StormPoint:
    time_aligned: pd.Timestamp
    lat: float
    lon: float
    radius_nm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build typhoon-affected masks from IBTrACS r30 for model grids."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--ibtracs-file",
        type=Path,
        default=Path("data/IBTrACS.WP.v04r01.nc"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("labels_r30"))
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--grid",
        choices=("oper", "wave"),
        default="oper",
        help="Target grid: oper(0.25deg) or wave(0.5deg).",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="",
        help="Optional comma-separated months for quick runs, e.g. 1,2,12",
    )
    parser.add_argument(
        "--radius-source",
        choices=("tokyo_kma_max", "tokyo_only", "kma_only"),
        default="tokyo_kma_max",
        help="How to build r30 radius from IBTrACS variables.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    return parser.parse_args()


def normalize_lon_360(lon: np.ndarray | float) -> np.ndarray | float:
    return np.mod(lon, 360.0)


def month_filter(months_arg: str) -> set[int] | None:
    if not months_arg.strip():
        return None
    months = {int(x.strip()) for x in months_arg.split(",") if x.strip()}
    invalid = [m for m in months if m < 1 or m > 12]
    if invalid:
        raise ValueError(f"Invalid months: {sorted(invalid)}")
    return months


def nanmax_preserve_nan(arr: np.ndarray, axis: int) -> np.ndarray:
    all_nan = np.all(np.isnan(arr), axis=axis)
    out = np.nanmax(np.where(np.isnan(arr), -np.inf, arr), axis=axis)
    out = out.astype(np.float64, copy=False)
    out[all_nan] = np.nan
    return out


def collect_r30_radius(ds: xr.Dataset, source: str) -> np.ndarray:
    tokyo_stack = np.stack(
        [
            ds["tokyo_r30_long"].values,
            ds["tokyo_r30_short"].values,
        ],
        axis=0,
    )
    tokyo_r30 = nanmax_preserve_nan(tokyo_stack, axis=0)

    kma_stack = np.stack(
        [
            ds["kma_r30_long"].values,
            ds["kma_r30_short"].values,
        ],
        axis=0,
    )
    kma_r30 = nanmax_preserve_nan(kma_stack, axis=0)

    if source == "tokyo_only":
        return tokyo_r30
    if source == "kma_only":
        return kma_r30

    both = np.stack([tokyo_r30, kma_r30], axis=0)
    return nanmax_preserve_nan(both, axis=0)


def load_storm_points(
    ibtracs_file: Path,
    start_year: int,
    end_year: int,
    radius_source: str,
) -> pd.DataFrame:
    with xr.open_dataset(ibtracs_file) as ds:
        iso = ds["iso_time"].values.astype("U19")
        time_flat = pd.to_datetime(iso.ravel(), errors="coerce")
        lat_flat = ds["lat"].values.ravel()
        lon_flat = normalize_lon_360(ds["lon"].values.ravel())
        radius_nm_flat = collect_r30_radius(ds, radius_source).ravel()

    start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    end = pd.Timestamp(f"{end_year + 1}-01-01 00:00:00")

    valid = (
        (~time_flat.isna())
        & np.isfinite(lat_flat)
        & np.isfinite(lon_flat)
        & np.isfinite(radius_nm_flat)
        & (radius_nm_flat > 0.0)
        & (lat_flat >= 0.0)
        & (lat_flat <= 60.0)
        & (lon_flat >= 100.0)
        & (lon_flat <= 180.0)
        & (time_flat >= start)
        & (time_flat < end)
    )

    df = pd.DataFrame(
        {
            "time": time_flat[valid],
            "lat": lat_flat[valid].astype(np.float64),
            "lon": lon_flat[valid].astype(np.float64),
            "radius_nm": radius_nm_flat[valid].astype(np.float64),
        }
    )
    # Model files are 3-hourly. Round IBTrACS timestamps to nearest 3-hour step.
    df["time_aligned"] = df["time"].dt.round("3h")
    df = df.sort_values("time_aligned").reset_index(drop=True)
    return df


def haversine_km(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def file_for_grid(month_dir: Path, grid: str) -> Path:
    if grid == "oper":
        return month_dir / "data_stream-oper_stepType-instant.nc"
    if grid == "wave":
        return month_dir / "data_stream-wave_stepType-instant.nc"
    raise ValueError(f"Unsupported grid: {grid}")


def iter_month_files(
    data_root: Path,
    years: Iterable[int],
    months_keep: set[int] | None,
    grid: str,
) -> Iterable[tuple[int, int, Path]]:
    for year in years:
        year_dir = data_root / str(year)
        if not year_dir.exists():
            continue
        for month_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            ym = month_dir.name
            if len(ym) != 6 or not ym.isdigit():
                continue
            y = int(ym[:4])
            m = int(ym[4:6])
            if y != year:
                continue
            if months_keep is not None and m not in months_keep:
                continue
            nc_file = file_for_grid(month_dir, grid)
            if nc_file.exists():
                yield y, m, nc_file


def build_month_mask(
    nc_file: Path,
    month_points: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with xr.open_dataset(nc_file) as ds:
        times = pd.to_datetime(ds["valid_time"].values)
        lats = ds["latitude"].values.astype(np.float64)
        lons = ds["longitude"].values.astype(np.float64)

    mask = np.zeros((times.size, lats.size, lons.size), dtype=np.uint8)
    if month_points.empty:
        return times.values, lats, lons, mask, 0

    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    hit_points = 0

    for row in month_points.itertuples(index=False):
        t_idx = time_to_idx.get(pd.Timestamp(row.time_aligned))
        if t_idx is None:
            continue

        radius_km = float(row.radius_nm) * NMILE_TO_KM
        lat0 = float(row.lat)
        lon0 = float(row.lon)

        lat_delta = radius_km / 111.0
        cos_lat = np.cos(np.deg2rad(lat0))
        cos_lat = max(cos_lat, 1.0e-6)
        lon_delta = radius_km / (111.0 * cos_lat)

        lat_idx = np.where((lats >= lat0 - lat_delta) & (lats <= lat0 + lat_delta))[0]
        lon_idx = np.where((lons >= lon0 - lon_delta) & (lons <= lon0 + lon_delta))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            continue

        sub_lat = lat2d[np.ix_(lat_idx, lon_idx)]
        sub_lon = lon2d[np.ix_(lat_idx, lon_idx)]
        dist = haversine_km(lat0, lon0, sub_lat, sub_lon)
        local_hit = dist <= radius_km
        if not np.any(local_hit):
            continue

        layer = mask[t_idx]
        local_mask = layer[np.ix_(lat_idx, lon_idx)]
        local_mask |= local_hit.astype(np.uint8)
        layer[np.ix_(lat_idx, lon_idx)] = local_mask
        hit_points += 1

    return times.values, lats, lons, mask, hit_points


def save_year_dataset(
    out_file: Path,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    grid: str,
    radius_source: str,
) -> None:
    ds_out = xr.Dataset(
        data_vars={
            "typhoon_affected": (("time", "latitude", "longitude"), data),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "description": "Typhoon impact mask generated from IBTrACS r30.",
            "mask_rule": "1 if grid point is within r30 of any storm center at the same aligned time, else 0.",
            "r30_units": "nautical_mile",
            "radius_source": radius_source,
            "grid_source": grid,
            "time_alignment": "IBTrACS iso_time rounded to nearest 3 hours",
            "domain": "0-60N, 100-180E",
        },
    )
    ds_out["typhoon_affected"].attrs["long_name"] = "Typhoon impacted flag"
    ds_out["typhoon_affected"].attrs["flag_values"] = [0, 1]
    ds_out["typhoon_affected"].attrs["flag_meanings"] = "not_affected affected"

    encoding = {
        "typhoon_affected": {
            "zlib": True,
            "complevel": 4,
            "dtype": "i1",
        }
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_file, encoding=encoding)


def main() -> None:
    args = parse_args()
    months_keep = month_filter(args.months)
    years = list(range(args.start_year, args.end_year + 1))

    storm_df = load_storm_points(
        ibtracs_file=args.ibtracs_file,
        start_year=args.start_year,
        end_year=args.end_year,
        radius_source=args.radius_source,
    )
    print(
        f"[INFO] Loaded {len(storm_df)} valid IBTrACS points "
        f"({args.start_year}-{args.end_year}, source={args.radius_source})."
    )

    month_files = list(iter_month_files(args.data_root, years, months_keep, args.grid))
    if not month_files:
        raise FileNotFoundError("No monthly files found for given arguments.")

    by_year: dict[int, list[tuple[int, Path]]] = {}
    for year, month, nc_file in month_files:
        by_year.setdefault(year, []).append((month, nc_file))

    for year in sorted(by_year):
        out_file = args.output_dir / args.grid / f"typhoon_r30_mask_{args.grid}_{year}.nc"
        if args.skip_existing and out_file.exists():
            print(f"[SKIP] {out_file}")
            continue

        month_items = sorted(by_year[year], key=lambda x: x[0])
        time_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        lat_ref: np.ndarray | None = None
        lon_ref: np.ndarray | None = None
        monthly_hits = 0

        print(f"[INFO] Year {year}: processing {len(month_items)} month files...")
        for month, nc_file in month_items:
            month_start = pd.Timestamp(year=year, month=month, day=1)
            month_end = month_start + pd.offsets.MonthBegin(1)
            month_df = storm_df[
                (storm_df["time_aligned"] >= month_start)
                & (storm_df["time_aligned"] < month_end)
            ]

            times, lats, lons, month_mask, hit_points = build_month_mask(nc_file, month_df)
            if lat_ref is None:
                lat_ref = lats
                lon_ref = lons
            else:
                if not np.array_equal(lat_ref, lats) or not np.array_equal(lon_ref, lons):
                    raise ValueError(f"Grid mismatch in file: {nc_file}")

            time_chunks.append(times)
            mask_chunks.append(month_mask)
            monthly_hits += hit_points
            affected_ratio = float(month_mask.sum()) / float(month_mask.size)
            print(
                f"  [MONTH {month:02d}] points={len(month_df):5d}, "
                f"used={hit_points:5d}, affected_ratio={affected_ratio:.6f}"
            )

        all_times = np.concatenate(time_chunks, axis=0)
        all_masks = np.concatenate(mask_chunks, axis=0)
        save_year_dataset(
            out_file=out_file,
            times=all_times,
            lats=lat_ref,
            lons=lon_ref,
            data=all_masks,
            grid=args.grid,
            radius_source=args.radius_source,
        )
        year_ratio = float(all_masks.sum()) / float(all_masks.size)
        print(
            f"[DONE] {out_file} | shape={all_masks.shape}, "
            f"used_points={monthly_hits}, affected_ratio={year_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
