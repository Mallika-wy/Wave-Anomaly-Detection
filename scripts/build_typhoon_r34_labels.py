#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


EARTH_RADIUS_KM = 6371.0
NMILE_TO_KM = 1.852


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build typhoon-affected masks from IBTrACS usa_r34 for model grids."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--ibtracs-file",
        type=Path,
        default=Path("data/IBTrACS.WP.v04r01.nc"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("labels_r34"))
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
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
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    parser.add_argument(
        "--soft-boundary-value",
        type=float,
        default=0.5,
        help="Soft-label value at the r34 boundary. Must be in (0, 1).",
    )
    parser.add_argument(
        "--soft-min-value",
        type=float,
        default=0.01,
        help="Minimum soft-label value to keep when truncating the Gaussian tail.",
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


def collect_usa_r34(ds: xr.Dataset) -> np.ndarray:
    if "usa_r34" not in ds:
        raise KeyError("IBTrACS file does not contain usa_r34.")
    radius = ds["usa_r34"].values.astype(np.float64)
    if radius.ndim != 3 or radius.shape[-1] != 4:
        raise ValueError(f"Expected usa_r34 shape (storm, date_time, 4), got {radius.shape}")
    return radius


def load_storm_points(
    ibtracs_file: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    with xr.open_dataset(ibtracs_file) as ds:
        time = pd.to_datetime(ds["time"].values.ravel(), errors="coerce")
        lat_flat = ds["lat"].values.ravel()
        lon_flat = normalize_lon_360(ds["lon"].values.ravel())
        radius_r34 = collect_usa_r34(ds).reshape(-1, 4)

    valid_quadrants = np.isfinite(radius_r34) & (radius_r34 > 0.0)
    has_any_radius = np.any(valid_quadrants, axis=1)

    start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    end = pd.Timestamp(f"{end_year + 1}-01-01 00:00:00")

    valid = (
        (~time.isna())
        & np.isfinite(lat_flat)
        & np.isfinite(lon_flat)
        & has_any_radius
        & (lat_flat >= 0.0)
        & (lat_flat <= 60.0)
        & (lon_flat >= 100.0)
        & (lon_flat <= 180.0)
        & (time >= start)
        & (time < end)
    )

    df = pd.DataFrame(
        {
            "time": time[valid],
            "lat": lat_flat[valid].astype(np.float64),
            "lon": lon_flat[valid].astype(np.float64),
            "radius_ne_nm": radius_r34[valid, 0].astype(np.float64),
            "radius_se_nm": radius_r34[valid, 1].astype(np.float64),
            "radius_sw_nm": radius_r34[valid, 2].astype(np.float64),
            "radius_nw_nm": radius_r34[valid, 3].astype(np.float64),
        }
    )
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
        for month_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
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


def quadrant_radius_km(
    row: pd.Series,
    dlat: np.ndarray,
    dlon: np.ndarray,
) -> np.ndarray:
    radii_km = np.zeros_like(dlat, dtype=np.float64)
    ne = max(float(row.radius_ne_nm), 0.0) * NMILE_TO_KM if np.isfinite(row.radius_ne_nm) else 0.0
    se = max(float(row.radius_se_nm), 0.0) * NMILE_TO_KM if np.isfinite(row.radius_se_nm) else 0.0
    sw = max(float(row.radius_sw_nm), 0.0) * NMILE_TO_KM if np.isfinite(row.radius_sw_nm) else 0.0
    nw = max(float(row.radius_nw_nm), 0.0) * NMILE_TO_KM if np.isfinite(row.radius_nw_nm) else 0.0

    radii_km[(dlat >= 0.0) & (dlon >= 0.0)] = ne
    radii_km[(dlat < 0.0) & (dlon >= 0.0)] = se
    radii_km[(dlat < 0.0) & (dlon < 0.0)] = sw
    radii_km[(dlat >= 0.0) & (dlon < 0.0)] = nw
    return radii_km


def sigma_from_radius_km(radius_km: np.ndarray, boundary_value: float) -> np.ndarray:
    sigma = np.zeros_like(radius_km, dtype=np.float64)
    valid = radius_km > 0.0
    if np.any(valid):
        denom = np.sqrt(-2.0 * np.log(boundary_value))
        sigma[valid] = radius_km[valid] / max(denom, 1.0e-8)
    return sigma


def build_month_mask(
    nc_file: Path,
    month_points: pd.DataFrame,
    soft_boundary_value: float,
    soft_min_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with xr.open_dataset(nc_file) as ds:
        times = pd.to_datetime(ds["valid_time"].values)
        lats = ds["latitude"].values.astype(np.float64)
        lons = ds["longitude"].values.astype(np.float64)

    mask = np.zeros((times.size, lats.size, lons.size), dtype=np.uint8)
    soft_mask = np.zeros((times.size, lats.size, lons.size), dtype=np.float32)
    if month_points.empty:
        return times.values, lats, lons, mask, soft_mask, 0

    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    hit_points = 0

    for row in month_points.itertuples(index=False):
        t_idx = time_to_idx.get(pd.Timestamp(row.time_aligned))
        if t_idx is None:
            continue

        radius_candidates = np.array(
            [row.radius_ne_nm, row.radius_se_nm, row.radius_sw_nm, row.radius_nw_nm],
            dtype=np.float64,
        )
        radius_candidates = radius_candidates[np.isfinite(radius_candidates) & (radius_candidates > 0.0)]
        if radius_candidates.size == 0:
            continue

        max_radius_km = float(radius_candidates.max()) * NMILE_TO_KM
        max_sigma_km = max_radius_km / max(np.sqrt(-2.0 * np.log(soft_boundary_value)), 1.0e-8)
        truncate_radius_km = max(
            max_radius_km,
            max_sigma_km * np.sqrt(-2.0 * np.log(max(soft_min_value, 1.0e-8))),
        )
        lat0 = float(row.lat)
        lon0 = float(row.lon)

        lat_delta = truncate_radius_km / 111.0
        cos_lat = max(np.cos(np.deg2rad(lat0)), 1.0e-6)
        lon_delta = truncate_radius_km / (111.0 * cos_lat)

        lat_idx = np.where((lats >= lat0 - lat_delta) & (lats <= lat0 + lat_delta))[0]
        lon_idx = np.where((lons >= lon0 - lon_delta) & (lons <= lon0 + lon_delta))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            continue

        sub_lat = lat2d[np.ix_(lat_idx, lon_idx)]
        sub_lon = lon2d[np.ix_(lat_idx, lon_idx)]
        dlat = sub_lat - lat0
        dlon = sub_lon - lon0
        dist = haversine_km(lat0, lon0, sub_lat, sub_lon)
        local_radius_km = quadrant_radius_km(row, dlat, dlon)
        local_sigma_km = sigma_from_radius_km(local_radius_km, soft_boundary_value)
        local_hit = (local_radius_km > 0.0) & (dist <= local_radius_km)
        local_soft = np.zeros_like(dist, dtype=np.float32)
        valid_soft = local_sigma_km > 0.0
        if np.any(valid_soft):
            local_soft[valid_soft] = np.exp(
                -(dist[valid_soft] ** 2) / (2.0 * np.square(local_sigma_km[valid_soft]))
            ).astype(np.float32)
            local_soft[local_soft < soft_min_value] = 0.0
        if not np.any(local_hit):
            if not np.any(local_soft > 0.0):
                continue

        layer = mask[t_idx]
        local_mask = layer[np.ix_(lat_idx, lon_idx)]
        local_mask |= local_hit.astype(np.uint8)
        layer[np.ix_(lat_idx, lon_idx)] = local_mask
        soft_layer = soft_mask[t_idx]
        existing_soft = soft_layer[np.ix_(lat_idx, lon_idx)]
        soft_layer[np.ix_(lat_idx, lon_idx)] = np.maximum(existing_soft, local_soft)
        hit_points += 1

    return times.values, lats, lons, mask, soft_mask, hit_points


def save_year_dataset(
    out_file: Path,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    soft_data: np.ndarray,
    grid: str,
    soft_boundary_value: float,
    soft_min_value: float,
) -> None:
    ds_out = xr.Dataset(
        data_vars={
            "typhoon_affected": (("time", "latitude", "longitude"), data),
            "typhoon_affected_soft": (("time", "latitude", "longitude"), soft_data),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "description": "Typhoon impact mask generated from IBTrACS usa_r34.",
            "mask_rule": "1 if grid point falls inside the matching usa_r34 quadrant radius at the aligned time, else 0.",
            "r34_units": "nautical_mile",
            "radius_source": "usa_r34",
            "quadrant_order": "NE,SE,SW,NW",
            "soft_label_formula": "exp(-(d^2)/(2*sigma^2))",
            "soft_boundary_value": float(soft_boundary_value),
            "soft_min_value": float(soft_min_value),
            "grid_source": grid,
            "time_alignment": "IBTrACS time rounded to nearest 3 hours",
            "domain": "0-60N, 100-180E",
        },
    )
    ds_out["typhoon_affected"].attrs["long_name"] = "Typhoon impacted flag"
    ds_out["typhoon_affected"].attrs["flag_values"] = [0, 1]
    ds_out["typhoon_affected"].attrs["flag_meanings"] = "not_affected affected"
    ds_out["typhoon_affected_soft"].attrs["long_name"] = "Soft typhoon impacted score"
    ds_out["typhoon_affected_soft"].attrs["valid_range"] = [0.0, 1.0]
    ds_out["typhoon_affected_soft"].attrs["boundary_value"] = float(soft_boundary_value)

    encoding = {
        "typhoon_affected": {
            "zlib": True,
            "complevel": 4,
            "dtype": "i1",
        },
        "typhoon_affected_soft": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        },
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_file, encoding=encoding)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.soft_boundary_value < 1.0):
        raise ValueError("--soft-boundary-value must be in (0, 1).")
    if not (0.0 < args.soft_min_value < 1.0):
        raise ValueError("--soft-min-value must be in (0, 1).")
    months_keep = month_filter(args.months)
    years = list(range(args.start_year, args.end_year + 1))

    storm_df = load_storm_points(
        ibtracs_file=args.ibtracs_file,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"[INFO] Loaded {len(storm_df)} valid IBTrACS usa_r34 points ({args.start_year}-{args.end_year}).")

    month_files = list(iter_month_files(args.data_root, years, months_keep, args.grid))
    if not month_files:
        raise FileNotFoundError("No monthly files found for given arguments.")

    by_year: dict[int, list[tuple[int, Path]]] = {}
    for year, month, nc_file in month_files:
        by_year.setdefault(year, []).append((month, nc_file))

    for year in sorted(by_year):
        out_file = args.output_dir / args.grid / f"typhoon_r34_mask_{args.grid}_{year}.nc"
        if args.skip_existing and out_file.exists():
            print(f"[SKIP] {out_file}")
            continue

        month_items = sorted(by_year[year], key=lambda x: x[0])
        time_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        soft_mask_chunks: list[np.ndarray] = []
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

            times, lats, lons, month_mask, month_soft_mask, hit_points = build_month_mask(
                nc_file,
                month_df,
                soft_boundary_value=float(args.soft_boundary_value),
                soft_min_value=float(args.soft_min_value),
            )
            if lat_ref is None:
                lat_ref = lats
                lon_ref = lons
            else:
                if not np.array_equal(lat_ref, lats) or not np.array_equal(lon_ref, lons):
                    raise ValueError(f"Grid mismatch in file: {nc_file}")

            time_chunks.append(times)
            mask_chunks.append(month_mask)
            soft_mask_chunks.append(month_soft_mask)
            monthly_hits += hit_points
            affected_ratio = float(month_mask.sum()) / float(month_mask.size)
            print(
                f"  [MONTH {month:02d}] points={len(month_df):5d}, "
                f"used={hit_points:5d}, affected_ratio={affected_ratio:.6f}"
            )

        all_times = np.concatenate(time_chunks, axis=0)
        all_masks = np.concatenate(mask_chunks, axis=0)
        all_soft_masks = np.concatenate(soft_mask_chunks, axis=0)
        save_year_dataset(
            out_file=out_file,
            times=all_times,
            lats=lat_ref,
            lons=lon_ref,
            data=all_masks,
            soft_data=all_soft_masks,
            grid=args.grid,
            soft_boundary_value=float(args.soft_boundary_value),
            soft_min_value=float(args.soft_min_value),
        )
        year_ratio = float(all_masks.sum()) / float(all_masks.size)
        print(
            f"[DONE] {out_file} | shape={all_masks.shape}, "
            f"used_points={monthly_hits}, affected_ratio={year_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
