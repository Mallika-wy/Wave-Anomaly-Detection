#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


BACKGROUND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ocean_backdrop",
    ["#031525", "#062b45", "#0c4a6e", "#0f766e"],
)
TRACK_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "storm_intensity",
    ["#7dd3fc", "#4ade80", "#facc15", "#fb923c", "#ef4444"],
)


@dataclass
class StormTrack:
    sid: str
    name: str
    times: pd.DatetimeIndex
    lats: np.ndarray
    lons: np.ndarray
    peak_wind: float
    peak_lon: float
    peak_lat: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a polished IBTrACS typhoon track map for a specific calendar year."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("IBTrACS.WP.v04r01.nc"),
        help="Path to the IBTrACS netCDF file.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Calendar year to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/wp_typhoon_tracks_2025.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=6,
        help="Number of strongest named storms to annotate.",
    )
    return parser.parse_args()


def decode_scalar(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


def choose_wind_series(ds: xr.Dataset) -> np.ndarray:
    for key in ("usa_wind", "wmo_wind", "tokyo_wind", "cma_wind"):
        if key in ds:
            return ds[key].values.astype(float)
    return np.full(ds["lat"].shape, np.nan, dtype=float)


def load_tracks(nc_path: Path, year: int) -> list[StormTrack]:
    ds = xr.open_dataset(nc_path)

    time_values = ds["time"].values
    lat_values = ds["lat"].values.astype(float)
    lon_values = ds["lon"].values.astype(float)
    wind_values = choose_wind_series(ds)
    names = ds["name"].values
    sids = ds["sid"].values

    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)

    tracks: list[StormTrack] = []
    for storm_idx in range(ds.sizes["storm"]):
        storm_times = pd.to_datetime(time_values[storm_idx], errors="coerce")
        in_year = (~pd.isna(storm_times)) & (storm_times >= start) & (storm_times < end)
        if not np.any(in_year):
            continue

        times = pd.DatetimeIndex(storm_times[in_year])
        lats = lat_values[storm_idx, in_year]
        lons = lon_values[storm_idx, in_year]
        winds = wind_values[storm_idx, in_year]

        valid_geo = np.isfinite(lats) & np.isfinite(lons)
        if int(valid_geo.sum()) == 0:
            continue

        times = pd.DatetimeIndex(times[valid_geo])
        lats = lats[valid_geo]
        lons = lons[valid_geo]
        winds = winds[valid_geo]

        if np.isfinite(winds).any():
            peak_idx = int(np.nanargmax(winds))
            peak_wind = float(np.nanmax(winds))
        else:
            peak_idx = len(lats) // 2
            peak_wind = np.nan

        tracks.append(
            StormTrack(
                sid=decode_scalar(sids[storm_idx]),
                name=decode_scalar(names[storm_idx]) or "UNNAMED",
                times=times,
                lats=lats,
                lons=lons,
                peak_wind=peak_wind,
                peak_lon=float(lons[peak_idx]),
                peak_lat=float(lats[peak_idx]),
            )
        )

    if not tracks:
        raise ValueError(f"No valid storm track points found in {year}.")

    return sorted(
        tracks,
        key=lambda item: (-np.inf if np.isnan(item.peak_wind) else item.peak_wind),
    )


def padded_extent(tracks: list[StormTrack]) -> tuple[float, float, float, float]:
    all_lons = np.concatenate([track.lons for track in tracks])
    all_lats = np.concatenate([track.lats for track in tracks])

    lon_min = max(0.0, np.floor((np.nanmin(all_lons) - 3.0) / 5.0) * 5.0)
    lon_max = min(180.0, np.ceil((np.nanmax(all_lons) + 3.0) / 5.0) * 5.0)
    lat_min = max(0.0, np.floor((np.nanmin(all_lats) - 3.0) / 5.0) * 5.0)
    lat_max = min(60.0, np.ceil((np.nanmax(all_lats) + 3.0) / 5.0) * 5.0)
    return lon_min, lon_max, lat_min, lat_max


def add_backdrop(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    xs = np.linspace(lon_min, lon_max, 800)
    ys = np.linspace(lat_min, lat_max, 500)
    xx, yy = np.meshgrid(xs, ys)

    backdrop = (
        0.54
        + 0.22 * np.sin((xx - lon_min) / 7.0)
        + 0.16 * np.cos((yy - lat_min) / 4.5)
        + 0.08 * np.sin((xx + 1.6 * yy) / 12.0)
    )
    backdrop = (backdrop - backdrop.min()) / (backdrop.max() - backdrop.min())

    ax.imshow(
        backdrop,
        extent=extent,
        origin="lower",
        cmap=BACKGROUND_CMAP,
        interpolation="bilinear",
        alpha=0.96,
        aspect="auto",
        zorder=0,
    )
    ax.contour(
        xx,
        yy,
        backdrop,
        levels=12,
        colors="white",
        linewidths=0.35,
        alpha=0.07,
        zorder=0.5,
    )


def add_region_labels(ax: plt.Axes) -> None:
    labels = [
        ("South China Sea", 114.0, 14.5),
        ("Philippine Sea", 134.5, 18.0),
        ("East China Sea", 126.0, 28.5),
        ("Western North Pacific", 152.0, 31.5),
    ]
    for text, x, y in labels:
        ax.text(
            x,
            y,
            text,
            color="#cfe8f3",
            fontsize=10,
            alpha=0.26,
            fontstyle="italic",
            ha="center",
            va="center",
            zorder=1,
        )


def intensity_norm(tracks: list[StormTrack]) -> mcolors.Normalize:
    peak_values = [track.peak_wind for track in tracks if np.isfinite(track.peak_wind)]
    vmax = max(120.0, float(np.ceil(max(peak_values, default=80.0) / 10.0) * 10.0))
    return mcolors.Normalize(vmin=15.0, vmax=vmax)


def title_case_name(name: str) -> str:
    if name.upper() == "UNNAMED":
        return "Unnamed"
    return name.title()


def plot_tracks(tracks: list[StormTrack], year: int, output_path: Path, label_top_k: int) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")

    extent = padded_extent(tracks)
    norm = intensity_norm(tracks)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#e5eef6",
            "xtick.color": "#dbeafe",
            "ytick.color": "#dbeafe",
        }
    )

    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    fig.patch.set_facecolor("#02111f")
    ax.set_facecolor("#041c32")

    add_backdrop(ax, extent)
    add_region_labels(ax)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(extent[0], extent[1] + 1, 10))
    ax.set_xticks(np.arange(extent[0], extent[1] + 1, 5), minor=True)
    ax.set_yticks(np.arange(extent[2], extent[3] + 1, 10))
    ax.set_yticks(np.arange(extent[2], extent[3] + 1, 5), minor=True)
    ax.grid(which="major", color="white", linewidth=0.8, alpha=0.17)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}°N"))

    for spine in ax.spines.values():
        spine.set_color("#dbeafe")
        spine.set_alpha(0.24)
        spine.set_linewidth(1.2)

    ordered_tracks = sorted(
        tracks,
        key=lambda item: (-1 if np.isnan(item.peak_wind) else item.peak_wind),
    )
    for track in ordered_tracks:
        color = TRACK_CMAP(norm(track.peak_wind if np.isfinite(track.peak_wind) else norm.vmin))
        width = 1.35 + 2.55 * norm(track.peak_wind if np.isfinite(track.peak_wind) else norm.vmin)

        ax.plot(
            track.lons,
            track.lats,
            color=color,
            lw=width,
            alpha=0.9,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            track.lons,
            track.lats,
            s=14,
            color=color,
            alpha=0.5,
            linewidths=0,
            zorder=3.1,
        )
        ax.scatter(
            track.lons[0],
            track.lats[0],
            s=58,
            marker="^",
            facecolor="#e0f2fe",
            edgecolor="#082f49",
            linewidth=1.0,
            zorder=4.2,
        )
        ax.scatter(
            track.lons[-1],
            track.lats[-1],
            s=66,
            marker="o",
            facecolor="#fbbf24",
            edgecolor="#78350f",
            linewidth=1.0,
            zorder=4.3,
        )

    named_tracks = [track for track in tracks if track.name.upper() != "UNNAMED" and np.isfinite(track.peak_wind)]
    label_offsets = [(14, 12), (14, -14), (-16, 14), (-18, -14), (16, 18), (-18, 18), (18, -20)]
    for idx, track in enumerate(sorted(named_tracks, key=lambda item: item.peak_wind, reverse=True)[:label_top_k]):
        dx, dy = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            title_case_name(track.name),
            xy=(track.peak_lon, track.peak_lat),
            xytext=(dx, dy),
            textcoords="offset points",
            color="#f8fafc",
            fontsize=10,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": (1, 1, 1, 0.5),
                "lw": 0.8,
                "shrinkA": 0,
                "shrinkB": 6,
            },
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": (2 / 255, 17 / 255, 31 / 255, 0.72),
                "ec": (1, 1, 1, 0.14),
                "lw": 0.6,
            },
            path_effects=[pe.withStroke(linewidth=2.2, foreground="#06121f", alpha=0.9)],
            zorder=5,
        )

    peak_track = max(
        (track for track in tracks if np.isfinite(track.peak_wind)),
        key=lambda item: item.peak_wind,
    )
    first_time = min(track.times[0] for track in tracks).strftime("%Y-%m-%d")
    last_time = max(track.times[-1] for track in tracks).strftime("%Y-%m-%d")
    named_count = sum(track.name.upper() != "UNNAMED" for track in tracks)

    summary = (
        f"{len(tracks)} storms in {year}\n"
        f"{named_count} named systems\n"
        f"{first_time} to {last_time}\n"
        f"Peak intensity: {title_case_name(peak_track.name)} ({peak_track.peak_wind:.0f} kt)"
    )
    ax.text(
        0.018,
        0.982,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#e5eef6",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.45",
            "fc": (3 / 255, 17 / 255, 32 / 255, 0.82),
            "ec": (1, 1, 1, 0.12),
            "lw": 0.8,
        },
        zorder=6,
    )

    ax.set_title(
        "Western North Pacific Typhoon Tracks in 2025",
        fontsize=21,
        color="#f8fafc",
        pad=18,
    )
    ax.text(
        0.5,
        1.015,
        "IBTrACS.WP.v04r01 | Track positions clipped to calendar year 2025 | Color shows each storm's peak USA wind",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#cbd5e1",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#e0f2fe",
            markeredgecolor="#082f49",
            markersize=9,
            label="First position in 2025",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#fbbf24",
            markeredgecolor="#78350f",
            markersize=9,
            label="Last position in 2025",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor=(2 / 255, 17 / 255, 31 / 255, 0.65),
        edgecolor=(1, 1, 1, 0.12),
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("#e5eef6")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=TRACK_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Peak USA Wind (kt)", color="#e5eef6")
    cbar.ax.yaxis.set_tick_params(color="#dbeafe")
    plt.setp(cbar.ax.get_yticklabels(), color="#dbeafe")
    cbar.outline.set_edgecolor((1, 1, 1, 0.18))

    fig.text(
        0.013,
        0.012,
        "Markers highlight the first and last best-track positions recorded within 2025.",
        color="#cbd5e1",
        fontsize=10,
    )

    fig.savefig(output_path, dpi=320, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path, pdf_path


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.input, args.year)
    png_path, pdf_path = plot_tracks(tracks, args.year, args.output, args.label_top_k)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Storm count: {len(tracks)}")


if __name__ == "__main__":
    main()
