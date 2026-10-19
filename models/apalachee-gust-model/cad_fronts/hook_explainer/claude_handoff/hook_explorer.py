#!/usr/bin/env python3
"""Portable viewer for the Apalachee idealized blue-hook experiment.

Preview fields and projected geometry are bundled in preview_data.npz.  The
optional --full-fields argument reads the original 3 km fields.npz archive.
This script intentionally leaves asinh normalization and gradient masking as
the next implementation task; see CLAUDE_PROMPT.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "preview_data.npz"
FIELD_BEGIN, END = 60.0, 80.0
TRACE_BEGIN = 64.0
UTC_START = datetime(2026, 12, 15, tzinfo=timezone.utc)


def load_data(full_fields: Path | None) -> dict:
    with np.load(DATA_PATH, allow_pickle=False) as d:
        result = {key: d[key] for key in d.files}
    meta = json.loads(str(result["metadata_json"]))
    result["metadata"] = meta
    if full_fields is None:
        result["theta_source_target_K"] = result["theta_source_target_K_3km"][::3, ::3]
        result["grid_spacing_m"] = float(meta["preview_grid_spacing_m"])
        result["resolution_label"] = "9 km sampled preview"
        return result

    with np.load(full_fields, allow_pickle=False) as source:
        fingerprint = str(source["source_fingerprint"])
        if fingerprint != str(result["model_fingerprint"]):
            raise ValueError("Full archive fingerprint does not match the bundled preview.")
        hours_all = source["hours"].astype(float)
        ti = np.flatnonzero((hours_all >= FIELD_BEGIN) & (hours_all <= END))
        if len(ti) == 0:
            raise ValueError("Full archive contains no fields from hours 60 through 80.")
        for key in ("theta_K", "F_K_100km_3h", "u_ms", "v_ms"):
            result[key] = np.asarray(source[key][ti], dtype=np.float32)
        result["cad_mask"] = np.asarray(source["cad_mask"], dtype=np.float32)
        result["theta_source_target_K"] = result["theta_source_target_K_3km"]
        result["hours"] = hours_all[ti].astype(np.float32)
    result["x_m"] = np.arange(result["theta_K"].shape[2], dtype=np.float32) * 3000.0
    result["y_m"] = np.arange(result["theta_K"].shape[1], dtype=np.float32) * 3000.0
    result["grid_spacing_m"] = 3000.0
    result["resolution_label"] = "original 3 km fields"
    return result


def label(hour: float) -> str:
    dt = UTC_START + timedelta(hours=float(hour))
    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%a Dec %d · %-I:%M %p EST")


def integrate_tracers(data: dict, dt_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Advect six 2-D markers through linearly interpolated archived winds."""
    hours = data["hours"].astype(float)
    x_axis, y_axis = data["x_m"].astype(float), data["y_m"].astype(float)
    dx = float(data["grid_spacing_m"])
    seed_xy = data["seed_xy_m"].astype(float)
    times = np.arange(TRACE_BEGIN, END + dt_s / 3600.0 * 0.5, dt_s / 3600.0)
    tracks = np.full((len(times), len(seed_xy), 2), np.nan, dtype=float)
    tracks[0] = seed_xy

    def velocity(t: float, xy: np.ndarray) -> np.ndarray:
        lo = int(np.clip(np.searchsorted(hours, t, side="right") - 1, 0, len(hours) - 2))
        frac = (t - hours[lo]) / (hours[lo + 1] - hours[lo])
        out = np.full_like(xy, np.nan, dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        coords = np.vstack(((xy[valid, 1] - y_axis[0]) / dx,
                            (xy[valid, 0] - x_axis[0]) / dx))
        for k, key in enumerate(("u_ms", "v_ms")):
            a = map_coordinates(data[key][lo], coords, order=1, mode="constant", cval=np.nan)
            b = map_coordinates(data[key][lo + 1], coords, order=1, mode="constant", cval=np.nan)
            out[valid, k] = (1.0 - frac) * a + frac * b
        return out

    for k in range(len(times) - 1):
        h = times[k]
        first = velocity(h, tracks[k])
        midpoint = tracks[k] + 0.5 * dt_s * first
        tracks[k + 1] = tracks[k] + dt_s * velocity(h + dt_s / 7200.0, midpoint)
    return times, tracks


def unpack_segments(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def make_figure(data: dict, hour: float, tracks: np.ndarray, track_times: np.ndarray):
    hours = data["hours"].astype(float)
    frame = int(np.argmin(np.abs(hours - hour)))
    hour = float(hours[frame])
    x_all, y_all, dx = data["x_m"], data["y_m"], float(data["grid_spacing_m"])
    left, right, bottom, top = data["metadata"]["display_extent_xy_m"]
    ix = np.flatnonzero((x_all >= left - dx) & (x_all <= right + dx))
    iy = np.flatnonzero((y_all >= bottom - dx) & (y_all <= top + dx))
    xs, ys = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)
    x, y = x_all[xs], y_all[ys]
    extent = (float(x[0] - dx / 2), float(x[-1] + dx / 2),
              float(y[0] - dx / 2), float(y[-1] + dx / 2))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.2), dpi=110, facecolor="white")
    fig.subplots_adjust(left=.045, right=.98, top=.86, bottom=.19, wspace=.08)
    colors = ["#116d91"] * 3 + ["#bf491c"] * 3
    for ax in axes:
        ax.set_facecolor("#e7edf2")
        ax.add_collection(LineCollection(unpack_segments(data["coast_xy"], data["coast_offsets"]),
                         colors="#33343a", linewidths=.7, zorder=5))
        ax.add_collection(LineCollection(unpack_segments(data["state_xy"], data["state_offsets"]),
                         colors="#666771", linewidths=.55, zorder=5))
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        for name, (cx, cy) in zip(data["city_names"], data["city_xy_m"]):
            name = str(name)
            ax.plot(cx, cy, "o", ms=2.5, color="#22232b", zorder=8)
            ax.annotate(name, (cx, cy), xytext=(4, 4), textcoords="offset points", fontsize=8,
                        zorder=8, bbox=dict(fc="white", ec="none", alpha=.8, pad=.8))
        ax.set_xlabel("Lambert x (m)")
        ax.set_ylabel("Lambert y (m)")

    theta = data["theta_K"][frame, ys, xs]
    front = data["F_K_100km_3h"][frame, ys, xs]
    im_theta = axes[0].imshow(theta, extent=extent, origin="lower", cmap="RdYlBu_r",
                              vmin=274, vmax=297, zorder=2, interpolation="nearest", aspect="auto")
    im_front = axes[1].imshow(front, extent=extent, origin="lower", cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-16, vcenter=0, vmax=16), zorder=2,
                              interpolation="nearest", aspect="auto")
    axes[0].set_title("Air masses + moving tracers", loc="left", weight="bold", fontsize=13)
    axes[1].set_title("Where the temperature contrast changes", loc="left", weight="bold", fontsize=13)
    qstep = max(1, round(84_000 / dx))
    xx, yy = np.meshgrid(x, y)
    arrows = axes[0].quiver(xx[::qstep, ::qstep], yy[::qstep, ::qstep],
        data["u_ms"][frame, ys, xs][::qstep, ::qstep], data["v_ms"][frame, ys, xs][::qstep, ::qstep],
        angles="xy", scale_units="xy", scale=1 / 3500, width=.0022, color="#303139",
        alpha=.65, zorder=6)
    axes[0].quiverkey(arrows, .14, .045, 20, "20 m/s layer wind", coordinates="axes",
                      labelpos="E", fontproperties={"size": 7})
    for ax in axes:
        ax.contour(xx, yy, theta, levels=np.arange(276, 298, 4), colors="#32333a",
                   linewidths=.55, alpha=.6, zorder=4)
    for ax in axes:
        ax.contour(xx, yy, data["cad_mask"][ys, xs], levels=[.5], colors=["#247f7b"],
                   linestyles="dashed", linewidths=1.0, zorder=6)

    if hour >= TRACE_BEGIN:
        tidx = int(np.argmin(np.abs(track_times - hour)))
        trail_start = max(0, tidx - int(round(4.0 / (track_times[1] - track_times[0]))))
        for ax in axes:
            for j, color in enumerate(colors):
                ax.plot(tracks[trail_start:tidx + 1, j, 0], tracks[trail_start:tidx + 1, j, 1],
                        lw=2, color=color, alpha=.95, zorder=9)
                ax.plot([tracks[tidx, j, 0]], [tracks[tidx, j, 1]], "o", ms=5,
                        mfc=color, mec="white", mew=.8, zorder=10)
    cx, cy = data["low_center_xy_m"][frame]
    for ax in axes:
        ax.plot([cx], [cy], marker="*", ms=14, mfc="#ffe17c", mec="#25252d", zorder=11)
        ax.annotate(f"L · {data['low_pressure_mb'][frame]:.0f} mb", (cx, cy),
                    xytext=(12, -18), textcoords="offset points", fontsize=9, weight="bold",
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=1.), zorder=11)

    fig.text(.035, .965, "Inside the blue hook", fontsize=23, weight="bold", color="#20212a")
    fig.text(.98, .965, f"SIMULATED · {data['resolution_label'].upper()}", ha="right",
             fontsize=9, weight="bold", color="#9b1c63")
    fig.text(.035, .92, f"{label(hour)} · arrows: lower-layer transport · contours: potential temperature",
             fontsize=10, color="#555660")
    c0 = fig.colorbar(im_theta, cax=fig.add_axes([.075, .105, .36, .017]), orientation="horizontal")
    c0.set_label("Potential temperature (K) · cooler → warmer", fontsize=9)
    c1 = fig.colorbar(im_front, cax=fig.add_axes([.57, .105, .36, .017]), orientation="horizontal", extend="both")
    c1.set_label("K / 100 km / 3 h · blue weakens, red strengthens", fontsize=9)
    for cb in (c0, c1): cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([0], [0], color="#116d91", marker="o", label="Hook-side tracers"),
                        Line2D([0], [0], color="#bf491c", marker="o", label="Gulf tracers"),
                        Line2D([0], [0], color="#247f7b", ls="--", label="Schematic CAD corridor")],
               loc="lower center", bbox_to_anchor=(.5, .019), ncol=3, frameon=False, fontsize=8)
    fig.text(.035, .008, "2-D prescribed-flow markers are not 3-D parcels. Archived kinematic F excludes the separate CAD restoring source Q.",
             fontsize=7, color="#666771")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-fields", type=Path, help="optional original fields.npz at 3 km")
    parser.add_argument("--hour", type=float, default=70.0, help="reference PNG time; default 70")
    parser.add_argument("--output", type=Path, default=HERE / "hook_preview.png")
    parser.add_argument("--video", type=Path, help="write 64-80 h animation (requires ffmpeg)")
    parser.add_argument("--no-video", action="store_true", help="explicitly render only the PNG (the default)")
    args = parser.parse_args()
    if not FIELD_BEGIN <= args.hour <= END:
        parser.error("--hour must be between 60 and 80")
    data = load_data(args.full_fields)
    times, tracks = integrate_tracers(data)
    finite_endpoints = np.isfinite(tracks[-1]).all(axis=1).sum()
    if not np.isfinite(tracks).any():
        raise RuntimeError("Tracer integration produced no finite positions; check data coordinates.")
    fig = make_figure(data, args.hour, tracks, times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    print(f"Saved {args.output} · {data['resolution_label']} · {finite_endpoints}/{len(tracks[0])} finite hour-80 endpoints")
    if args.video:
        if not shutil.which("ffmpeg"):
            parser.error("--video requires ffmpeg on PATH")
        hours = data["hours"].astype(float)
        frame_indices = np.flatnonzero((hours >= TRACE_BEGIN) & (hours <= END))
        args.video.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hook_frames_", dir=args.video.parent) as temp:
            temp_path = Path(temp)
            for num, fi in enumerate(frame_indices):
                frame_fig = make_figure(data, float(hours[fi]), tracks, times)
                frame_fig.savefig(temp_path / f"frame_{num:03d}.png", dpi=100, facecolor="white")
                plt.close(frame_fig)
            encoded = temp_path / "hook_preview.encoding.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", "5",
                            "-i", str(temp_path / "frame_%03d.png"), "-vf",
                            "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-b:v", "2200k",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)], check=True)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(encoded),
                            "-f", "null", "-"], check=True)
            encoded.replace(args.video)
        print(f"Saved {args.video}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
