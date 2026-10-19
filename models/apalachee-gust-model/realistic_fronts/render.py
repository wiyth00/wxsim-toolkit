#!/usr/bin/env python3
"""Render reproducible maps and an animation for the dry thermal-front scenario."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("APALACHEE_SKIP_CACHE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import model as fm
from plotkit import COLORS as GUST_COLORS, FILL_LEVELS as GUST_LEVELS, INK, MUTED, draw_cbar

MAP = getattr(fm, "m", None)
if MAP is None:
    MAP = getattr(getattr(fm, "bg", None), "m", None)
if MAP is None:
    raise AttributeError("realistic_fronts.model must expose its Basemap as `m` or `bg.m`")

ACCENT = "#9b1c63"
THETA_CMAP = "RdYlBu_r"
F_LEVEL = 2.0
F_LEVELS = np.arange(-F_LEVEL, F_LEVEL + 0.001, 0.25)
F_TICKS = np.arange(-2, 2.01, 0.5)
TIMES_LIFECYCLE = (36, 54, 78, 108)
FRAME_DPI = 115


def _load_case(case: Path) -> dict:
    fields_path, peak_path = case / "fields.npz", case / "peak.npz"
    if not fields_path.is_file() or not peak_path.is_file():
        raise FileNotFoundError(f"{case} must contain fields.npz and peak.npz")
    with np.load(fields_path) as z:
        fields = {k: z[k] for k in z.files}
    with np.load(peak_path) as z:
        peak = {k: z[k] for k in z.files}
    required = {"hours", "theta_K", "F_K_100km_3h", "gust_mph", "p_mb", "u_ms", "v_ms"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"{fields_path} is missing fields: {', '.join(sorted(missing))}")
    if "G" not in peak or "T" not in peak:
        raise ValueError(f"{peak_path} must contain G and T")
    expected = (len(fields["hours"]),) + np.shape(fm.X)
    for key in required - {"hours"}:
        if fields[key].shape != expected:
            raise ValueError(f"{key} has shape {fields[key].shape}, expected {expected}")
    for key in ("G", "T"):
        if peak[key].shape != np.shape(fm.X):
            raise ValueError(f"peak {key} has shape {peak[key].shape}, expected {np.shape(fm.X)}")
    return {"case": case, "fields": fields, "peak": peak}


def _interior_mask(shape: tuple[int, int]) -> np.ndarray:
    """Mask the outer three grid cells, where finite differences are least reliable."""
    mask = np.ones(shape, dtype=bool)
    mask[:3, :] = mask[-3:, :] = False
    mask[:, :3] = mask[:, -3:] = False
    return mask


def _map_background(ax):
    m = MAP
    m.drawmapboundary(fill_color="#e7edf2", linewidth=0, ax=ax)
    m.fillcontinents(color="#f5f3ee", lake_color="#e7edf2", zorder=0.4, ax=ax)
    try:
        m.drawcounties(linewidth=0.12, color="#a6a5a3", zorder=2.6, ax=ax)
    except Exception:
        pass
    m.drawstates(linewidth=0.55, color="#53535a", zorder=3, ax=ax)
    m.drawcountries(linewidth=0.55, color="#53535a", zorder=3, ax=ax)
    m.drawcoastlines(linewidth=0.6, color="#36383d", zorder=3, ax=ax)


def _time_index(hours: np.ndarray, hour: float) -> int:
    return int(np.nanargmin(np.abs(hours - hour)))


def _front_cmap():
    return matplotlib.colormaps["RdBu_r"]


def _draw_time_axes(ax_g, ax_f, data: dict, index: int, draw_background: bool = True):
    fields = data["fields"]
    h = float(fields["hours"][index])
    theta = fields["theta_K"][index]
    front = np.ma.masked_where(~_interior_mask(theta.shape), fields["F_K_100km_3h"][index])
    gust = np.ma.masked_where(~_interior_mask(theta.shape), fields["gust_mph"][index])
    pressure = np.ma.masked_where(~_interior_mask(theta.shape), fields["p_mb"][index])
    if draw_background:
        _map_background(ax_g)
    gust_cf = ax_g.contourf(fm.X, fm.Y, gust, levels=GUST_LEVELS, colors=GUST_COLORS, extend="max", alpha=0.9, zorder=2)
    p_levels = np.arange(936, 1025, 4)
    cs = ax_g.contour(fm.X, fm.Y, pressure, levels=p_levels, colors="#34333a", linewidths=0.62, alpha=0.9, zorder=4)
    p_labels = ax_g.clabel(cs, inline=True, fontsize=6, fmt="%d", colors="#34333a")
    ax_g.set_title("Surface gust proxy + sea-level pressure", loc="left", fontsize=10, color=INK, pad=7, weight="semibold")

    if draw_background:
        _map_background(ax_f)
    cf = ax_f.contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(), norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", alpha=0.96, zorder=2)
    tlevels = np.arange(270, 321, 2)
    tc = ax_f.contour(fm.X, fm.Y, np.ma.masked_where(~_interior_mask(theta.shape), theta), levels=tlevels,
                      colors="#252932", linewidths=0.46, alpha=0.8, zorder=4)
    theta_labels = ax_f.clabel(tc, inline=True, fontsize=5.7, fmt="%.0f", colors="#252932")
    ax_f.set_title("Signed frontogenesis + potential temperature θ (K)", loc="left", fontsize=10, color=INK, pad=7, weight="semibold")
    return h, gust_cf, cf, [gust_cf, cs, cf, tc, *p_labels, *theta_labels]


def _case_name(data: dict) -> str:
    names = {"size_2p5x": "2.5× length scale", "size_3x": "3× length scale"}
    return names.get(data["case"].name, data["case"].name.replace("_", " "))


def _title(fig, data: dict, hour: float, suffix: str = ""):
    fig.text(0.035, 0.976, f"Dry thermal-front diagnostics · {_case_name(data)}{suffix}",
             ha="left", va="top", fontsize=15, weight="bold", color=INK)
    fig.text(0.965, 0.949, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2,
             weight="bold", color=ACCENT)
    fig.text(0.035, 0.949, f"Apalachee Bay sou’easter · model hour {hour:g} · minimum 939 mb restored · prescribed dry-flow experiment",
             ha="left", va="top", fontsize=8.5, color=MUTED)


def _add_footer(fig, left=""):
    note = "Frontogenesis colors are signed; positive = gradient strengthening, negative = frontolysis. " \
           "Outer 3 grid cells are masked. Scenario output, not a forecast."
    fig.text(0.035, 0.016, (left + "   ·   " if left else "") + note,
             ha="left", va="bottom", fontsize=7, color=MUTED)


def render_snapshot(data: dict, out: Path, requested_hour: float = 54.0):
    fields = data["fields"]
    i = _time_index(fields["hours"], requested_hour)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), dpi=170, facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.89, bottom=0.19, wspace=0.08)
    h, gust_cf, cf, _ = _draw_time_axes(*axes, data, i)
    _title(fig, data, h)
    cbar = fig.colorbar(cf, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]), orientation="horizontal", ticks=F_TICKS)
    cbar.set_label("Frontogenesis  (K per 100 km per 3 h)", fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=7, length=2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Instantaneous gust proxy (mph)", ha="center", fontsize=8, color=INK)
    fig.text(0.745, 0.105, "Frontogenesis · signed (K / 100 km / 3 h)", ha="center", fontsize=8, color=INK)
    fig.text(0.50, 0.049, "Pressure contours: 4 mb interval", ha="center", fontsize=7.5, color=MUTED)
    _add_footer(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return h


def render_lifecycle(data: dict, out: Path):
    fields = data["fields"]
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 9.7), dpi=145, facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.89, bottom=0.13, hspace=0.24, wspace=0.07)
    last_cf = None
    actual = []
    for col, requested in enumerate(TIMES_LIFECYCLE):
        i = _time_index(fields["hours"], requested)
        h = float(fields["hours"][i]); actual.append(h)
        front = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["F_K_100km_3h"][i])
        theta = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["theta_K"][i])
        _map_background(axes[0, col]); _map_background(axes[1, col])
        last_cf = axes[0, col].contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(),
                                        norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", zorder=2)
        tlevels = np.arange(270, 321, 2)
        tc = axes[0, col].contour(fm.X, fm.Y, theta, levels=tlevels, colors="#202631", linewidths=0.45, zorder=4)
        axes[0, col].clabel(tc, inline=True, fontsize=5.2, fmt="%.0f")
        axes[0, col].set_title(f"Hour {h:g}", fontsize=9, color=INK, weight="semibold")

        # Theta row uses actual temperature values with a fixed shared range.
        tcf = axes[1, col].contourf(fm.X, fm.Y, theta, levels=np.arange(270, 321, 2), cmap=THETA_CMAP,
                                    extend="both", zorder=2)
        axes[1, col].contour(fm.X, fm.Y, theta, levels=np.arange(270, 321, 2), colors="#252932", linewidths=0.35, zorder=4)
        axes[1, col].set_title("Potential temperature (K)", fontsize=8, color=INK)
    _title(fig, data, actual[1], " · lifecycle")
    cb = fig.colorbar(last_cf, ax=axes[0, :].tolist(), orientation="horizontal", fraction=0.06, pad=0.08, ticks=F_TICKS)
    cb.set_label("Frontogenesis  (K per 100 km per 3 h); blue = frontolysis, orange = frontogenesis", fontsize=8)
    cb2 = fig.colorbar(tcf, ax=axes[1, :].tolist(), orientation="horizontal", fraction=0.06, pad=0.08,
                       ticks=np.arange(270, 321, 10))
    cb2.set_label("Potential temperature θ (K)", fontsize=8)
    _add_footer(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_frame(data: dict, index: int, out: Path):
    fields = data["fields"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), dpi=FRAME_DPI, facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.89, bottom=0.19, wspace=0.08)
    h, gust_cf, cf, _ = _draw_time_axes(*axes, data, index)
    _title(fig, data, h)
    cbar = fig.colorbar(cf, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]), orientation="horizontal", ticks=F_TICKS)
    cbar.set_label("Frontogenesis  (K per 100 km per 3 h)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Instantaneous gust proxy (mph)", ha="center", fontsize=8)
    fig.text(0.745, 0.105, "Frontogenesis · signed (K / 100 km / 3 h)", ha="center", fontsize=8)
    fig.text(0.50, 0.049, "Pressure contours: 4 mb interval", ha="center", fontsize=7.5, color=MUTED)
    _add_footer(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def render_animation(data: dict, out_dir: Path, fps: int = 8):
    frames = out_dir / "front_frames"
    frames.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), dpi=FRAME_DPI, facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.89, bottom=0.19, wspace=0.08)
    title_text = None
    time_text = None
    old_dynamic = []
    for i, hour in enumerate(data["fields"]["hours"]):
        for artist in old_dynamic:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        h, gust_cf, front_cf, old_dynamic = _draw_time_axes(*axes, data, i, draw_background=(i == 0))
        if i == 0:
            title_text = fig.text(0.035, 0.976, f"Dry thermal-front diagnostics · {_case_name(data)}",
                                  ha="left", va="top", fontsize=15, weight="bold", color=INK)
            time_text = fig.text(0.035, 0.949, "", ha="left", va="top", fontsize=8.5, color=MUTED)
            fig.text(0.965, 0.949, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2, weight="bold", color=ACCENT)
            draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
            front_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), cmap=_front_cmap())
            cbar = fig.colorbar(front_sm, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]),
                                orientation="horizontal", ticks=F_TICKS, extend="both")
            cbar.ax.tick_params(labelsize=7, length=2)
            fig.text(0.255, 0.105, "Instantaneous gust proxy (mph)", ha="center", fontsize=8)
            fig.text(0.745, 0.105, "Frontogenesis · signed (K / 100 km / 3 h)", ha="center", fontsize=8)
            fig.text(0.50, 0.049, "Pressure contours: 4 mb interval", ha="center", fontsize=7.5, color=MUTED)
            _add_footer(fig)
        time_text.set_text(f"Apalachee Bay sou’easter · model hour {h:g} · minimum 939 mb restored · prescribed dry-flow experiment")
        target = frames / f"frame_{i:03d}.png"
        fig.savefig(target, facecolor="white")
    plt.close(fig)
    from PIL import Image
    frame_paths = sorted(frames.glob("frame_*.png"))
    expected_count = len(data["fields"]["hours"])
    if len(frame_paths) != expected_count:
        raise RuntimeError(f"found {len(frame_paths)} frame PNGs; expected {expected_count}")
    for frame in frame_paths:
        with Image.open(frame) as img:
            img.verify()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"ffmpeg unavailable; PIL verified {len(frame_paths)} frames in {frames}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "frontogenesis.mp4"
    tmp = out_dir / ".frontogenesis.tmp.mp4"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame_%03d.png"),
           "-xerror", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a nonempty video")
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(tmp), "-f", "null", "-"], check=True)
    os.replace(tmp, target)
    return target


def render_comparison(cases: list[dict], out: Path, hour: float = 54.0):
    if len(cases) != 2:
        raise ValueError("comparison requires exactly two cases")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 14.0), dpi=165, facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.98, top=0.89, bottom=0.19, hspace=0.16, wspace=0.08)
    last_cf = None
    for row, data in enumerate(cases):
        peak = np.ma.masked_where(~_interior_mask(data["peak"]["G"].shape), data["peak"]["G"])
        _map_background(axes[row, 0])
        axes[row, 0].contourf(fm.X, fm.Y, peak, levels=GUST_LEVELS, colors=GUST_COLORS, extend="max", alpha=0.92, zorder=2)
        axes[row, 0].set_title(f"{_case_name(data)} · peak gust {data['peak']['G'].max():.1f} mph", loc="left", fontsize=10, color=INK, weight="semibold")
        fields = data["fields"]
        i = _time_index(fields["hours"], hour)
        front = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["F_K_100km_3h"][i])
        theta = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["theta_K"][i])
        _map_background(axes[row, 1])
        last_cf = axes[row, 1].contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(),
                                        norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", zorder=2)
        tc = axes[row, 1].contour(fm.X, fm.Y, theta, levels=np.arange(270, 321, 2), colors="#202631", linewidths=0.45, zorder=4)
        axes[row, 1].clabel(tc, inline=True, fontsize=5.3, fmt="%.0f")
        axes[row, 1].set_title(f"Hour {fields['hours'][i]:g} · signed frontogenesis + θ (K)", loc="left", fontsize=10, color=INK, weight="semibold")
    fig.text(0.035, 0.974, "Thermal-front experiment · size sensitivity", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.965, 0.974, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2, weight="bold", color=ACCENT)
    fig.text(0.035, 0.949, "Minimum 939 mb restored · scale changes model length scale, not an exact 34-kt radius.",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    cb = fig.colorbar(last_cf, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]), orientation="horizontal", ticks=F_TICKS)
    cb.set_label("Frontogenesis (K per 100 km per 3 h); blue = frontolysis, orange = frontogenesis", fontsize=7.2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Maximum gust proxy (mph)", ha="center", fontsize=8)
    fig.text(0.745, 0.105, "Frontogenesis at requested hour", ha="center", fontsize=8)
    _add_footer(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _validate_provenance(data: dict):
    summary_file = data["case"] / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"required case summary is missing: {summary_file}")
    summary = json.loads(summary_file.read_text())
    if "source_fingerprint" not in data["fields"]:
        raise ValueError(f"{data['case'] / 'fields.npz'} lacks source_fingerprint")
    if "source_fingerprint" not in summary:
        raise ValueError(f"{summary_file} lacks source_fingerprint")
    if summary.get("source_changes_invalidate_outputs") is not True:
        raise ValueError(f"{summary_file} does not affirm source-change cache invalidation")
    field_value = np.asarray(data["fields"]["source_fingerprint"]).item()
    field_sig = field_value.decode() if isinstance(field_value, bytes) else str(field_value)
    summary_sig = str(summary["source_fingerprint"])
    if field_sig != summary_sig:
        raise ValueError(f"fields/summary provenance mismatch in {data['case']}")
    if "source_fingerprint" not in data["peak"]:
        raise ValueError(f"{data['case'] / 'peak.npz'} lacks source_fingerprint")
    peak_value = np.asarray(data["peak"]["source_fingerprint"]).item()
    peak_sig = peak_value.decode() if isinstance(peak_value, bytes) else str(peak_value)
    if peak_sig != summary_sig:
        raise ValueError(f"peak/summary provenance mismatch in {data['case']}")
    runtime_sig = getattr(fm, "SOURCE_FINGERPRINT", None)
    if runtime_sig is not None and str(runtime_sig) != field_sig:
        raise ValueError(f"saved output is stale for the current model source: {data['case']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, help="one case directory containing fields.npz and peak.npz")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CASE_A", "CASE_B"), help="render 2.5x vs 3x comparison")
    parser.add_argument("--root", type=Path, default=HERE, help="directory for default case lookup and comparison output")
    parser.add_argument("--hour", type=float, default=54.0, help="snapshot hour (default: 54)")
    parser.add_argument("--animation", action="store_true", help="write all half-hour frames and encode MP4")
    parser.add_argument("--fps", type=int, default=8, help="animation frame rate")
    args = parser.parse_args(argv)
    if bool(args.case) == bool(args.compare):
        parser.error("choose exactly one of --case or --compare")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.compare:
        cases = [_load_case(p.resolve()) for p in args.compare]
        for data in cases: _validate_provenance(data)
        out = args.root.resolve() / "size_comparison.png"
        render_comparison(cases, out, args.hour)
        print(f"saved {out}")
        return 0
    case = args.case.resolve()
    data = _load_case(case)
    _validate_provenance(data)
    snapshot_hour = render_snapshot(data, case / "front_snapshot.png", args.hour)
    render_lifecycle(data, case / "front_lifecycle.png")
    print(f"saved {case / 'front_snapshot.png'} and {case / 'front_lifecycle.png'} (hour {snapshot_hour:g})")
    if args.animation:
        video = render_animation(data, case, args.fps)
        if video:
            print(f"saved {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
