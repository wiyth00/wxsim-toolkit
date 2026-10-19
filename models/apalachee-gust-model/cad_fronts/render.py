#!/usr/bin/env python3
"""Render reproducible CAD scenario maps and animations from validated archives."""

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
from matplotlib.colors import Normalize, TwoSlopeNorm
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
    raise AttributeError("cad_fronts.model must expose its Basemap as `m` or `bg.m`")

ACCENT = "#9b1c63"
THETA_CMAP = "RdYlBu_r"
F_LEVEL = 4.0
F_LEVELS = np.arange(-F_LEVEL, F_LEVEL + 0.001, 0.5)
F_TICKS = np.arange(-4, 4.01, 1)
THETA_LEVELS = np.array([267, 270, 273, 276, 280, 284, 288, 292, 296, 300], dtype=float)
THETA_NORM = Normalize(vmin=265, vmax=302, clip=False)
TIMES_LIFECYCLE = (36, 54, 78, 108)
FRAME_DPI = 115
CITY_LABELS = (("Atlanta", -84.3880, 33.7490), ("Athens", -83.3780, 33.9519),
               ("Gainesville", -83.8241, 34.2979), ("Charlotte", -80.8431, 35.2271))
CAD_STATIONS = (("Columbus", -84.9877, 32.4609), ("Macon", -83.6324, 32.8407),
                ("Augusta", -81.9748, 33.4735), ("Columbia", -81.0348, 34.0007))


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
    for key, value in fields.items():
        if key != "hours" and isinstance(value, np.ndarray) and value.shape == expected:
            continue
        if key.endswith(("_ms", "_m", "_K", "_F", "_mb")) and isinstance(value, np.ndarray) and value.ndim == 3 and value.shape != expected:
            raise ValueError(f"optional field {key} has shape {value.shape}, expected {expected}")
    stations_path = case / "station_series.npz"
    station_series = {}
    if stations_path.is_file():
        with np.load(stations_path) as z:
            station_series = {k: z[k] for k in z.files}
    return {"case": case, "fields": fields, "peak": peak, "station_series": station_series}


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


def _field(fields: dict, *names):
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _frame_at(value: np.ndarray, index: int) -> np.ndarray:
    a = np.asarray(value)
    return a[index] if a.ndim == 3 else a


def _surface_temperature_f(fields: dict, index: int) -> np.ndarray:
    saved = _field(fields, "surface_T_F", "surface_temp_F", "surface_temperature_F")
    if saved is not None:
        return np.asarray(_frame_at(saved, index), dtype=float)
    theta = np.asarray(fields["theta_K"][index], dtype=float)
    pressure = np.asarray(fields["p_mb"][index], dtype=float)
    temp_k = theta * np.power(np.maximum(pressure, 1.0) / 1000.0, 0.2854)
    return (temp_k - 273.15) * 9.0 / 5.0 + 32.0


def _cad_mask(fields: dict) -> np.ndarray:
    saved = _field(fields, "cad_mask", "cad_mask_static", "cad_corridor_mask")
    if saved is not None:
        a = np.asarray(saved)
        return a[0] if a.ndim == 3 else a
    fn = getattr(fm, "cad_mask", None)
    return np.asarray(fn(fm.X, fm.Y)) if fn else np.zeros(np.shape(fm.X), dtype=float)


def _draw_static_overlays(ax, fields: dict):
    """Geographic context shared by every case and every rendered frame."""
    mask = _cad_mask(fields)
    if np.isfinite(mask).any() and float(np.nanmax(mask)) > 0:
        ax.contour(fm.X, fm.Y, mask, levels=[0.22, 0.50], colors=["#186f79", "#074b55"],
                   linewidths=[1.1, 0.8], linestyles=["--", "-"], zorder=6)
    for name, lon, lat in CITY_LABELS:
        x, y = MAP(lon, lat)
        ax.plot(x, y, marker="o", ms=2.5, color="#202631", zorder=8)
        ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                    color="#202631", weight="semibold", zorder=9,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72})
    # The high center is deliberately off-domain: anchor the note at the NE map edge.
    ax.annotate("1049 mb high · SW Quebec\n(center off map)", xy=(0.98, 0.96), xycoords="axes fraction",
                xytext=(0.66, 0.88), textcoords="axes fraction", fontsize=6.5, ha="right", va="top",
                color="#34333a", arrowprops={"arrowstyle": "->", "lw": 0.75, "color": "#34333a"},
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#92949a", "alpha": 0.88}, zorder=10)


def _wind_arrows(ax, fields: dict, index: int):
    """Sparse lower-level arrows show the schematic barrier-parallel CAD jet."""
    u = _field(fields, "cad_u_ms", "cad_jet_u_ms", "u_cad_ms", "u_lower_ms", "lower_u_ms", "u_ms")
    v = _field(fields, "cad_v_ms", "cad_jet_v_ms", "v_cad_ms", "v_lower_ms", "lower_v_ms", "v_ms")
    if u is None or v is None:
        return None
    step = max(1, min(fm.X.shape) // 22)
    mask = _cad_mask(fields) > 0.18
    sl = (slice(None, None, step), slice(None, None, step))
    valid = mask[sl]
    if not valid.any():
        return None
    return ax.quiver(fm.X[sl][valid], fm.Y[sl][valid], np.asarray(u[index])[sl][valid] * 5000.0,
                     np.asarray(v[index])[sl][valid] * 5000.0, color="#111820", alpha=0.78,
                     angles="xy", scale_units="xy", scale=1, width=0.0022, headwidth=3.5, zorder=7)


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
        _draw_static_overlays(ax_g, fields)
    gust_cf = ax_g.contourf(fm.X, fm.Y, gust, levels=GUST_LEVELS, colors=GUST_COLORS, extend="max", alpha=0.9, zorder=2)
    p_levels = np.arange(936, 1050, 4)
    cs = ax_g.contour(fm.X, fm.Y, pressure, levels=p_levels, colors="#34333a", linewidths=0.62, alpha=0.9, zorder=4)
    p_labels = ax_g.clabel(cs, inline=True, fontsize=6, fmt="%d", colors="#34333a")
    q = _wind_arrows(ax_g, fields, index)
    ax_g.set_title("Surface gust proxy + total pressure + lower-layer CAD flow", loc="left", fontsize=10, color=INK, pad=7, weight="semibold")

    if draw_background:
        _map_background(ax_f)
        _draw_static_overlays(ax_f, fields)
    cf = ax_f.contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(), norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", alpha=0.96, zorder=2)
    tlevels = THETA_LEVELS
    tc = ax_f.contour(fm.X, fm.Y, np.ma.masked_where(~_interior_mask(theta.shape), theta), levels=tlevels,
                      colors="#252932", linewidths=0.46, alpha=0.8, zorder=4)
    theta_labels = ax_f.clabel(tc, inline=True, fontsize=5.7, fmt="%.0f", colors="#252932")
    ax_f.set_title("Signed kinematic frontogenesis + θ contours (K)", loc="left", fontsize=10, color=INK, pad=7, weight="semibold")
    return h, gust_cf, cf, [gust_cf, cs, cf, tc, *p_labels, *theta_labels] + ([q] if q is not None else [])


def _case_name(data: dict) -> str:
    names = {"size_2p5x": "2.5× length scale", "size_3x": "3× length scale"}
    return names.get(data["case"].name, data["case"].name.replace("_", " "))


def _title(fig, data: dict, hour: float, suffix: str = ""):
    fig.text(0.035, 0.976, f"Appalachian CAD corridor · {_case_name(data)}{suffix}",
             ha="left", va="top", fontsize=15, weight="bold", color=INK)
    fig.text(0.965, 0.949, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2,
             weight="bold", color=ACCENT)
    fig.text(0.035, 0.949, f"Dec 15–19 fictional scenario · model hour {hour:g} · Quebec high 1049 mb off-domain; cyclone minimum 939 mb",
             ha="left", va="top", fontsize=8.5, color=MUTED)


def _add_footer(fig, left=""):
    note = "Signed kinematic F: + strengthens |∇θ|; − weakens it. CAD outline/flow are schematic; no resolved terrain. Scenario, not a forecast."
    fig.text(0.035, 0.016, (left + "   ·   " if left else "") + note,
             ha="left", va="bottom", fontsize=7, color=MUTED)


def render_snapshot(data: dict, out: Path, requested_hour: float = 48.0):
    fields = data["fields"]
    i = _time_index(fields["hours"], requested_hour)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), dpi=170, facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.89, bottom=0.19, wspace=0.08)
    h, gust_cf, cf, _ = _draw_time_axes(*axes, data, i)
    _title(fig, data, h)
    cbar = fig.colorbar(cf, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]), orientation="horizontal", ticks=F_TICKS)
    cbar.set_label("Kinematic frontogenesis  (K per 100 km per 3 h)", fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=7, length=2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Instantaneous gust proxy (mph); arrows: lower-layer transport", ha="center", fontsize=8, color=INK)
    fig.text(0.745, 0.105, "Signed kinematic F over θ (K) contours", ha="center", fontsize=8, color=INK)
    fig.text(0.50, 0.049, "Total-pressure contours: 4 mb interval · 1049 mb Quebec high center is off map", ha="center", fontsize=7.5, color=MUTED)
    _add_footer(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return h


def render_cad_detail(data: dict, out: Path, requested_hour: float = 48.0):
    """Render CAD fields, recomputing lightweight diagnostics from this validated archive."""
    fields = data["fields"]
    i = _time_index(fields["hours"], requested_hour)
    depth = _field(fields, "cad_depth_m", "depth_m", "mixing_depth_m")
    inversion = _field(fields, "cad_inversion_K", "inversion_K", "inversion_strength_K")
    upper_u = _field(fields, "upper_u_ms", "u_upper_ms")
    upper_v = _field(fields, "upper_v_ms", "v_upper_ms")
    lower_u = _field(fields, "lower_u_ms", "u_lower_ms", "cad_lower_u_ms", "u_ms")
    lower_v = _field(fields, "lower_v_ms", "v_lower_ms", "cad_lower_v_ms", "v_ms")
    if any(value is None for value in (depth, inversion, upper_u, upper_v, lower_u, lower_v)):
        summary = json.loads((data["case"] / "summary.json").read_text())
        config = summary.get("config")
        if not isinstance(config, dict) or "radius_scale" not in config:
            raise ValueError(f"{data['case'] / 'summary.json'} lacks the run configuration needed for CAD diagnostics")
        cfg = fm.Config(**config)
        flow = fm.flow(float(fields["hours"][i]), cfg)
        sf = fm.surface_gust(float(fields["hours"][i]), fields["theta_K"][i], flow, cfg)
        depth = depth if depth is not None else sf.get("mixing_depth_m")
        inversion = inversion if inversion is not None else sf.get("inversion_K")
        upper_u = upper_u if upper_u is not None else flow.get("upper_u_ms")
        upper_v = upper_v if upper_v is not None else flow.get("upper_v_ms")
        lower_u = lower_u if lower_u is not None else flow.get("u_ms")
        lower_v = lower_v if lower_v is not None else flow.get("v_ms")
    if any(value is None for value in (depth, inversion, upper_u, upper_v, lower_u, lower_v)):
        return False
    theta = np.asarray(fields["theta_K"][i], dtype=float)
    temp_f = _surface_temperature_f(fields, i)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.3), dpi=160, facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.18, wspace=0.08)
    for ax in axes:
        _map_background(ax)
        _draw_static_overlays(ax, fields)
    temp_levels = np.arange(10, 71, 5)
    cf0 = axes[0].contourf(fm.X, fm.Y, theta, levels=THETA_LEVELS, cmap=THETA_CMAP,
                           norm=THETA_NORM, extend="both", zorder=2)
    tc = axes[0].contour(fm.X, fm.Y, temp_f, levels=temp_levels, colors="#252932", linewidths=0.55,
                         linestyles="--", zorder=4)
    axes[0].clabel(tc, inline=True, fontsize=5.5, fmt="%.0f°F")
    axes[0].set_title("Potential temperature θ (K)\nDashed: sea-level-equivalent dry T proxy (°F)", loc="left", fontsize=9, color=INK)
    for name, lon, lat in CAD_STATIONS:
        sx, sy = MAP(lon, lat)
        iy, ix = np.unravel_index(np.nanargmin((fm.X - sx) ** 2 + (fm.Y - sy) ** 2), fm.X.shape)
        axes[0].plot(sx, sy, marker="o", ms=3.2, mfc="white", mec="#252932", mew=0.8, zorder=9)
        axes[0].annotate(f"{name} proxy {temp_f[iy, ix]:.0f}°F", (sx, sy), xytext=(4, 3), textcoords="offset points",
                         fontsize=6.3, color="#202631", zorder=10,
                         bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.84})
    cad_region = _cad_mask(fields) > 0.12
    depth2d = np.ma.masked_where(~cad_region, np.asarray(_frame_at(depth, i), dtype=float))
    inv2d = np.ma.masked_where(~cad_region, np.asarray(_frame_at(inversion, i), dtype=float))
    cf1 = axes[1].contourf(fm.X, fm.Y, depth2d, levels=np.arange(450, 851, 25), cmap="YlGnBu_r", extend="both", zorder=2)
    dc = axes[1].contour(fm.X, fm.Y, inv2d, levels=np.arange(6, 11, 1), colors="#632c55", linewidths=0.65, zorder=4)
    axes[1].clabel(dc, inline=True, fontsize=5.7, fmt="%.0f K")
    axes[1].set_title("Prescribed CAD mixing-depth proxy (m)\nInversion strength contours (K); CAD mask only", loc="left", fontsize=9, color=INK)
    step = max(1, min(fm.X.shape) // 23)
    sl = (slice(None, None, step), slice(None, None, step))
    msk = _cad_mask(fields)[sl] > 0.12
    xq, yq = fm.X[sl][msk], fm.Y[sl][msk]
    q1 = axes[2].quiver(xq, yq, _frame_at(upper_u, i)[sl][msk] * 5000.0, _frame_at(upper_v, i)[sl][msk] * 5000.0,
                        color="#173b66", angles="xy", scale_units="xy", scale=1, width=0.0024, zorder=6)
    q2 = axes[2].quiver(xq, yq, _frame_at(lower_u, i)[sl][msk] * 5000.0,
                        _frame_at(lower_v, i)[sl][msk] * 5000.0,
                        color="#c44931", angles="xy", scale_units="xy", scale=1, width=0.0024, alpha=0.82, zorder=7)
    axes[2].set_title("Corridor flow · blue upper, red lower layer\nLower arrows are not 10-m surface winds", loc="left", fontsize=9, color=INK)
    axes[2].quiverkey(q1, 0.80, 0.06, 50_000, "10 m/s upper flow", labelpos="E", coordinates="axes", fontproperties={"size": 6.5})
    axes[2].quiverkey(q2, 0.80, 0.01, 50_000, "10 m/s lower layer", labelpos="E", coordinates="axes", fontproperties={"size": 6.5})
    _title(fig, data, float(fields["hours"][i]), " · CAD detail")
    fig.colorbar(cf0, ax=axes[0], orientation="horizontal", fraction=0.055, pad=0.11, label="Potential temperature θ (K)")
    fig.colorbar(cf1, ax=axes[1], orientation="horizontal", fraction=0.055, pad=0.11, label="Prescribed CAD mixing-depth proxy (m)")
    fig.text(0.035, 0.035, "Dry T proxy uses total model pressure; not station pressure, a 2-m forecast, or a freezing-rain diagnostic. Inversion/depth are parameterized.",
             ha="left", va="bottom", fontsize=7, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


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
        _map_background(axes[0, col]); _draw_static_overlays(axes[0, col], fields)
        _map_background(axes[1, col]); _draw_static_overlays(axes[1, col], fields)
        last_cf = axes[0, col].contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(),
                                        norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", zorder=2)
        tc = axes[0, col].contour(fm.X, fm.Y, theta, levels=THETA_LEVELS, colors="#202631", linewidths=0.45, zorder=4)
        axes[0, col].clabel(tc, inline=True, fontsize=5.2, fmt="%.0f")
        axes[0, col].set_title(f"Hour {h:g}", fontsize=9, color=INK, weight="semibold")

        tcf = axes[1, col].contourf(fm.X, fm.Y, theta, levels=THETA_LEVELS, cmap=THETA_CMAP,
                                    norm=THETA_NORM, extend="both", zorder=2)
        axes[1, col].contour(fm.X, fm.Y, theta, levels=THETA_LEVELS, colors="#252932", linewidths=0.35, zorder=4)
        axes[1, col].set_title("Potential temperature θ (K)", fontsize=8, color=INK)
    _title(fig, data, actual[1], " · lifecycle")
    cb = fig.colorbar(last_cf, ax=axes[0, :].tolist(), orientation="horizontal", fraction=0.06, pad=0.08, ticks=F_TICKS)
    cb.set_label("Kinematic frontogenesis  (K per 100 km per 3 h); blue = frontolysis, orange = strengthening", fontsize=8)
    cb2 = fig.colorbar(tcf, ax=axes[1, :].tolist(), orientation="horizontal", fraction=0.06, pad=0.08,
                       ticks=[267, 273, 280, 288, 296, 302])
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
    cbar.set_label("Kinematic frontogenesis  (K per 100 km per 3 h)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Instantaneous gust proxy (mph); lower-layer arrows", ha="center", fontsize=8)
    fig.text(0.745, 0.105, "Signed kinematic F (K / 100 km / 3 h)", ha="center", fontsize=8)
    fig.text(0.50, 0.049, "Total-pressure contours: 4 mb · Quebec high center off map", ha="center", fontsize=7.5, color=MUTED)
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
            title_text = fig.text(0.035, 0.976, f"Appalachian CAD corridor · {_case_name(data)}",
                                  ha="left", va="top", fontsize=15, weight="bold", color=INK)
            time_text = fig.text(0.035, 0.949, "", ha="left", va="top", fontsize=8.5, color=MUTED)
            fig.text(0.965, 0.949, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2, weight="bold", color=ACCENT)
            draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
            front_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), cmap=_front_cmap())
            cbar = fig.colorbar(front_sm, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]),
                                orientation="horizontal", ticks=F_TICKS, extend="both")
            cbar.ax.tick_params(labelsize=7, length=2)
            fig.text(0.255, 0.105, "Instantaneous gust proxy (mph); arrows: lower-layer transport", ha="center", fontsize=8)
            fig.text(0.745, 0.105, "Signed kinematic F over θ (K) contours", ha="center", fontsize=8)
            fig.text(0.50, 0.049, "Total pressure contours · Quebec 1049 mb high center off map", ha="center", fontsize=7.5, color=MUTED)
            _add_footer(fig)
        time_text.set_text(f"Dec 15–19 fictional scenario · hour {h:g} · Quebec high 1049 mb off-domain; cyclone minimum 939 mb")
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


def render_comparison(cases: list[dict], out: Path, hour: float = 48.0):
    if len(cases) != 2:
        raise ValueError("comparison requires exactly two cases")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 14.0), dpi=165, facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.98, top=0.89, bottom=0.19, hspace=0.16, wspace=0.08)
    last_cf = None
    for row, data in enumerate(cases):
        peak = np.ma.masked_where(~_interior_mask(data["peak"]["G"].shape), data["peak"]["G"])
        _map_background(axes[row, 0]); _draw_static_overlays(axes[row, 0], data["fields"])
        axes[row, 0].contourf(fm.X, fm.Y, peak, levels=GUST_LEVELS, colors=GUST_COLORS, extend="max", alpha=0.92, zorder=2)
        axes[row, 0].set_title(f"{_case_name(data)} · peak gust {data['peak']['G'].max():.1f} mph", loc="left", fontsize=10, color=INK, weight="semibold")
        fields = data["fields"]
        i = _time_index(fields["hours"], hour)
        front = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["F_K_100km_3h"][i])
        theta = np.ma.masked_where(~_interior_mask(fields["theta_K"].shape[-2:]), fields["theta_K"][i])
        _map_background(axes[row, 1]); _draw_static_overlays(axes[row, 1], fields)
        last_cf = axes[row, 1].contourf(fm.X, fm.Y, front, levels=F_LEVELS, cmap=_front_cmap(),
                                        norm=TwoSlopeNorm(vmin=-F_LEVEL, vcenter=0, vmax=F_LEVEL), extend="both", zorder=2)
        tc = axes[row, 1].contour(fm.X, fm.Y, theta, levels=THETA_LEVELS, colors="#202631", linewidths=0.45, zorder=4)
        axes[row, 1].clabel(tc, inline=True, fontsize=5.3, fmt="%.0f")
        axes[row, 1].set_title(f"Hour {fields['hours'][i]:g} · kinematic F + θ (K)", loc="left", fontsize=10, color=INK, weight="semibold")
    fig.text(0.035, 0.974, "Appalachian CAD corridor · length-scale sensitivity", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.965, 0.974, "SIMULATED SCENARIO", ha="right", va="top", fontsize=8.2, weight="bold", color=ACCENT)
    fig.text(0.035, 0.949, "Quebec high 1049 mb off-domain · 939 mb cyclone minimum · scale changes pressure-profile length.",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    cb = fig.colorbar(last_cf, cax=fig.add_axes([0.57, 0.078, 0.35, 0.016]), orientation="horizontal", ticks=F_TICKS)
    cb.set_label("Kinematic frontogenesis (K per 100 km per 3 h); blue = frontolysis, orange = strengthening", fontsize=7.2)
    draw_cbar(fig, [0.08, 0.078, 0.35, 0.016])
    fig.text(0.255, 0.105, "Maximum gust proxy (mph)", ha="center", fontsize=8)
    fig.text(0.745, 0.105, "Kinematic frontogenesis at requested hour", ha="center", fontsize=8)
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
    if runtime_sig is None or str(runtime_sig) != field_sig:
        raise ValueError(f"saved output is stale for the current model source: {data['case']}")
    runtime_version = str(getattr(fm, "MODEL_VERSION", ""))
    if not runtime_version or str(summary.get("model_version", "")) != runtime_version:
        raise ValueError(f"summary model_version does not match current CAD model in {data['case']}")
    for archive_name, archive in (("fields.npz", data["fields"]), ("peak.npz", data["peak"])):
        if "model_version" not in archive:
            raise ValueError(f"{data['case'] / archive_name} lacks model_version")
        version = np.asarray(archive["model_version"]).item()
        version = version.decode() if isinstance(version, bytes) else str(version)
        if version != runtime_version:
            raise ValueError(f"{archive_name} model_version does not match current CAD model in {data['case']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, help="one case directory containing fields.npz and peak.npz")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CASE_A", "CASE_B"), help="render 2.5x vs 3x comparison")
    parser.add_argument("--root", type=Path, default=HERE, help="directory for default case lookup and comparison output")
    parser.add_argument("--hour", type=float, default=48.0, help="snapshot hour (default: 48; reference time)")
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
    detail = render_cad_detail(data, case / "cad_detail.png", args.hour)
    files = f"{case / 'front_snapshot.png'}, {case / 'front_lifecycle.png'}"
    if detail:
        files += f", {case / 'cad_detail.png'}"
    print(f"saved {files} (hour {snapshot_hour:g})")
    if args.animation:
        video = render_animation(data, case, args.fps)
        if video:
            print(f"saved {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
