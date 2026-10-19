"""Dry, passive-theta experiment on the Apalachee scenario's projected grid.

This is a parametric balanced-vortex and scalar-transport experiment, not an
NWP model. Cyclone pressure and gradient wind are axisymmetric; translation
and weak environmental deformation are separate prescribed transport terms.
Potential temperature is passive, so it does not feed back on pressure/winds
and there is no latent heating or resolved boundary layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import PchipInterpolator

# stormmodel.py computes its legacy cache at import unless explicitly skipped.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_old_skip = os.environ.get("APALACHEE_SKIP_CACHE")
os.environ["APALACHEE_SKIP_CACHE"] = "1"
try:
    import stormmodel as bg
finally:
    if _old_skip is None:
        os.environ.pop("APALACHEE_SKIP_CACHE", None)
    else:
        os.environ["APALACHEE_SKIP_CACHE"] = _old_skip

DX = bg.DX
X, Y = bg.X, bg.Y
m = bg.m
HOURS = np.arange(30.0, 112.0 + 0.25, 0.5, dtype=np.float64)
MODEL_VERSION = "dry-frontogenesis-v1.0"
AIR_DENSITY_KG_M3 = 1.15
THETA_REF_K = 290.0
OMEGA_S = 7.2921159e-5
DIFFUSIVITY_M2_S = 300.0
MAX_DT_S = 120.0
FRAME_SCALE = 100_000.0 * 3.0 * 3600.0
_RMAX = PchipInterpolator(bg.RMAX[:, 0], bg.RMAX[:, 1], extrapolate=True)
_WP_PRESSURE = PchipInterpolator(bg.WP[:, 0], bg.WP[:, 3], extrapolate=True)
_Y_MEAN = float(np.nanmean(Y))
_WEDGE_X, _WEDGE_Y = bg.m(-82.9, 34.3)


@dataclass(frozen=True)
class Config:
    """The only case-varying parameter is the prescribed RMW multiplier."""

    radius_scale: float = 2.5

    def __post_init__(self) -> None:
        if self.radius_scale not in (2.5, 3.0):
            raise ValueError("radius_scale must be 2.5 or 3.0")


def _center(t: float) -> tuple[float, float, float]:
    lon, lat = bg.center_ll(t)
    cx, cy = bg.m(lon, lat)
    return float(cx), float(cy), float(lat)


def _track_velocity(t: float) -> tuple[float, float]:
    """Track translation in m/s from a centered one-hour difference."""
    t0 = max(float(bg.th[0]), t - 0.5)
    t1 = min(float(bg.th[-1]), t + 0.5)
    x0, y0, _ = _center(t0)
    x1, y1, _ = _center(t1)
    seconds = (t1 - t0) * 3600.0
    return (x1 - x0) / seconds, (y1 - y0) / seconds


def flow(t: float, config: Config) -> dict[str, np.ndarray | float]:
    """Return pressure-only circular gradient balance plus transport winds.

    ``dpdr_pa_m`` and ``gradient_wind_ms`` describe the balanced circular
    component only. ``u_ms``/``v_ms`` add the separately prescribed storm
    translation and a weak, localized environmental strain for scalar
    transport. The projection is treated as Cartesian on the 3-km LCC grid.
    """
    t = float(np.clip(t, bg.th[0], bg.th[-1]))
    cx, cy, lat = _center(t)
    dx = X.astype(np.float64) - cx
    dy = Y.astype(np.float64) - cy
    radius = np.hypot(dx, dy)
    r_safe = np.maximum(radius, 1.0)
    rmax_m = float(_RMAX(t)) * 1000.0 * config.radius_scale
    pc_mb = float(_WP_PRESSURE(t))
    dp_pa = (1012.0 - pc_mb) * 100.0
    exponent = 1.0
    ratio = rmax_m / r_safe
    core = np.exp(-np.power(ratio, exponent))
    p_mb = pc_mb + (dp_pa / 100.0) * core
    dpdr = dp_pa * exponent * np.power(rmax_m, exponent) * np.power(r_safe, -exponent - 1.0) * core
    f = 2.0 * OMEGA_S * np.sin(np.deg2rad(lat))
    gradient_wind = np.sqrt(np.square(0.5 * f * radius) + (radius / AIR_DENSITY_KG_M3) * dpdr) - 0.5 * f * radius
    gradient_wind = np.maximum(gradient_wind, 0.0)
    tangential_u = np.divide(-gradient_wind * dy, radius, out=np.zeros_like(radius), where=radius > 0)
    tangential_v = np.divide(gradient_wind * dx, radius, out=np.zeros_like(radius), where=radius > 0)

    # Prescribed storm translation and weak normal strain are transport terms;
    # they are not included in the radial gradient-wind balance.
    trans_u, trans_v = _track_velocity(t)
    envelope = np.exp(-np.square(radius / 650_000.0))
    strain_s = 1.5e-5
    deformation_u = -strain_s * dx * envelope
    deformation_v = strain_s * dy * envelope
    u = tangential_u + trans_u + deformation_u
    v = tangential_v + trans_v + deformation_v
    return {
        "p_mb": p_mb.astype(np.float32),
        "u_ms": u.astype(np.float32),
        "v_ms": v.astype(np.float32),
        "gradient_wind_ms": gradient_wind.astype(np.float32),
        "dpdr_pa_m": dpdr.astype(np.float32),
        "radius_m": radius.astype(np.float32),
        "f_s": float(f),
        "theta_ref_K": THETA_REF_K,
        "rmax_m": float(rmax_m),
        "translation_u_ms": float(trans_u),
        "translation_v_ms": float(trans_v),
    }


def _ambient_theta(x: np.ndarray, y: np.ndarray, t: float, config: Config) -> np.ndarray:
    """Analytic open-boundary environment: broad baroclinicity and cold wedge."""
    # 0.55 K/100 km meridional gradient, with a weak fixed continental wedge.
    theta = THETA_REF_K - 5.5e-6 * (y - _Y_MEAN)
    wedge = np.exp(-np.square((x - _WEDGE_X) / 320_000.0)
                   - np.square((y - _WEDGE_Y) / 180_000.0))
    return (theta - 2.0 * wedge).astype(np.float32)


def initial_theta(t: float, config: Config) -> np.ndarray:
    """Initial passive potential temperature with open warm/cold-sector fronts."""
    cx, cy, _ = _center(t)
    dx = X.astype(np.float64) - cx
    dy = Y.astype(np.float64) - cy
    theta = _ambient_theta(X, Y, t, config).astype(np.float64)
    # A 6-K NW-to-SE frontal contrast, 55-km half-width; warm air is southeast.
    signed_normal_m = (dx - dy) / np.sqrt(2.0)
    front = 3.0 * np.tanh(signed_normal_m / 55_000.0)
    warm_sector = 1.3 * np.exp(-np.square((dx - 220_000.0) / 360_000.0)
                                - np.square((dy + 120_000.0) / 260_000.0))
    cold_wedge = -1.6 * np.exp(-np.square((dx + 180_000.0) / 320_000.0)
                               - np.square((dy - 220_000.0) / 190_000.0))
    return (theta + front + warm_sector + cold_wedge).astype(np.float32)


def frontogenesis(theta: np.ndarray, u: np.ndarray, v: np.ndarray,
                  dx: float = DX) -> np.ndarray:
    """Signed 2-D kinematic frontogenesis in K m^-1 s^-1."""
    theta = np.asarray(theta, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ty, tx = np.gradient(theta, dx, dx, edge_order=1)
    uy, ux = np.gradient(u, dx, dx, edge_order=1)
    vy, vx = np.gradient(v, dx, dx, edge_order=1)
    grad = np.hypot(tx, ty)
    numerator = -(tx * tx * ux + tx * ty * (uy + vx) + ty * ty * vy)
    out = np.divide(numerator, grad, out=np.zeros_like(numerator), where=grad > 1e-14)
    return out.astype(np.float32)


def _sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return map_coordinates(field, [y / DX, x / DX], order=1, mode="nearest", prefilter=False)


def advect_theta(theta: np.ndarray, t: float, dt_s: float, config: Config,
                 *, ambient_boundary: bool = True) -> np.ndarray:
    """One monotone bilinear semi-Lagrangian step with an RK2 trajectory.

    ``t`` is the beginning of the interval. The 120-s limit keeps trajectory
    motion adequately resolved despite semi-Lagrangian stability. The public
    helper uses an analytic ambient inflow boundary by default. Set
    ``ambient_boundary=False`` for closed-edge manufactured scalar checks.
    """
    if not (0.0 < dt_s <= MAX_DT_S + 1e-9):
        raise ValueError(f"dt_s must be in (0, {MAX_DT_S:g}]")
    a = np.asarray(theta, dtype=np.float32)
    if a.shape != X.shape:
        raise ValueError(f"theta must have grid shape {X.shape}, got {a.shape}")
    if not ambient_boundary and float(np.ptp(a)) < 1e-7:
        return np.full_like(a, float(a.flat[0]))
    mid = flow(t + dt_s / 7200.0, config)
    u0 = np.asarray(mid["u_ms"])
    v0 = np.asarray(mid["v_ms"])
    # Midpoint estimate of velocity along the backward trajectory.
    xm = X - 0.5 * dt_s * u0
    ym = Y - 0.5 * dt_s * v0
    um = _sample(u0, xm, ym)
    vm = _sample(v0, xm, ym)
    xd = X - dt_s * um
    yd = Y - dt_s * vm
    advected = _sample(a, xd, yd).astype(np.float32)
    outside = (xd < X[0, 0]) | (xd > X[0, -1]) | (yd < Y[0, 0]) | (yd > Y[-1, 0])
    if ambient_boundary and outside.any():
        analytic = _ambient_theta(xd[outside], yd[outside], t, config)
        advected[outside] = analytic
    # Explicit 5-point Laplacian: alpha=K*dt/dx^2 <= 0.004 for this step.
    padded = np.pad(advected, 1, mode="edge")
    lap = (padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1]
           + padded[2:, 1:-1] - 4.0 * advected) / (DX * DX)
    return (advected + DIFFUSIVITY_M2_S * dt_s * lap).astype(np.float32)


def surface_gust(t: float, theta: np.ndarray,
                 flowdict: dict[str, np.ndarray | float],
                 config: Config) -> dict[str, np.ndarray]:
    """Inherited bulk-mixing closure with stability from transported theta."""
    theta = np.asarray(theta, dtype=np.float32)
    # The cold anomaly from the synoptic ambient field raises stability and
    # reduces mixed depth. This is a prescribed dry surface closure.
    ambient = _ambient_theta(X, Y, t, config)
    cold_anomaly = np.maximum(ambient - theta, 0.0)
    stable = bg.smooth(0.8, 4.0, cold_anomaly)
    inversion = 1.5 + 5.0 * stable
    depth = np.maximum(180.0, 900.0 - 550.0 * stable)
    speed = np.hypot(flowdict["u_ms"], flowdict["v_ms"])
    gust_ms, mean_ms, _ = bg.mixing_gust(speed, inversion, depth,
                                          exposure_factor=bg.exposure)
    return {
        "gust_mph": np.maximum(gust_ms * bg.MS2MPH, 5.0).astype(np.float32),
        "mean_mph": (mean_ms * bg.MS2MPH).astype(np.float32),
    }


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    # bg.signature() includes the parent source and optional aligned terrain.npz.
    digest.update(bg.signature().encode("ascii"))
    for path in (_HERE / "model.py", _ROOT / "station_catalog.py"):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


SOURCE_FINGERPRINT = _source_fingerprint()


def _stations() -> tuple[list[str], np.ndarray, np.ndarray]:
    from station_catalog import ST
    ids = [row[0] for row in ST]
    xy = np.asarray([bg.m(row[3], row[2]) for row in ST], dtype=float)
    return ids, xy[:, 0], xy[:, 1]


def simulate(config: Config, outdir: str | os.PathLike[str], *,
             start_hour: float = 12.0, end_hour: float = 112.0,
             dt_s: float = MAX_DT_S, progress: bool = True) -> dict[str, object]:
    """Integrate theta and write fields, peak, station series, and provenance.

    The default stores 165 half-hour fields (30--112 h), evaluates peak gust
    every two minutes, and samples 69 stations every six minutes. Shorter
    ``start_hour``/``end_hour`` intervals are supported for smoke runs.
    """
    if dt_s != MAX_DT_S:
        raise ValueError(f"full model uses dt_s={MAX_DT_S:g} seconds")
    if not (bg.th[0] <= start_hour < end_hour <= 112.0):
        raise ValueError("require bg.th[0] <= start_hour < end_hour <= 112")
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = SOURCE_FINGERPRINT
    theta = initial_theta(start_hour, config)
    station_ids, station_x, station_y = _stations()
    station_hours = np.arange(start_hour, end_hour + 1e-6, 0.1, dtype=np.float64)
    station_g = np.zeros((len(station_hours), len(station_ids)), dtype=np.float32)
    station_m = np.zeros_like(station_g)
    stored_hours = HOURS[(HOURS >= start_hour - 1e-8) & (HOURS <= end_hour + 1e-8)]
    nframes = len(stored_hours)
    shape3 = (nframes,) + X.shape
    fields = {
        "hours": stored_hours.astype(np.float32),
        "theta_K": np.empty(shape3, np.float32),
        "F_K_100km_3h": np.empty(shape3, np.float32),
        "gust_mph": np.empty(shape3, np.float16),
        "p_mb": np.empty(shape3, np.float16),
        "u_ms": np.empty(shape3, np.float32),
        "v_ms": np.empty(shape3, np.float32),
    }
    peak_gust = np.zeros(X.shape, dtype=np.float32)
    peak_time = np.full(X.shape, float(start_hour), dtype=np.float32)
    station_index = frame_index = 0
    total_steps = int(round((end_hour - start_hour) * 3600.0 / dt_s))
    report_every = max(1, int(round(6.0 * 3600.0 / dt_s)))

    for step in range(total_steps + 1):
        t = start_hour + step * dt_s / 3600.0
        fl = flow(t, config)
        sf = surface_gust(t, theta, fl, config)
        gust = sf["gust_mph"]
        update = gust > peak_gust
        peak_gust[update] = gust[update]
        peak_time[update] = t

        # Six-minute samples are every third 120-s step.
        if step % 3 == 0 and station_index < len(station_hours):
            station_g[station_index] = _sample(gust, station_x, station_y)
            station_m[station_index] = _sample(sf["mean_mph"], station_x, station_y)
            station_index += 1

        if frame_index < nframes and abs(t - float(stored_hours[frame_index])) < 1e-7:
            fields["theta_K"][frame_index] = theta
            fields["F_K_100km_3h"][frame_index] = frontogenesis(theta, fl["u_ms"], fl["v_ms"]) * FRAME_SCALE
            fields["gust_mph"][frame_index] = gust.astype(np.float16)
            fields["p_mb"][frame_index] = fl["p_mb"].astype(np.float16)
            fields["u_ms"][frame_index] = fl["u_ms"]
            fields["v_ms"][frame_index] = fl["v_ms"]
            frame_index += 1

        if step == total_steps:
            break
        theta = advect_theta(theta, t, dt_s, config, ambient_boundary=True)
        if progress and step > 0 and step % report_every == 0:
            print(f"{t:.0f} h / {end_hour:.0f} h", flush=True)

    # Trim station output to actual samples for a non-aligned short interval.
    station_hours = station_hours[:station_index]
    station_g = station_g[:station_index]
    station_m = station_m[:station_index]
    theta_range = ([float(fields["theta_K"].min()), float(fields["theta_K"].max())]
                   if nframes else [None, None])
    f_range = ([float(fields["F_K_100km_3h"].min()), float(fields["F_K_100km_3h"].max())]
               if nframes else [None, None])
    pressure_min = float(min(bg.pressure(start_hour + k * dt_s / 3600.0)
                             for k in range(total_steps + 1)))
    metadata = {
        "simulated": True,
        "model_version": MODEL_VERSION,
        "source_fingerprint": fingerprint,
        "source_changes_invalidate_outputs": True,
        "config": asdict(config),
        "projection": "parent Lambert conformal grid treated as Cartesian; DX=3000 m",
        "pressure": "original WP central pressure, no size multiplier; B=1; dp=1012-pc",
        "transport": "pressure-only gradient wind + prescribed track translation + 1.5e-5 s-1 localized strain",
        "theta": "dry passive potential temperature; no two-way dynamics or latent heating",
        "initial_front_geometry": "single idealized NW-SE baroclinic zone plus localized warm-sector tongue",
        "theta_initial_time_hour": float(start_hour),
        "dt_s": float(dt_s),
        "integration_start_hour": float(start_hour),
        "integration_end_hour": float(end_hour),
        "integration_steps": int(total_steps),
        "diffusivity_m2_s": DIFFUSIVITY_M2_S,
        "stored_theta_dtype": "float32 to retain resolved gradients",
        "stored_wind_dtype": "float32 to reproduce archived frontogenesis",
        "stored_frontogenesis_dtype": "float32; signed K per 100 km per 3 h",
        "pressure_min_mb": pressure_min,
        "peak_gust_max_mph": float(peak_gust.max()),
        "peak_land_max_mph": float(peak_gust[bg.land].max()),
        "stored_theta_minmax_K": theta_range,
        "stored_frontogenesis_minmax_K_100km_3h": f_range,
        "station_ids": station_ids,
    }
    fields["source_fingerprint"] = np.asarray(fingerprint)
    fields["model_version"] = np.asarray(MODEL_VERSION)
    np.savez_compressed(out / "fields.npz", **fields)
    np.savez_compressed(out / "peak.npz", G=peak_gust.astype(np.float32), T=peak_time.astype(np.float32),
                        source_fingerprint=np.asarray(fingerprint), model_version=np.asarray(MODEL_VERSION))
    np.savez_compressed(out / "station_series.npz", hours=station_hours.astype(np.float32),
                        gust_mph=station_g, mean_mph=station_m,
                        station_ids=np.asarray(station_ids), source_fingerprint=np.asarray(fingerprint),
                        model_version=np.asarray(MODEL_VERSION))
    with (out / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, choices=(2.5, 3.0), default=2.5)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--start-hour", type=float, default=12.0, help=argparse.SUPPRESS)
    parser.add_argument("--end-hour", type=float, default=112.0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(f"Starting {MODEL_VERSION}, radius scale {args.scale:g}x", flush=True)
    meta = simulate(Config(args.scale), args.out,
                    start_hour=args.start_hour, end_hour=args.end_hour)
    print(f"Wrote outputs; source fingerprint {meta['source_fingerprint'][:12]}", flush=True)


if __name__ == "__main__":
    main()
