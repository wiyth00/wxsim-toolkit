"""Idealized pressure, CAD transport, and scalar experiment on a projected grid.

The cyclone pressure perturbation is axisymmetric, while an off-domain Quebec
high supplies a broad background pressure gradient. A schematic Appalachian
ridge mask prescribes a shallow cold dome and barrier-parallel transport jet.
This is an idealized parameterization, not NWP: no DEM, resolved terrain,
hydrostatic column, or two-way thermal/dynamic feedback is present.
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
MODEL_VERSION = "schematic-cad-v1.0"
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
CAD_HIGH_LONLAT = (-73.0, 48.0)  # off-domain high center, SW Quebec
CAD_HIGH_X, CAD_HIGH_Y = bg.m(*CAD_HIGH_LONLAT)
CAD_HIGH_CENTER_MB = 1049.0
BACKGROUND_FAR_MB = 1012.0
BACKGROUND_SCALE_M = 1_350_000.0
CAD_AXIS_LONLAT = ((-79.4, 36.7), (-80.7, 35.5), (-82.4, 34.4), (-84.0, 33.3))
_CAD_AXIS_XY = np.asarray([bg.m(lon, lat) for lon, lat in CAD_AXIS_LONLAT], dtype=float)
_CAD_CENTER_X, _CAD_CENTER_Y = bg.m(-83.25, 34.15)
CAD_HALF_LENGTH_M = 470_000.0
CAD_HALF_WIDTH_M = 185_000.0
CAD_THETA_CORE_K = 270.0
CAD_RESTORE_TAU_S = 6.0 * 3600.0
CAD_JET_MIN_MS = 8.0
CAD_JET_MAX_MS = 15.0
CAD_DEPTH_MAX_M = 800.0
CAD_DEPTH_MIN_M = 500.0
CAD_INVERSION_MIN_K = 6.0
CAD_INVERSION_MAX_K = 10.0


@dataclass(frozen=True)
class Config:
    """Case scale plus an explicit CAD on/off experiment switch."""

    radius_scale: float = 2.5
    cad_enabled: bool = True

    def __post_init__(self) -> None:
        if self.radius_scale not in (2.5, 3.0):
            raise ValueError("radius_scale must be 2.5 or 3.0")


def cad_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Schematic east-side Appalachian cold-dome mask, not a DEM barrier.

    The corridor runs NE-to-SW across northern Georgia and the southern
    Appalachians. The Gaussian represents a broad idealized ridge/cold-air
    corridor rather than actual terrain or a drainage model.
    """
    cached = globals().get("_CAD_MASK_CACHE")
    if cached is not None and cached[0] == id(x) and cached[1] == id(y):
        return cached[2]
    dx = np.asarray(x, dtype=np.float64) - _CAD_CENTER_X
    dy = np.asarray(y, dtype=np.float64) - _CAD_CENTER_Y
    along = 0.80 * dx + 0.60 * dy
    cross = -0.60 * dx + 0.80 * dy
    ridge = np.exp(-np.square(along / CAD_HALF_LENGTH_M)
                   - np.square(cross / CAD_HALF_WIDTH_M))
    # Smooth southern/eastern edge near the fall line, approximated by
    # Columbus-Macon-Augusta-Columbia. This geographic gate is schematic.
    lon, lat = m(np.asarray(x), np.asarray(y), inverse=True)
    boundary_lat = 32.7 + 0.28 * (np.asarray(lon) + 83.65)
    south_gate = 0.5 * (1.0 + np.tanh((np.asarray(lat) - boundary_lat) * 111_000.0 / 55_000.0))
    return (ridge * south_gate).astype(np.float32)


_CAD_MASK_CACHE = (id(X), id(Y), cad_mask(X, Y))


def _synoptic_high_pressure(x: np.ndarray | float,
                           y: np.ndarray | float) -> np.ndarray:
    """Broad Quebec-high pressure used for the free-flow wind background."""
    xx, yy = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    radius2 = np.square(xx - CAD_HIGH_X) + np.square(yy - CAD_HIGH_Y)
    return BACKGROUND_FAR_MB + (CAD_HIGH_CENTER_MB - BACKGROUND_FAR_MB) * np.exp(
        -0.5 * radius2 / (BACKGROUND_SCALE_M * BACKGROUND_SCALE_M))


def background_pressure(x: np.ndarray | float, y: np.ndarray | float,
                        t: float = 51.0, config: Config | None = None) -> np.ndarray:
    """Broad Quebec high plus a schematic hydrostatic CAD pressure ridge.

    Accepts arbitrary projected coordinates so the off-domain high can be
    inspected directly. Pressure falls toward the Southeast; this is a
    prescribed field, not an analysis. With CAD disabled only the Quebec high
    remains. The ridge pressure is a first-order hydrostatic estimate.
    """
    del t
    xx, yy = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    high = _synoptic_high_pressure(xx, yy)
    if config is None or config.cad_enabled:
        ambient_theta = THETA_REF_K - 5.5e-6 * (yy - _Y_MEAN)
        # Preserve original grid-array identity so cad_mask can reuse its
        # cached mask instead of repeating inverse projection each flow step.
        mask = cad_mask(x, y).astype(np.float64)
        ridge_depth = CAD_DEPTH_MAX_M
        ridge_mb = (AIR_DENSITY_KG_M3 * 9.81 * ridge_depth
                    * np.maximum(ambient_theta - CAD_THETA_CORE_K, 0.0)
                    / np.maximum(ambient_theta, 1.0) / 100.0)
        high = high + mask * ridge_mb
    return high


def _background_geostrophic_wind(lat: float,
                                 config: Config | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Free-flow geostrophic wind from the broad Quebec high alone.

    The hydrostatic CAD ridge remains in the total near-surface pressure field,
    but its shallow pressure gradient decays with height. The separate
    prescribed barrier jet represents shallow CAD transport; including the
    ridge gradient in this upper/free-flow wind would create unsupported edge
    jets from a single-level pressure bulge.
    """
    del config  # CAD on/off does not change the deep synoptic high.
    pbg = _synoptic_high_pressure(X, Y)
    py, px = np.gradient(pbg * 100.0, DX, DX, edge_order=1)
    f = max(2.0 * OMEGA_S * np.sin(np.deg2rad(lat)), 1.0e-5)
    return (-py / (AIR_DENSITY_KG_M3 * f)).astype(np.float32), \
        (px / (AIR_DENSITY_KG_M3 * f)).astype(np.float32)


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
    """Return circular balance, free-flow wind, and shallow CAD transport.

    ``dpdr_pa_m`` and ``gradient_wind_ms`` describe the balanced circular
    component only. ``upper_u_ms``/``upper_v_ms`` add translation, weak
    deformation, and the Quebec-high-only geostrophic component. The shallow
    hydrostatic CAD ridge is excluded from this upper wind. ``u_ms``/``v_ms``
    are lower-layer transport, blended toward an 8-15 m/s southwestward
    barrier jet inside the schematic CAD corridor; they are not 10-m winds.
    Total pressure is not in gradient balance.
    """
    t = float(np.clip(t, bg.th[0], bg.th[-1]))
    cx, cy, lat = _center(t)
    dx = X.astype(np.float64) - cx
    dy = Y.astype(np.float64) - cy
    radius = np.hypot(dx, dy)
    r_safe = np.maximum(radius, 1.0)
    rmax_m = float(_RMAX(t)) * 1000.0 * config.radius_scale
    pc_mb = float(_WP_PRESSURE(t))
    pbg = background_pressure(X, Y, t=t, config=config)
    pbg_center = float(background_pressure(cx, cy, t=t, config=config))
    dp_pa = (pbg_center - pc_mb) * 100.0
    exponent = 1.0
    ratio = rmax_m / r_safe
    core = np.exp(-np.power(ratio, exponent))
    # Pressure is pc at the center and approaches the local background outward.
    # dpdr below is the positive outward derivative of this circular term.
    p_mb = pbg - (dp_pa / 100.0) * (1.0 - core)
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
    bg_u, bg_v = _background_geostrophic_wind(lat, config)
    upper_u = tangential_u + trans_u + deformation_u + bg_u
    upper_v = tangential_v + trans_v + deformation_v + bg_v
    mask = cad_mask(X, Y) if config.cad_enabled else np.zeros_like(X, dtype=np.float32)
    jet_speed = CAD_JET_MIN_MS + (CAD_JET_MAX_MS - CAD_JET_MIN_MS) * mask
    # NE-to-SW along the schematic barrier.
    jet_u, jet_v = -0.80 * jet_speed, -0.60 * jet_speed
    u = (1.0 - mask) * upper_u + mask * jet_u
    v = (1.0 - mask) * upper_v + mask * jet_v
    return {
        "p_mb": p_mb.astype(np.float32),
        "u_ms": u.astype(np.float32),
        "v_ms": v.astype(np.float32),
        "upper_u_ms": upper_u.astype(np.float32),
        "upper_v_ms": upper_v.astype(np.float32),
        "cad_mask": mask.astype(np.float32),
        "cad_jet_speed_ms": (mask * jet_speed).astype(np.float32),
        "gradient_wind_ms": gradient_wind.astype(np.float32),
        "dpdr_pa_m": dpdr.astype(np.float32),
        "radius_m": radius.astype(np.float32),
        "f_s": float(f),
        "theta_ref_K": THETA_REF_K,
        "rmax_m": float(rmax_m),
        "translation_u_ms": float(trans_u),
        "translation_v_ms": float(trans_v),
        "background_pressure_center_mb": pbg_center,
        "circular_dp_center_pa": float(dp_pa),
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
    # Broad warm-sector air south/east of the CAD edge; prescribed background,
    # not diabatic heating. It keeps the idealized boundary near the fall line.
    warm_x, warm_y = bg.m(-83.5, 31.8)
    southeast_warm = 6.5 * np.exp(-np.square((X - warm_x) / 420_000.0)
                                  -np.square((Y - warm_y) / 300_000.0))
    cold_wedge = -1.6 * np.exp(-np.square((dx + 180_000.0) / 320_000.0)
                               - np.square((dy - 220_000.0) / 190_000.0))
    theta = theta + front + warm_sector + southeast_warm + cold_wedge
    if config.cad_enabled:
        mask = cad_mask(X, Y).astype(np.float64)
        # Analytic initial cold dome; the same target is continuously restored
        # after transport to represent explicit, parameterized cold-air supply.
        theta += mask * (CAD_THETA_CORE_K - theta)
    return theta.astype(np.float32)


def theta_source(theta: np.ndarray, config: Config) -> np.ndarray:
    """Explicit CAD restoring tendency Q in K/s, separate from horizontal F."""
    if not config.cad_enabled:
        return np.zeros_like(theta, dtype=np.float32)
    mask = cad_mask(X, Y).astype(np.float64)
    ambient = _ambient_theta(X, Y, 0.0, config).astype(np.float64)
    target = ambient + mask * (CAD_THETA_CORE_K - ambient)
    source = mask * (target - np.asarray(theta, dtype=np.float64)) / CAD_RESTORE_TAU_S
    return source.astype(np.float32)


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
    result = advected + DIFFUSIVITY_M2_S * dt_s * lap
    # Q is an explicitly parameterized supply/restoring term, not an implicit
    # part of the horizontal kinematic frontogenesis diagnostic.
    result += dt_s * theta_source(result, config)
    return result.astype(np.float32)


def surface_gust(t: float, theta: np.ndarray,
                 flowdict: dict[str, np.ndarray | float],
                 config: Config) -> dict[str, np.ndarray]:
    """Bulk closure combining free-flow mixing with shallow CAD transport."""
    theta = np.asarray(theta, dtype=np.float32)
    # The cold anomaly from the synoptic ambient field raises stability and
    # reduces mixed depth. This is a prescribed dry surface closure.
    ambient = _ambient_theta(X, Y, t, config)
    cold_anomaly = np.maximum(ambient - theta, 0.0)
    stable = bg.smooth(10.0, 18.0, cold_anomaly)
    mask = (cad_mask(X, Y) if config.cad_enabled else np.zeros_like(X, dtype=np.float32))
    cad_inversion = CAD_INVERSION_MIN_K + (CAD_INVERSION_MAX_K - CAD_INVERSION_MIN_K) * stable
    cad_depth = CAD_DEPTH_MAX_M - (CAD_DEPTH_MAX_M - CAD_DEPTH_MIN_M) * stable
    # Preserve the original weak closure outside CAD; interpolate into the
    # strongly stable 6-10 K, 500-800 m parameterization inside the cold dome.
    inversion = 1.5 + mask * (cad_inversion - 1.5)
    depth = 900.0 + mask * (cad_depth - 900.0)
    upper_speed = np.hypot(flowdict["upper_u_ms"], flowdict["upper_v_ms"])
    floor_speed = np.hypot(flowdict["u_ms"], flowdict["v_ms"])
    # Strong stability limits transfer of free-flow winds. Shallow transport
    # influences this proxy but is not interpreted as a 10-m wind floor.
    coupling = 0.72 - 0.50 * stable * mask
    speed = coupling * upper_speed + (1.0 - coupling) * floor_speed
    gust_ms, mean_ms, _ = bg.mixing_gust(speed, inversion, depth,
                                          exposure_factor=bg.exposure)
    gust_ms = np.maximum(gust_ms, mean_ms)
    return {
        "gust_mph": np.maximum(gust_ms * bg.MS2MPH, 5.0).astype(np.float32),
        "mean_mph": (np.minimum(gust_ms, np.maximum(mean_ms, 0.0))
                     * bg.MS2MPH).astype(np.float32),
        "inversion_K": inversion.astype(np.float32),
        "mixing_depth_m": depth.astype(np.float32),
        "stability_fraction": stable.astype(np.float32),
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
        "cad_mask": cad_mask(X, Y).astype(np.float32),
    }
    peak_gust = np.zeros(X.shape, dtype=np.float32)
    peak_time = np.full(X.shape, float(start_hour), dtype=np.float32)
    station_index = frame_index = 0
    total_steps = int(round((end_hour - start_hour) * 3600.0 / dt_s))
    report_every = max(1, int(round(6.0 * 3600.0 / dt_s)))
    pressure_grid_min = float("inf")
    max_abs_theta_source = 0.0

    for step in range(total_steps + 1):
        t = start_hour + step * dt_s / 3600.0
        fl = flow(t, config)
        pressure_grid_min = min(pressure_grid_min, float(np.min(fl["p_mb"])))
        max_abs_theta_source = max(max_abs_theta_source,
                                   float(np.max(np.abs(theta_source(theta, config)))))
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
    ng_station_stats = None
    if "KATL" in station_ids and station_g.size:
        katl_i = station_ids.index("KATL")
        ng_station_stats = {
            "station_id": "KATL",
            "gust_peak_mph": float(np.max(station_g[:, katl_i])),
            "gust_peak_hour": float(station_hours[int(np.argmax(station_g[:, katl_i]))]),
            "mean_peak_mph": float(np.max(station_m[:, katl_i])),
        }
    theta_range = ([float(fields["theta_K"].min()), float(fields["theta_K"].max())]
                   if nframes else [None, None])
    f_range = ([float(fields["F_K_100km_3h"].min()), float(fields["F_K_100km_3h"].max())]
               if nframes else [None, None])
    metadata = {
        "simulated": True,
        "model_version": MODEL_VERSION,
        "source_fingerprint": fingerprint,
        "source_changes_invalidate_outputs": True,
        "config": asdict(config),
        "projection": "parent Lambert conformal grid treated as Cartesian; DX=3000 m",
        "pressure": "WP central-pressure parameter with spatial 1049-mb SW Quebec background; circular deficit only is in radial gradient balance; total minimum can shift slightly",
        "pressure_background": {
            "high_center_lonlat": list(CAD_HIGH_LONLAT),
            "high_center_mb": CAD_HIGH_CENTER_MB,
            "far_field_mb": BACKGROUND_FAR_MB,
            "gaussian_scale_m": BACKGROUND_SCALE_M,
            "cad_ridge": "shallow hydrostatic SLP bulge; excluded from upper/free-flow pressure-gradient wind because it decays with height",
            "cad_ridge_depth_m": CAD_DEPTH_MAX_M,
            "gradient_direction": "pressure decreases southeast from the off-domain Quebec high",
            "circular_gradient_balance_only": True,
        },
        "cad": {
            "enabled": bool(config.cad_enabled),
            "corridor_lonlat_approx": [list(p) for p in CAD_AXIS_LONLAT],
            "mask": "schematic Gaussian Appalachian corridor; no DEM or resolved terrain",
            "southern_eastern_edge": "smooth geographic gate near the Columbus-Macon-Augusta-Columbia fall line; boundary latitude 32.7 + 0.28*(longitude + 83.65), 55 km transition scale",
            "transport": "southwestward shallow barrier-parallel jet",
            "jet_speed_range_ms": [CAD_JET_MIN_MS, CAD_JET_MAX_MS],
            "core_theta_target_K": CAD_THETA_CORE_K,
            "restoring_timescale_h": CAD_RESTORE_TAU_S / 3600.0,
            "thermal_source": "explicit Q = mask*(target-theta)/tau; archived horizontal kinematic F excludes Q",
            "max_abs_theta_source_K_s": max_abs_theta_source,
            "core_theta_range_K": [CAD_THETA_CORE_K, CAD_THETA_CORE_K],
            "warm_sector": "prescribed +6.5 K broad anomaly centered near (-83.5, 31.8); not heating",
            "inversion_range_K": [CAD_INVERSION_MIN_K, CAD_INVERSION_MAX_K],
            "mixing_depth_range_m": [CAD_DEPTH_MIN_M, CAD_DEPTH_MAX_M],
        },
        "transport": "upper/free-flow wind adds circular vortex, translation, weak strain, and Quebec-high-only geostrophic wind; shallow CAD SLP ridge is excluded from upper wind; lower u_ms/v_ms blend toward prescribed CAD jet",
        "surface_gust": "stability-limited upper-wind mixing plus a shallow transport contribution; lower u_ms/v_ms are not 10-m winds; strong upper wind does not imply automatic surface coupling; gust >= mean",
        "theta": "dry scalar advection-diffusion plus explicit CAD restoring source; no two-way dynamics or latent heating",
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
        "pressure_min_mb": pressure_grid_min,
        "pressure_grid_min_mb": pressure_grid_min,
        "pressure_central_parameter_min_mb": float(np.min(bg.WP[:, 3])),
        "north_georgia_station_stat": ng_station_stats,
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
