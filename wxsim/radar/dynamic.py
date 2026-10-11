"""Analytic, state-driven radar echo for evolving synthetic storms.

The field is a reduced-order state model.  A seeded scalar texture is
advected through age-dependent updraft and buoyancy states; condensate,
vertical vorticity, and a cold-pool state interact before being converted to
linear reflectivity.  This gives changing irregular structure without mutable
noise, shape catalogs, or a full fluid solver.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


_TAU = 2.0 * np.pi


def _number(params: Mapping[str, object] | None, key: str, default: float) -> float:
    """Read one finite scalar parameter without mutating the caller's mapping."""
    if not params:
        return default
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # Clipping keeps extreme z/parameter values finite and avoids overflow
    # warnings in large radar grids.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _vertical_weight(z: np.ndarray, bottom: float, top: float) -> np.ndarray:
    """Smoothly gate an echo between its cloud base and echo top."""
    width_bottom = 0.28 + 0.10 * min(bottom, 2.0)
    width_top = 0.55 + 0.06 * max(top - 7.0, 0.0)
    return _sigmoid((z - bottom) / width_bottom) * _sigmoid((top - z) / width_top)


def _stable_phase(seed: int, storm_id: str) -> tuple[float, float, float]:
    """Return deterministic phases in [0, 2pi), independent of process hash state."""
    payload = f"{int(seed)}:{storm_id}".encode("utf-8", "replace")
    digest = hashlib.blake2b(payload, digest_size=24).digest()
    ints = np.frombuffer(digest, dtype=np.uint64)
    return tuple(float((int(x) / 18446744073709551616.0) * _TAU) for x in ints[:3])


def _precip_fraction(value: float) -> float:
    """Map the generator's signed LP (-1) to HP (+1) state into 0..1."""
    return _clip01(0.5 * (value + 1.0))


def reach_km(age_min: float, params: Mapping[str, object] | None) -> float:
    """Return a conservative horizontal radius for culling radar/MRMS work.

    The bound includes the broad line and precipitation shield, so callers may
    safely skip all points outside it.  ``age_min`` is accepted to keep the
    function parallel with :func:`reflectivity`; growth is capped smoothly.
    """
    age = _number({"age": age_min}, "age", 0.0)
    age = float(np.clip(age, 0.0, 360.0))
    scale = float(np.clip(_number(params, "scale", 1.0), 0.15, 4.0))
    line = _clip01(_number(params, "line", 0.0))
    prec = _precip_fraction(_number(params, "prec", 0.0))
    bow = _clip01(_number(params, "bow", 0.0))
    lowtop = _clip01(_number(params, "lowtop", 0.0))
    growth = 0.88 + 0.18 * float(_smoothstep(age / 35.0))
    # The +10 km margin covers the warped coordinate field and smooth edge.
    radius = scale * growth * (20.0 + 35.0 * prec + 47.0 * line + 12.0 * bow)
    radius *= 1.0 - 0.12 * lowtop
    return float(np.clip(radius + 12.0, 12.0, 180.0))


def reflectivity(
    u: object,
    v: object,
    z_km: object,
    age_min: object,
    params: Mapping[str, object] | None,
    seed: int = 17,
    storm_id: str = "dynamic",
) -> np.ndarray:
    """Evaluate a deterministic evolving storm field in dBZ.

    ``u`` and ``v`` are storm-relative kilometres (forward and left), and
    ``z_km`` is height AGL.  All four coordinate arguments are broadcast with
    NumPy.  Values outside the finite storm support are ``-inf``; no random
    generator or mutable module state is used.
    """
    # Broadcast first so scalar and mixed-grid calls follow one vector path.
    try:
        u_arr, v_arr, z_arr, age_arr = np.broadcast_arrays(
            np.asarray(u, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(z_km, dtype=float),
            np.asarray(age_min, dtype=float),
        )
    except (TypeError, ValueError):
        # This makes malformed object-like inputs fail as a normal NumPy value
        # error while keeping the public return contract for valid arrays.
        raise ValueError("u, v, z_km, and age_min must be numeric broadcastable arrays") from None

    age = np.where(np.isfinite(age_arr), np.clip(age_arr, 0.0, 360.0), 0.0)
    phase0, phase1, phase2 = _stable_phase(seed, storm_id)

    scale = float(np.clip(_number(params, "scale", 1.0), 0.15, 4.0))
    sup0 = _clip01(_number(params, "sup", 0.25))
    # ``prec`` follows the generator's signed morphology state: -1 is LP,
    # 0 is classic, and +1 is HP.  The internal moisture fraction is 0..1.
    prec0 = _precip_fraction(_number(params, "prec", 0.0))
    hook0 = _clip01(_number(params, "hook", 0.20))
    elev = _clip01(_number(params, "elev", 0.0))
    high_based = _clip01(_number(params, "hb", 0.0))
    hail = float(np.clip(_number(params, "hail", 58.0), 35.0, 80.0))
    dz = float(np.clip(_number(params, "dz", 0.0), -60.0, 20.0))
    line = _clip01(_number(params, "line", 0.0))
    bow = _clip01(_number(params, "bow", 0.0))
    tor = _clip01(_number(params, "tor", 0.0))
    cycle = _number(params, "cycle", 0.0) % 1.0
    lowtop = _clip01(_number(params, "lowtop", 0.0))
    split_core = _clip01(_number(params, "split_core", 0.0))
    merge = _clip01(_number(params, "merge", 0.0))
    merge_fade = _clip01(_number(params, "merge_fade", 0.0))

    # Optional environment forcing fields are neutral when omitted.  They make
    # the analytic field useful directly from generator keyframes while leaving
    # explicit morphology state in control for hand-authored storms.
    mlcape = float(np.clip(_number(params, "mlcape", 1050.0), 0.0, 5000.0))
    shear = float(np.clip(_number(params, "shear_kt", 35.0), 0.0, 120.0))
    rh_low = float(np.clip(_number(params, "rh_low", 55.0), 0.0, 100.0))
    srh01 = float(np.clip(_number(params, "srh01", 100.0), 0.0, 500.0))
    env_cape = _clip01((mlcape - 300.0) / 1800.0)
    env_shear = _clip01((shear - 18.0) / 52.0)
    env_moist = _clip01((rh_low - 35.0) / 55.0)
    env_srh = _clip01(srh01 / 250.0)
    organization = _clip01(0.78 * sup0 + 0.12 * env_shear + 0.10 * env_cape)
    moisture = _clip01(0.78 * prec0 + 0.22 * env_moist)
    hook_state = _clip01(0.72 * hook0 + 0.18 * env_srh + 0.10 * organization)

    growth = _smoothstep(age / 18.0)
    fade = 1.0 - _smoothstep((age - 175.0) / 95.0)
    mature = growth * fade
    # A finite lifecycle envelope means age=0 and very old exploratory calls
    # remain safe, while the caller's life metadata can still remove dead cells.
    lifecycle_db = -10.0 * (1.0 - growth) - 12.0 * (1.0 - fade)
    state_phase = _TAU * (age / 42.0 + cycle)

    # Semi-Lagrangian backtrace: the storm-relative flow is simple, but its
    # displacement is age dependent, which lets the same analytic state be
    # sampled at arbitrary times without a mutable timestep cache.
    adv_u = scale * (0.018 + 0.018 * organization) * age
    adv_v = scale * (0.008 * np.sin(state_phase * 0.31 + phase0)) * age
    x0, y0 = u_arr - adv_u, v_arr - adv_v
    texture = (
        0.57 * np.sin(0.105 * x0 + 0.071 * y0 - 0.018 * age + phase0)
        + 0.29 * np.sin(0.043 * x0 - 0.131 * y0 + 0.011 * age + phase1)
        + 0.14 * np.cos(0.019 * x0 + 0.027 * y0 - 0.007 * age + phase2)
    )
    texture = np.clip(texture, -1.0, 1.0)
    radius = np.hypot(x0, y0)

    # Updraft and buoyancy are the source state.  Their horizontal scale grows
    # with organization and moisture; split_core modulates a smooth instability
    # wave, rather than selecting two pre-drawn cells.
    source_u = scale * (2.8 + 2.5 * organization) * (0.90 + 0.14 * mature)
    source_v = scale * (2.5 + 2.2 * organization) * (0.92 + 0.12 * mature)
    source_metric = (x0 / source_u) ** 2 + (y0 / source_v) ** 2
    split_wave = 1.0 + split_core * (0.52 + 0.48 * mature) * np.cos(
        (y0 / (source_v + 0.5)) * np.pi + 0.35 * state_phase + phase1
    )
    updraft = np.exp(-0.5 * source_metric) * np.clip(0.78 + 0.25 * texture, 0.42, 1.08)
    updraft = np.clip(updraft * split_wave, 0.0, 1.35)
    buoyancy = np.clip(
        (0.28 + 0.72 * mature) * updraft * (0.72 + 0.28 * env_cape)
        + 0.18 * np.maximum(texture, 0.0) * (0.35 + 0.65 * moisture),
        0.0,
        1.45,
    )

    # Condensate is a residence-time integral of buoyant ascent.  The cloud
    # state is deliberately wider than the updraft, and the line state changes
    # its cross-flow diffusion so nearby cells can consolidate into a segment.
    cloud_u = source_u * (1.9 + 0.8 * moisture)
    cloud_v = source_v * (1.8 + 2.0 * line + 0.6 * moisture)
    cloud_metric = (x0 / cloud_u) ** 2 + (y0 / cloud_v) ** 2
    cloud_state = np.exp(-0.5 * cloud_metric) * (0.48 + 0.52 * np.maximum(texture, -0.5))
    residence = 1.0 - np.exp(-age / (9.0 + 8.0 * moisture))
    condensate = np.clip(
        (0.34 + 0.66 * residence) * (0.70 * buoyancy + 0.30 * cloud_state)
        * (0.62 + 0.55 * moisture),
        0.0,
        1.7,
    )

    # Vertical vorticity is sourced by buoyancy gradients and environmental
    # shear.  It transports the condensate through a smooth swirl displacement;
    # this interaction creates asymmetric wraps and occlusion without a drawn
    # hook or appendage.
    radial_angle = np.arctan2(y0, x0)
    vort_strength = (0.18 + 0.82 * organization) * (0.45 + 0.55 * env_srh) * (1.0 + 0.95 * tor)
    vorticity = vort_strength * buoyancy * (
        0.58 * np.sin(radial_angle + 0.34 * state_phase + phase2)
        + 0.42 * texture
    )
    cycle_gate = 1.0 - 0.38 * organization * (0.5 + 0.5 * hook_state) * (
        0.5 + 0.5 * np.cos(_TAU * cycle + radial_angle - phase0)
    )
    vorticity *= cycle_gate
    swirl = scale * (0.65 + 1.35 * hook_state) * vorticity / (1.0 + 0.12 * radius)
    xh = x0 - swirl * y0
    yh = y0 + swirl * x0
    transported_metric = (xh / (cloud_u + 0.8)) ** 2 + (yh / (cloud_v + 0.8)) ** 2
    transported = np.exp(-0.5 * transported_metric)
    condensate = np.clip(condensate * (0.72 + 0.58 * transported), 0.0, 2.0)

    # Cold-pool state is generated by condensate loading and low-level cooling.
    # Its age-dependent advection and bow curvature come from outflow forcing,
    # so line/bow controls change the state evolution rather than add a shape.
    pool_speed = scale * (2.5 + 8.0 * moisture + 5.0 * np.maximum(buoyancy, 0.0))
    pool_length = scale * (7.0 + 25.0 * moisture + 32.0 * line)
    pool_shift = pool_speed * (0.45 + 0.55 * mature)
    pool_curve = bow * scale * 8.0 * np.tanh(y0 / max(pool_length, 1.0))
    xp = x0 + pool_shift + pool_curve
    pool_metric = (xp / (5.0 + 4.0 * scale)) ** 2 + (y0 / max(pool_length, 1.0)) ** 2
    cold_pool = np.exp(-0.5 * pool_metric) * (0.25 + 0.75 * condensate) * (0.35 + 0.65 * mature)
    cold_pool *= np.clip(0.86 + 0.30 * texture + 0.22 * line, 0.35, 1.25)

    # Hydrometeor loading is the observable state.  Precipitation, ice/hail,
    # and low-level vorticity all contribute through one bounded linear-Z field.
    hail_strength = _clip01((hail - 48.0) / 26.0) * (0.20 + 0.80 * organization)
    rain_state = np.clip(condensate + 0.34 * cold_pool, 0.0, 2.5)
    ice_state = np.clip(
        rain_state * (0.12 + 0.88 * hail_strength) * (0.50 + 0.50 * buoyancy)
        * (1.0 + 0.42 * np.maximum(vorticity, 0.0)),
        0.0,
        2.5,
    )
    echo_bottom = 0.08 + 2.1 * elev + 1.1 * high_based
    echo_top = 7.4 + 4.8 * organization + 0.9 * moisture - 2.7 * lowtop
    rain_vertical = _vertical_weight(z_arr, echo_bottom, echo_top)
    ice_vertical = _vertical_weight(z_arr, 0.06 + 2.4 * elev + 1.4 * high_based, echo_top + 0.4)
    rain_db = 20.0 + 29.0 * np.clip(rain_state, 0.0, 1.2) + 8.0 * organization + 4.0 * moisture + dz + lifecycle_db
    ice_db = hail + 2.0 * mature + dz + lifecycle_db - 9.0 * (1.0 - hail_strength)
    linear = (
        np.power(10.0, np.clip(rain_db / 10.0, -8.0, 8.0)) * rain_state * rain_vertical
        + np.power(10.0, np.clip(ice_db / 10.0, -8.0, 8.0)) * ice_state * ice_vertical
    )
    linear *= 1.0 + 0.26 * merge * np.exp(-0.5 * (radius / (scale * (5.0 + 3.5 * merge))) ** 2)
    linear *= np.power(10.0, -8.0 * merge_fade / 10.0)

    # Tornadic vorticity and the cycle state reduce low-level loading during an
    # occlusion pulse, while preserving a smooth finite field everywhere else.
    low_level = _vertical_weight(z_arr, echo_bottom, min(echo_top, 3.5))
    linear *= np.power(10.0, -3.5 * tor * np.maximum(-vorticity, 0.0) * low_level / 10.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dbz = 10.0 * np.log10(linear)
    echo_texture = (
        2.2 * np.sin(0.22 * x0 + 0.15 * y0 + 0.035 * age + phase0)
        + 1.2 * np.sin(0.09 * x0 - 0.19 * y0 - 0.021 * age + phase1)
        + 0.7 * np.cos(0.035 * x0 + 0.045 * y0 + 0.012 * age + phase2)
    )
    dbz = dbz + echo_texture * (0.20 + 0.80 * mature)

    # Explicit finite support keeps -inf meaningful to the sampler and avoids
    # spending work on Gaussian tails.  Vertical margins preserve smooth fade.
    reach = reach_km(float(np.max(age)) if age.size else 0.0, params)
    top = 7.4 + 4.8 * organization + 0.9 * moisture - 2.7 * lowtop
    bottom = min(0.0, 0.10 + 2.1 * elev + 1.1 * high_based) - 1.5
    valid = np.isfinite(u_arr) & np.isfinite(v_arr) & np.isfinite(z_arr) & np.isfinite(age_arr)
    valid &= radius <= reach
    valid &= (z_arr >= bottom) & (z_arr <= top + 1.5)
    valid &= np.isfinite(dbz) & (linear > 1e-10)
    return np.where(valid, dbz, -np.inf).astype(float, copy=False)


__all__ = ["reflectivity", "reach_km"]
