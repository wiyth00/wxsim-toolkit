"""Rule-based rainfall field for WPC-style products (QPF maps, ERO, storm-total reports).

Not a model of physics, just a controllable sum of recognizable pieces, all in inches on a lon/lat grid:
  pre     predecessor rain event(s): a ridge of rain along an axis ([lon, lat] points, per-point peak inches,
          Gaussian half-width in km)
  track   tropical-cyclone rain sampled hourly along a storm's track (tc_*.json): each hour adds
          rate_in_hr * exp(-(r/width_km)^2) around a center shifted `shift_km` toward `shift_dir` (the left-of-track
          rain shield of an extratropical transition). Slower storms automatically rain more.
  orog    multiplicative bumps [lon, lat, radius_km, factor] for upslope enhancement
plus a smooth random texture (seeded). Every product in wxsim.wpc reads this same field, so the QPF map,
the ERO and the storm-total rainfall lists always agree.
"""
import json, math
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ..tropical import storm as S

KM_PER_DEG = 111.2


def grid(extent, res=0.04):
    lon0, lon1, lat0, lat1 = extent
    lons = np.arange(lon0, lon1 + res / 2, res)
    lats = np.arange(lat0, lat1 + res / 2, res)
    return np.meshgrid(lons, lats)


def _km(LON, LAT, lon, lat):
    """Local flat-earth dx, dy (km) from a point."""
    return (LON - lon) * KM_PER_DEG * math.cos(math.radians(lat)), (LAT - lat) * KM_PER_DEG


def _seg_dist(LON, LAT, a, b):
    """Distance (km) from every grid point to segment a-b, plus the fraction t along it."""
    lat_m = (a[1] + b[1]) / 2
    kx = KM_PER_DEG * math.cos(math.radians(lat_m))
    ax, ay, bx, by = a[0] * kx, a[1] * KM_PER_DEG, b[0] * kx, b[1] * KM_PER_DEG
    px, py = LON * kx, LAT * KM_PER_DEG
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def pre_field(LON, LAT, spec):
    out = np.zeros_like(LON)
    for band in spec:
        pts, peaks, w = band['axis'], band['peak_in'], band['width_km']
        for i in range(len(pts) - 1):
            d, t = _seg_dist(LON, LAT, pts[i], pts[i + 1])
            peak = peaks[i] + t * (peaks[i + 1] - peaks[i])
            out = np.maximum(out, peak * np.exp(-(d / w) ** 2))
    return out


def hourly_positions(st, t0, t1):
    """Past fixes + live forecast points, linearly interpolated every hour in [t0, t1]."""
    fixes = [(r['time'], r['lat'], r['lon'], r['status']) for r in st.past if r['time'] < st.adv_time]
    fixes += [(p.time, p.lat, p.lon, p.status) for p in st.live_points]
    out, t = [], t0.replace(minute=0)
    while t <= t1:
        for a, b in zip(fixes, fixes[1:]):
            if a[0] <= t <= b[0]:
                f = (t - a[0]).total_seconds() / max((b[0] - a[0]).total_seconds(), 1)
                out.append((t, a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), b[3] if f > 0.5 else a[3]))
                break
        t += timedelta(hours=1)
    return out


def track_field(LON, LAT, spec, base_dir, t0, t1):
    sp = Path(spec['storm_file'])
    cfg = json.load(open(sp if sp.exists() else Path(base_dir) / sp))
    st = S.Storm(cfg)
    hrs = hourly_positions(st, t0, t1)
    out = np.zeros_like(LON)
    rate0, w = spec.get('rate_in_hr', 1.3), spec.get('width_km', 110)
    pt_rate = spec.get('post_tropical_rate_factor', 0.75)
    shift, sdir = spec.get('shift_km', {}), spec.get('shift_dir_deg', 300)
    decay = spec.get('post_tropical_decay_hr')      # e-folding time of rain rate after transition (None = off)
    t_pt = next((h[0] for h in hourly_positions(st, st.past[0]['time'], hrs[-1][0] if hrs else t1)
                 if h[3] not in S.TROPICAL and h[0] > st.past[0]['time']), None) if hrs else None
    for i, (t, la, lo, status) in enumerate(hrs):
        tropical = status in S.TROPICAL
        rate = rate0 * (1 if tropical else pt_rate)
        if not tropical and decay and t_pt:
            rate *= math.exp(-max((t - t_pt).total_seconds() / 3600, 0) / decay)
        sk = shift.get('tropical', 0) if tropical else shift.get('post_tropical', 60)
        la_c = la + sk * math.cos(math.radians(sdir)) / KM_PER_DEG
        lo_c = lo + sk * math.sin(math.radians(sdir)) / (KM_PER_DEG * math.cos(math.radians(la)))
        dx, dy = _km(LON, LAT, lo_c, la_c)
        wid = w * (1 if tropical else spec.get('post_tropical_width_factor', 1.3))
        out += rate * np.exp(-(dx * dx + dy * dy) / wid ** 2)
    return out, hrs


def orog_factor(LON, LAT, spec):
    f = np.ones_like(LON)
    for lo, la, r, fac in spec:
        dx, dy = _km(LON, LAT, lo, la)
        f *= 1 + (fac - 1) * np.exp(-(dx * dx + dy * dy) / r ** 2)
    return f


def texture(shape, seed, amp, sigma=6):
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(shape), sigma)
    n /= (n.std() or 1)
    return np.clip(1 + amp * n, 0.4, 1.8)


def build(cfg, base_dir, window=None, parts=('pre', 'track', 'orog')):
    """Returns LON, LAT, total inches (and the hourly track used). window = (t0, t1) UTC for the track part;
    the pre part is included whole whenever 'pre' is in parts."""
    LON, LAT = grid(cfg['grid_extent'], cfg.get('res', 0.04))
    tot = np.zeros_like(LON)
    if 'pre' in parts:
        tot += pre_field(LON, LAT, cfg.get('pre', []))
    hrs = []
    if 'track' in parts and cfg.get('track'):
        t0, t1 = window or (S.parse_z(cfg['track']['start']), S.parse_z(cfg['track']['end']))
        tr, hrs = track_field(LON, LAT, cfg['track'], base_dir, t0, t1)
        tot += tr
    if 'orog' in parts:
        tot *= orog_factor(LON, LAT, cfg.get('orog', []))
    tot *= texture(tot.shape, cfg.get('seed', 1), cfg.get('texture', 0.15))
    return LON, LAT, gaussian_filter(tot, 1.0), hrs


def sample(LON, LAT, field, lat, lon):
    j = int(round((lat - LAT[0, 0]) / (LAT[1, 0] - LAT[0, 0])))
    i = int(round((lon - LON[0, 0]) / (LON[0, 1] - LON[0, 0])))
    if 0 <= j < field.shape[0] and 0 <= i < field.shape[1]:
        return float(field[j, i])
    return float('nan')


def top_places(field_cfg, base_dir, states, n_per_state=4, force=(), window=None):
    """Storm-total amounts at gazetteer places (data/places_us.csv), top n per state plus forced names.
    Returns {state: [(name, inches), ...]} sorted high to low."""
    import csv
    from .. import geo
    LON, LAT, f, _ = build(field_cfg, base_dir, window=window)
    by = {s: [] for s in states}
    with open(geo.DATA / 'places_us.csv') as fh:
        for row in csv.DictReader(fh):
            if row['state'] in by:
                v = sample(LON, LAT, f, float(row['lat']), float(row['lon']))
                if v == v:
                    by[row['state']].append((row['name'], v))
    out = {}
    for s, rows in by.items():
        rows.sort(key=lambda r: -r[1])
        keep = rows[:n_per_state]
        for fn in force:
            nm, st_ = fn.rsplit(' ', 1)
            if st_ == s and all(k[0] != nm for k in keep):
                keep += [r for r in rows if r[0] == nm][:1]
        out[s] = sorted(keep, key=lambda r: -r[1])
    return out
