"""Forecast maximum sustained wind and maximum gust swaths from a storm JSON (deterministic).

usage: python -m wxsim.tropical.windmap scenarios/<name>/tcwind_XXXX.json [--outdir DIR]

The storm is stepped every `step_min` minutes along the track (past fix at/before the advisory, then the
forecast points), with vmax and the 34/50/64-kt quadrant radii interpolated linearly in time. At each step a
radial profile is built per azimuth through (RMW, vmax), (r64, 64), (r50, 50), (r34, 34) and a tail to 20 kt at
`tail_factor` * r34, with radii interpolated between quadrant centers. A zero radius caps that direction below
the threshold. The map keeps the maximum over time. Gusts = sustained * gust factor (land/water).
This is the deterministic swath implied by the advisory; the probabilistic Holland-profile ensemble is a
separate, later module.
"""
import argparse, json, math, warnings
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from shapely import contains_xy

from .. import geo
from . import storm as S

MPH_LEVELS = [20, 30, 39, 50, 58, 65, 74, 85, 96, 111, 130, 157, 200]
MPH_COLORS = ['#cfe8f7', '#8cc7ee', '#fff1a8', '#ffd24d', '#ffa033', '#ff6a1a', '#e81c1c', '#b30f2e', '#8a0f6b',
              '#b43cd6', '#e38bf5', '#ffffff']
CMAP = ListedColormap(MPH_COLORS)
NORM = BoundaryNorm(MPH_LEVELS, CMAP.N)
QUAD_CENTER = {'NE': 45, 'SE': 135, 'SW': 225, 'NW': 315}


def _states(st):
    """(time, lat, lon, vmax, radii, status) knots: last past fix before the advisory, then live points."""
    ks = []
    prev = [r for r in st.past if r['time'] < st.adv_time]
    if prev:
        r = prev[-1]
        ks.append((r['time'], r['lat'], r['lon'], r['vmax'], st.current.radii, r['status']))
    ks += [(p.time, p.lat, p.lon, p.vmax, p.radii, p.status) for p in st.live_points]
    return ks


def _radius(radii, key, quad):
    q = radii.get(key) or {}
    return q.get(quad, 0) if q else 0


def _interp_state(a, b, f):
    t = a[0] + (b[0] - a[0]) * f
    lat = a[1] + f * (b[1] - a[1]); lon = a[2] + f * (b[2] - a[2])
    vmax = a[3] + f * (b[3] - a[3])
    radii = {}
    for key in ('34', '50', '64'):
        radii[key] = {q: _radius(a[4], key, q) + f * (_radius(b[4], key, q) - _radius(a[4], key, q)) for q in S.QUADS}
    return t, lat, lon, vmax, radii, (a[5] if f < 0.5 else b[5])


def field_at(LON, LAT, lat, lon, vmax, radii, status, cfg):
    """Sustained wind (kt) on the grid for one storm state."""
    kx = 111.2 * math.cos(math.radians(lat))
    dx = (LON - lon) * kx; dy = (LAT - lat) * 111.2
    r_nm = np.hypot(dx, dy) / 1.852
    az = np.degrees(np.arctan2(dx, dy)) % 360
    rmw_cfg = cfg.get('rmw_nm', {})
    if status in S.TROPICAL:
        rmw = rmw_cfg.get('tropical', 20)
    elif status == 'EX':
        rmw = rmw_cfg.get('extratropical', rmw_cfg.get('post_tropical', 50))
    else:
        rmw = rmw_cfg.get('post_tropical', 50)
    tail = cfg.get('tail_factor', 1.8)
    out = np.zeros_like(LON)
    quads = list(QUAD_CENTER.items())
    # radii at arbitrary azimuth: linear between quadrant centers
    def rad_at(key):
        cs = np.array([QUAD_CENTER[q] for q in S.QUADS] + [405])
        vs = np.array([radii[key][q] for q in S.QUADS] + [radii[key]['NE']])
        a = np.where(az < 45, az + 360, az)
        return np.interp(a, cs, vs)
    r64, r50, r34 = rad_at('64'), rad_at('50'), rad_at('34')
    peak = np.full_like(LON, vmax)
    peak = np.where((r64 <= 0) & (vmax >= 64), np.minimum(peak, 60), peak)
    peak = np.where((r50 <= 0) & (peak >= 50), np.minimum(peak, 47), peak)
    peak = np.where((r34 <= 0) & (peak >= 34), np.minimum(peak, 32), peak)
    # piecewise-linear profile, vectorized: evaluate each segment and keep where r falls inside it
    knots = [(np.full_like(LON, 0.0), np.zeros_like(LON)), (np.full_like(LON, rmw), peak)]
    for rk, v in ((r64, 64), (r50, 50), (r34, 34)):
        ok = rk > rmw
        knots.append((np.where(ok, rk, np.nan), np.where(ok, np.minimum(v, peak), np.nan)))
    r34e = np.where(r34 > 0, r34, rmw * 2.5)
    knots.append((np.maximum(r34e * tail, rmw * 3), np.full_like(LON, 20.0)))
    knots.append((np.maximum(r34e * tail * 1.8, rmw * 5), np.zeros_like(LON)))
    # sequential valid knots per point
    prev_r, prev_v = knots[0]
    res = np.zeros_like(LON); done = np.zeros(LON.shape, bool)
    for kr, kv in knots[1:]:
        valid = ~np.isnan(kr)
        seg = valid & ~done & (r_nm <= kr)
        span = np.where(valid, kr - prev_r, 1)
        f = np.clip((r_nm - prev_r) / np.where(span > 0, span, 1), 0, 1)
        res = np.where(seg, prev_v + f * (np.where(valid, kv, 0) - prev_v), res)
        done |= seg
        prev_r = np.where(valid, kr, prev_r); prev_v = np.where(valid, kv, prev_v)
    return res


def swath(st, cfg, LON, LAT):
    ks = _states(st)
    t_end = S.parse_z(cfg['end']) if cfg.get('end') else ks[-1][0]
    step = timedelta(minutes=cfg.get('step_min', 20))
    vmax_grid = np.zeros_like(LON)
    t = S.parse_z(cfg['start']) if cfg.get('start') else st.adv_time
    while t <= t_end:
        for a, b in zip(ks, ks[1:]):
            if a[0] <= t <= b[0]:
                f = (t - a[0]).total_seconds() / max((b[0] - a[0]).total_seconds(), 1)
                _, la, lo, vm, rd, stt = _interp_state(a, b, f)
                vmax_grid = np.maximum(vmax_grid, field_at(LON, LAT, la, lo, vm, rd, stt, cfg))
                break
        t += step
    return vmax_grid


def modifier_field(cfg, LON, LAT, res):
    """Optional multiplier zones on the swath ("modifiers": [{"polygon": [[lat, lon], ...], "factor": 0.6,
    "feather_km": 40}]). Use factor < 1 for a stable layer that decouples the surface from the wind field
    (a cold-air-damming wedge, a cold marine layer) and > 1 for zones of strong downward mixing (a sting jet).
    Edges are feathered with a Gaussian so there is no step in the map. No modifiers -> all ones."""
    out = np.ones_like(LON)
    for m in cfg.get('modifiers', []):
        from shapely.geometry import Polygon
        from scipy.ndimage import gaussian_filter
        poly = Polygon([(lo, la) for la, lo in m['polygon']])
        mask = contains_xy(poly, LON, LAT).astype(float)
        sig = m.get('feather_km', 40) / (res * 111.2)
        w = np.clip(gaussian_filter(mask, sig), 0, 1)
        out *= 1 + w * (m['factor'] - 1)
    return out


def render(cfg, outdir):
    matplotlib.rcdefaults()
    base = Path(cfg.get('_base_dir', '.'))
    sp = Path(cfg['storm_file'])
    st = S.Storm(json.load(open(sp if sp.exists() else base / sp)))
    lon0, lon1, lat0, lat1 = cfg['extent']
    res = cfg.get('res', 0.025)
    LON, LAT = np.meshgrid(np.arange(lon0 - 0.5, lon1 + 0.5, res), np.arange(lat0 - 0.5, lat1 + 0.5, res))
    sus_kt = swath(st, cfg, LON, LAT) * modifier_field(cfg, LON, LAT, res)
    land = contains_xy(geo.land(), LON, LAT)
    gf = cfg.get('gust_factor', {'land': 1.35, 'water': 1.2})
    gust_kt = sus_kt * np.where(land, gf['land'], gf['water'])
    sus, gust = sus_kt * 1.15078, gust_kt * 1.15078

    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=0.6)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    X, Y = tr.transform(LON, LAT)
    fig = plt.figure(figsize=(16, 8.8), facecolor='white')
    fig.text(0.01, 0.975, cfg.get('title', f'{st.title}: Forecast Peak Winds'), fontsize=18, weight='bold', va='top')
    fig.text(0.01, 0.935, cfg.get('subtitle', ''), fontsize=10.5, va='top', color='#333')
    fig.text(0.99, 0.975, 'SIMULATED', fontsize=18, weight='bold', va='top', ha='right', color='#b00000')
    hrs = hourly = []
    for k, (fld, name) in enumerate(((sus, 'Maximum sustained wind (mph)'), (gust, 'Maximum wind gust (mph)'))):
        ax = fig.add_axes([0.005 + k * 0.5, 0.1, 0.49, 0.765])
        ax.set_facecolor('#dfe7ee')
        geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#fbfaf5', edgecolor='none', zorder=1)
        f = np.where(fld >= MPH_LEVELS[0], fld, np.nan)
        ax.contourf(X, Y, f, levels=MPH_LEVELS, cmap=CMAP, norm=NORM, zorder=2, alpha=0.9)
        geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#555', county_lw=0.2,
                    county_alpha=0.4, state_color='black', state_lw=1.1, z_county=3, z_state=4)
        # track
        ks = _states(st)
        txy = np.array([tr.transform(k_[2], k_[1]) for k_ in ks])
        ax.plot(txy[:, 0], txy[:, 1], color='black', lw=1.6, zorder=5)
        for k_ in ks:
            x, y = tr.transform(k_[2], k_[1])
            ax.plot(x, y, 'o', ms=6, mfc='black' if k_[5] in S.TROPICAL else 'white', mec='black', zorder=6)
            # optional "track_labels": list of UTC times to label (default: every point inside the map)
            want = cfg.get('track_labels')
            if lon0 < k_[2] < lon1 and lat0 < k_[1] < lat1 and \
                    (want is None or k_[0].strftime('%Y-%m-%dT%H:%MZ') in want):
                ax.text(x - 9000, y, S.local_label(k_[0], st.tz), ha='right', va='center', fontsize=8.5,
                        weight='bold', zorder=6, path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
        for cname, la, lo in cfg.get('cities', []):
            j = int(round((la - LAT[0, 0]) / res)); i = int(round((lo - LON[0, 0]) / res))
            v = fld[j, i]
            x, y = tr.transform(lo, la)
            ax.plot(x, y, 's', ms=3.5, color='black', zorder=7)
            ax.text(x + 6000, y + 5000, f'{cname} {v:.0f}', fontsize=8.8, weight='bold', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
        geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
        ax.set_title(name, fontsize=13, weight='bold', loc='left')
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation='horizontal',
                      ticks=MPH_LEVELS[:-1])
    cb.set_label(cfg.get('cbar_label', 'mph   (39 = tropical storm force, 74 = hurricane force, 111 = major hurricane)'),
                 fontsize=9)
    foot = cfg.get('footnote', f'Deterministic swath from Advisory {st.adv_num} wind radii')
    fig.text(0.99, 0.012, f"{foot}; gusts = sustained × {gf['land']} over land, × {gf['water']} over water",
             fontsize=7.5, style='italic', ha='right', color='#555')
    out = Path(outdir) / cfg.get('output', f'{st.atcf}_wind_swath_adv{st.adv_num}.png')
    fig.savefig(out, dpi=140, facecolor='white'); plt.close(fig)
    print('saved', out)
    for cname, la, lo in cfg.get('cities', []):
        j = int(round((la - LAT[0, 0]) / res)); i = int(round((lo - LON[0, 0]) / res))
        print(f'  {cname:<14} sustained {sus[j, i]:5.0f} mph   gust {gust[j, i]:5.0f} mph')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); cfg['_base_dir'] = str(Path(a.config).parent)
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir)


if __name__ == '__main__':
    main()
