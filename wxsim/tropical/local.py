"""Zoomed local-office view of inland tropical warnings from a storm JSON.

usage: python -m wxsim.tropical.local scenarios/<name>/tcwarn_XXXX.json [--outdir DIR]

A tcwarn_*.json points at a tc_*.json (`storm_file`) and sets the office area (`cwa_counties`, "Name ST"),
map `extent`, `cities`, and the hourly-position window. Counties are shaded by the highest inland warning in
the storm's `inland_ww`; the office area is outlined, and the forecast track is interpolated hourly (linear
between forecast points) with local-time center positions.
"""
import argparse, json, warnings
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle
import numpy as np
from shapely.ops import unary_union

from .. import geo
from ..hazards import county_index
from . import storm as S

PRIORITY = {'HU.W': 4, 'TR.W': 3, 'HU.A': 2, 'TR.A': 1}


def hourly_track(st, t0, t1):
    """Linear interpolation of the live forecast points at every whole hour between t0 and t1 (UTC)."""
    pts = [p for p in st.live_points]
    out = []
    t = t0.replace(minute=0)
    while t <= t1:
        for a, b in zip(pts, pts[1:]):
            if a.time <= t <= b.time:
                f = (t - a.time).total_seconds() / (b.time - a.time).total_seconds()
                out.append((t, a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon)))
                break
        t += timedelta(hours=1)
    return out


def render(cfg, outdir):
    matplotlib.rcdefaults()
    base = Path(cfg.get('_base_dir', '.'))
    sp = Path(cfg['storm_file'])
    storm_cfg = json.load(open(sp if sp.is_absolute() or sp.exists() else base / sp))
    st = S.Storm(storm_cfg)
    idx = county_index()

    # highest warning per county
    level = {}
    for blk in storm_cfg.get('inland_ww', []):
        for name in blk['counties']:
            k = name.lower()
            if k in idx and PRIORITY[blk['type']] > PRIORITY.get(level.get(k), 0):
                level[k] = blk['type']
    cwa = [n.lower() for n in cfg['cwa_counties']]
    missing = [n for n in cwa if n not in idx]
    if missing:
        print('  warning: CWA counties not found:', ', '.join(missing))
    cwa = [n for n in cwa if n in idx]
    cwa_geom = unary_union([idx[n][1] for n in cwa]).buffer(0.002).buffer(-0.002)

    lon0, lon1, lat0, lat1 = cfg['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=0.6)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)

    fig = plt.figure(figsize=(12.8, 11.0), facecolor='#1f2a36')
    ax = fig.add_axes([0.012, 0.012, 0.976, 0.845])
    ax.set_facecolor('#a9c8e4')
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#f3f0e6', edgecolor='none', zorder=1)

    colors = {t: S.WW_TYPES[t][1] for t in S.WW_TYPES}
    for k, t in level.items():
        g = idx[k][1]
        if g.intersects(EXT):
            inside = k in cwa
            geo.draw(ax, g, proj, facecolor=colors[t], edgecolor='none', alpha=0.78 if inside else 0.35, zorder=2)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#3d3d3d', county_lw=0.4,
                county_alpha=0.7, state_color='black', state_lw=1.6, z_county=3, z_state=4)
    geo.draw(ax, cwa_geom, proj, facecolor='none', edgecolor='black', lw=3.2, zorder=5)
    geo.draw(ax, cwa_geom, proj, facecolor='none', edgecolor='white', lw=1.0, zorder=5.1)

    # track: hourly positions inside the window
    w = cfg.get('track_window', {})
    t0 = S.parse_z(w['start']) if 'start' in w else st.adv_time
    t1 = S.parse_z(w['end']) if 'end' in w else st.adv_time + timedelta(hours=24)
    hrs = hourly_track(st, t0, t1)
    xy = np.array([tr.transform(lo, la) for _, la, lo in hrs])
    ax.plot(xy[:, 0], xy[:, 1], color='black', lw=2.2, zorder=6)
    ax.plot(xy[:, 0], xy[:, 1], color='white', lw=0.8, ls=(0, (3, 3)), zorder=6.1)
    for (t, la, lo), (x, y) in zip(hrs, xy):
        if not (lon0 + 0.25 <= lo <= lon1 and lat0 + 0.1 <= la <= lat1 - 0.05):
            continue
        ax.plot(x, y, 'o', ms=9, mfc='black', mec='white', mew=1.4, zorder=7)
        lt = t + timedelta(hours=st.tz['offset_h'])
        lab = f"{lt.hour % 12 or 12} {'AM' if lt.hour < 12 else 'PM'}"
        ax.text(x - 9000, y, lab, ha='right', va='center', fontsize=10.5, weight='bold', zorder=8,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    for name, la, lo in cfg.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 's', ms=4.5, color='black', zorder=8)
        ax.text(x + 5000, y + 4000, name, fontsize=10, weight='bold', zorder=8,
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    for s in ax.spines.values():
        s.set_color('black'); s.set_linewidth(1.2)

    # counts inside the office area
    n = len(cwa)
    cnt = {t: sum(1 for k in cwa if level.get(k) == t) for t in S.WW_TYPES}
    title = cfg.get('title', f'{st.title}: Inland Tropical Warnings')
    fig.text(0.015, 0.962, title, color='white', fontsize=21, weight='bold', va='center')
    fig.text(0.015, 0.922, cfg.get('subtitle', ''), color='#d6dde6', fontsize=12, va='center')
    fig.text(0.985, 0.962, 'SIMULATED', color='#ffd84a', fontsize=19, weight='bold', va='center', ha='right')
    upd = st.adv_time + timedelta(hours=st.tz['offset_h'])
    fig.text(0.015, 0.887, f"Valid {upd.hour % 12 or 12}:{upd.minute:02d} {'AM' if upd.hour < 12 else 'PM'} "
             f"{st.tz['name']} {upd:%a %b} {upd.day} {upd.year}  •  NHC-style Advisory {st.adv_num}",
             color='#aab6c4', fontsize=11, va='center', ha='left')

    lg = ax.inset_axes([0.012, 0.012, 0.36, 0.2]); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.set_xticks([]); lg.set_yticks([]); lg.patch.set_alpha(0.93); lg.set_zorder(20)
    lg.text(0.04, 0.86, f"{cfg.get('office_label', 'Office area')} ({n} counties)", fontsize=11, weight='bold',
            va='center')
    y = 0.62
    for t in ('HU.W', 'TR.W', 'HU.A', 'TR.A'):
        if not any(v == t for v in level.values()):
            continue
        lg.add_patch(Rectangle((0.04, y - 0.08), 0.1, 0.16, facecolor=colors[t], edgecolor='black', lw=0.8))
        lg.text(0.18, y, f"{S.WW_TYPES[t][0]}: {cnt[t]} of {n} counties", fontsize=10.5, va='center')
        y -= 0.24
    none = n - sum(cnt.values())
    if none:
        lg.add_patch(Rectangle((0.04, y - 0.08), 0.1, 0.16, facecolor='#f3f0e6', edgecolor='black', lw=0.8))
        lg.text(0.18, y, f"No tropical warning: {none}", fontsize=10.5, va='center')
    ax.text(0.99, 0.012, 'Hourly center positions interpolated from the forecast track. County list approximate.',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic', zorder=21,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
    ax.text(0.5, 0.5, 'SIMULATED', transform=ax.transAxes, fontsize=110, color='black', alpha=0.04,
            ha='center', va='center', rotation=28, weight='bold', zorder=19)

    out = Path(outdir) / cfg.get('output', f'{st.atcf}_local_warnings_adv{st.adv_num}.png')
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print('saved', out, '|', ', '.join(f"{t} {cnt[t]}" for t in cnt if cnt[t]), f'of {n}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg['_base_dir'] = str(Path(a.config).parent)
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir)


if __name__ == '__main__':
    main()
