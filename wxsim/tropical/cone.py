"""NHC-style 5-day track-forecast cone from a storm JSON (tc_*.json).

usage: python -m wxsim.tropical.cone scenarios/<name>/tc_XXXX.json [--outdir DIR]

Cone = union of a circle at every forecast point (radius from cone.radii_nmi, illustrative) plus the convex
hull of each consecutive pair of circles. Days 1-3 are solid white, days 4-5 stippled. Also draws the dashed
past track, D/S/H/M markers with local-time labels, current 34-kt and 64-kt wind extent from the quadrant
radii, breakpoint-to-breakpoint coastal watches/warnings, optional inland county warnings, and the info box.
"""
import argparse, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .. import geo
from . import storm as S
from . import coast

GEOD = Geod(ellps='WGS84')
NM = 1852.0
LAND, OCEAN = '#f4f1e8', '#9fc3e3'
WIND34, WIND64 = '#f0a24a', '#b0102a'


# ----------------------------------------------------------------- geometry builders (lon/lat -> projected m)
def geod_ring(lat, lon, r_nmi, az0=0, az1=360, n=None):
    n = n or max(8, int(abs(az1 - az0) / 4))
    az = np.linspace(az0, az1, n)
    lons, lats, _ = GEOD.fwd(np.full(n, lon), np.full(n, lat), az, np.full(n, r_nmi * NM))
    return list(zip(lons, lats))


def cone_polygons(st, tr):
    """Returns (day1-3 polygon, day4-5-only polygon) in projected metres."""
    pts = st.live_points
    circles = []
    for p in pts:
        r = max(st.cone_radii[p.tau], 1)
        ring = [tr.transform(x, y) for x, y in geod_ring(p.lat, p.lon, r, n=120)]
        circles.append((p.tau, Polygon(ring)))
    pieces = [(circles[0][0], circles[0][1])]
    for (t0, c0), (t1, c1) in zip(circles, circles[1:]):
        pieces.append((t1, unary_union([c0, c1]).convex_hull))
    early = unary_union([g for t, g in pieces if t <= 72])
    full = unary_union([g for _, g in pieces])
    return early, full.difference(early).buffer(0)


def wind_extent(p, key):
    """Quadrant-radii wind field (lon/lat polygon); zero-radius quadrants collapse to the center."""
    q = p.radii.get(key)
    if not q or not any(q.values()):
        return None
    ring = []
    for quad in S.QUADS:
        a0, a1 = S.QUAD_BEARINGS[quad]
        r = q.get(quad, 0)
        ring += [(p.lon, p.lat)] if r <= 0 else geod_ring(p.lat, p.lon, r, a0, a1)
    return Polygon(ring).buffer(0)


def smooth_wind(g, proj):
    """Round the quadrant corners a little, the way the NHC wind-field graphics look."""
    g = proj(g)
    return g.buffer(12000, join_style=1).buffer(-12000, join_style=1)


def draw_proj(ax, g, **kw):
    """Draw an already-projected polygon."""
    geo.draw(ax, g, lambda x: x, **kw)


# ----------------------------------------------------------------- main render
def render(cfg, outdir):
    matplotlib.rcdefaults()
    st = S.Storm(cfg)
    cc = cfg['cone']
    lon0, lon1, lat0, lat1 = cc['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=2.0)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)

    # figure sized to the map aspect
    xa, ya = tr.transform(lon0, lat1); xb, yb = tr.transform(lon1, lat0)
    x0, y0 = tr.transform(lon0, lat0); x1, y1 = tr.transform(lon1, lat1)
    w = min(x1, xb) - max(x0, xa); h = min(y1, ya) - max(y0, yb)
    fw = 13.0
    fig = plt.figure(figsize=(fw, fw * h / w + 0.55), facecolor='white')
    top = 1 - 0.55 / (fw * h / w + 0.55)
    ax = fig.add_axes([0, 0, 1, top])

    # --- basemap: land, states (no counties, like the NHC cone)
    ax.set_facecolor(OCEAN)
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor=LAND, edgecolor='#5d6b78', lw=0.6, zorder=1)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_box=geo.box(0, 0, 0.01, 0.01),
                state_color='#7c858d', state_lw=0.7, z_state=4)

    # --- inland county warnings (optional)
    if cc.get('show_inland', True) and cfg.get('inland_ww'):
        from ..hazards import county_index
        idx = county_index()
        for t in S.WW_ORDER:
            for blk in [b for b in cfg['inland_ww'] if b['type'] == t]:
                gs = []
                for name in blk['counties']:
                    hit = idx.get(name.lower())
                    if hit is None:
                        print(f'  warning: county "{name}" not found'); continue
                    gs.append(hit[1])
                u = unary_union(gs).buffer(0.003).buffer(-0.003)
                col = S.WW_TYPES[t][1]
                geo.draw(ax, u, proj, facecolor=col, edgecolor='none', alpha=0.28, zorder=2)
                geo.draw(ax, u, proj, facecolor='none', edgecolor=col, lw=1.0, alpha=0.9, zorder=2.1)

    # --- cone
    early, late = cone_polygons(st, tr)
    draw_proj(ax, late, facecolor='white', edgecolor='none', alpha=0.35, zorder=5)
    draw_proj(ax, late, facecolor='none', edgecolor='#6c6c6c', hatch='....', lw=0, zorder=5.1)
    draw_proj(ax, early, facecolor='white', edgecolor='none', alpha=0.62, zorder=5.2)
    draw_proj(ax, unary_union([early, late]), facecolor='none', edgecolor='black', lw=1.1, zorder=6.5)

    # --- current wind extent
    cur = st.current
    g34, g64 = wind_extent(cur, '34'), wind_extent(cur, '64')
    if g34 is not None:
        draw_proj(ax, smooth_wind(g34, proj), facecolor=WIND34, edgecolor='#a8641c', lw=0.8, alpha=0.5, zorder=6)
    if g64 is not None:
        draw_proj(ax, smooth_wind(g64, proj), facecolor=WIND64, edgecolor='#6a0818', lw=0.8, alpha=0.6, zorder=6.1)

    # --- coastal watches/warnings (warnings drawn last, on top)
    for t in S.WW_ORDER:
        for seg in [s for s in cfg.get('coastal_ww', []) if s['type'] == t]:
            ln = coast.segment_line(seg['from'], seg['to'])
            xy = np.array([tr.transform(x, y) for x, y in ln.coords])
            lw = 6.5 if t.endswith('W') else 5.0
            ax.plot(xy[:, 0], xy[:, 1], color=S.WW_TYPES[t][1], lw=lw, solid_capstyle='butt',
                    solid_joinstyle='round', zorder=7 + (0.5 if t.endswith('W') else 0),
                    path_effects=[pe.Stroke(linewidth=lw + 1.4, foreground='#222222'), pe.Normal()])

    # --- cities
    for name, la, lo in cc.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 'o', ms=3, color='#333333', zorder=8)
        ax.text(x + 9000, y + 6000, name, fontsize=8.5, color='#222222', zorder=8,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    # --- past track (dashed) into the current position
    past = [(r['lon'], r['lat']) for r in st.past if r['time'] < st.adv_time] + [(cur.lon, cur.lat)]
    pxy = np.array([tr.transform(*p) for p in past])
    ax.plot(pxy[:, 0], pxy[:, 1], color='black', lw=1.8, ls=(0, (4, 3)), zorder=9)

    # --- forecast track line + markers + local-time labels
    live = st.live_points
    fxy = np.array([tr.transform(p.lon, p.lat) for p in live])
    ax.plot(fxy[:, 0], fxy[:, 1], color='black', lw=1.0, zorder=9)
    rects = []                                    # placed label boxes (x0, y0, x1, y1), projected m
    marks = [tuple(xy) for xy in fxy]
    LW, LH = 150000, 36000
    def box_for(x, y, dx, dy, ha):
        x0 = x + dx if ha == 'left' else x + dx - LW if ha == 'right' else x + dx - LW / 2
        return (x0, y + dy - LH / 2, x0 + LW, y + dy + LH / 2)
    def free(b):
        if any(b[0] < r[2] and r[0] < b[2] and b[1] < r[3] and r[1] < b[3] for r in rects):
            return False
        return not any(b[0] - 22000 < mx < b[2] + 22000 and b[1] - 22000 < my < b[3] + 22000 for mx, my in marks)
    for p, (x, y) in zip(live, fxy):
        letter = S.marker_letter(p.status, p.vmax)
        trop = p.tropical
        ax.add_patch(Circle((x, y), 21000, facecolor='black' if trop else 'white', edgecolor='black',
                            lw=1.4, zorder=10))
        ax.text(x, y - 1500, letter, ha='center', va='center', fontsize=9, weight='bold',
                color='white' if trop else 'black', zorder=11)
    for p, (x, y) in zip(live, fxy):
        lab = S.local_label(p.time, st.tz)
        cands = [(32000, 0, 'left'), (-32000, 0, 'right'), (0, -42000, 'center'), (0, 42000, 'center'),
                 (32000, -40000, 'left'), (-32000, 40000, 'right'), (32000, 40000, 'left'), (-32000, -40000, 'right'),
                 (0, -80000, 'center'), (0, 80000, 'center')]
        pick = next((c for c in cands if free(box_for(x, y, *c))), cands[0])
        dx, dy, ha = pick
        if pick is not cands[0] and abs(dy) >= 80000:
            ax.plot([x, x + dx], [y + (21000 if dy > 0 else -21000), y + dy - (LH / 2 if dy > 0 else -LH / 2)],
                    color='black', lw=0.6, zorder=10)
        rects.append(box_for(x, y, dx, dy, ha))
        ax.text(x + dx, y + dy, lab, ha=ha, va='center', fontsize=9, weight='bold', zorder=11,
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])

    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    for s in ax.spines.values():
        s.set_color('black'); s.set_linewidth(1.2)

    # --- top banner
    ban = fig.add_axes([0, top, 1, 1 - top]); ban.axis('off')
    ban.add_patch(Rectangle((0, 0), 1, 1, transform=ban.transAxes, facecolor='#1c2e4a'))
    ban.text(0.012, 0.5, 'Note: The cone contains the probable path of the storm center but does not show the size '
             'of the storm. Hazardous conditions can occur outside of the cone.', color='white', fontsize=9.5,
             va='center', ha='left')
    ban.text(0.988, 0.5, 'SIMULATED', color='#ffd84a', fontsize=15, weight='bold', va='center', ha='right')

    info_box(fig, ax, st, cfg)

    ax.text(0.5, 0.5, 'SIMULATED', transform=ax.transAxes, fontsize=90, color='black', alpha=0.045,
            ha='center', va='center', rotation=25, weight='bold', zorder=20)
    if cc.get('illustrative', True):
        ax.text(0.006, 0.008, 'Cone radii illustrative (configurable table) — not an official forecast product',
                transform=ax.transAxes, fontsize=7.5, style='italic', color='#333', zorder=21,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

    out = Path(outdir) / cc.get('output', f'{st.atcf}_5day_cone_adv{st.adv_num}.png')
    fig.savefig(out, dpi=150, facecolor='white'); plt.close(fig)
    print('saved', out)
    return out


def info_box(fig, ax, st, cfg):
    """NHC-style legend/info box in the lower right of the map."""
    b = ax.inset_axes([0.625, 0.012, 0.367, 0.318]); b.set_xlim(0, 1); b.set_ylim(0, 1)
    b.set_xticks([]); b.set_yticks([])
    b.patch.set_facecolor('white'); b.patch.set_alpha(0.96)
    for s in b.spines.values():
        s.set_color('black'); s.set_linewidth(1.0)
    b.set_zorder(30)
    cur = st.current
    lt = cur.time + S.timedelta(hours=st.tz['offset_h'])
    h12 = lt.hour % 12 or 12
    ampm = 'AM' if lt.hour < 12 else 'PM'
    bdir, bkt = st.motion()
    y = 0.955
    b.text(0.03, y, st.title, fontsize=12.5, weight='bold', va='top')
    b.text(0.97, y, 'SIMULATED', fontsize=9, weight='bold', color='#b00000', va='top', ha='right')
    y -= 0.085
    b.text(0.03, y, f"{lt.strftime('%A %B')} {lt.day}, {lt.year}   {h12} {ampm} {st.tz['name']}   "
                    f"Advisory {st.adv_num}", fontsize=8.6, va='top')
    y -= 0.068
    b.text(0.03, y, f"Current information:  ×   Center location {abs(cur.lat):.1f} {'N' if cur.lat >= 0 else 'S'} "
                    f"{abs(cur.lon):.1f} {'W' if cur.lon < 0 else 'E'}", fontsize=8.2, va='top')
    y -= 0.06
    b.text(0.03, y, f"Maximum sustained wind {S.kt_to_mph(cur.vmax)} mph     "
                    f"Movement {S.compass16(bdir)} at {S.motion_mph(bkt)} mph", fontsize=8.2, va='top')
    b.plot([0.02, 0.98], [y - 0.07, y - 0.07], color='#999', lw=0.6)

    y -= 0.115
    b.text(0.03, y, 'Forecast positions:', fontsize=8.2, va='center')
    b.add_patch(Circle((0.33, y), 0.018, transform=b.transData, facecolor='black'))
    b.text(0.36, y, 'Tropical Cyclone', fontsize=8.2, va='center')
    b.add_patch(Circle((0.64, y), 0.018, facecolor='white', edgecolor='black', lw=1))
    b.text(0.67, y, 'Post/Potential TC', fontsize=8.2, va='center')
    y -= 0.07
    b.text(0.03, y, 'Sustained winds:  D < 39 mph   S 39-73 mph   H 74-110 mph   M > 110 mph',
           fontsize=7.9, va='center')
    y -= 0.075
    b.text(0.03, y, 'Potential track area:', fontsize=8.2, va='center')
    b.add_patch(Rectangle((0.33, y - 0.025), 0.07, 0.05, facecolor='white', edgecolor='black', lw=0.8))
    b.text(0.415, y, 'Day 1-3', fontsize=8.2, va='center')
    b.add_patch(Rectangle((0.55, y - 0.025), 0.07, 0.05, facecolor='white', edgecolor='black', lw=0.8, hatch='....'))
    b.text(0.635, y, 'Day 4-5', fontsize=8.2, va='center')
    y -= 0.075
    b.text(0.03, y, 'Watches:', fontsize=8.2, va='center')
    for x, t, lab in ((0.2, 'HU.A', 'Hurricane'), (0.5, 'TR.A', 'Trop Stm')):
        b.add_patch(Rectangle((x, y - 0.02), 0.07, 0.04, facecolor=S.WW_TYPES[t][1], edgecolor='black', lw=0.6))
        b.text(x + 0.085, y, lab, fontsize=8.2, va='center')
    y -= 0.065
    b.text(0.03, y, 'Warnings:', fontsize=8.2, va='center')
    for x, t, lab in ((0.2, 'HU.W', 'Hurricane'), (0.5, 'TR.W', 'Trop Stm')):
        b.add_patch(Rectangle((x, y - 0.02), 0.07, 0.04, facecolor=S.WW_TYPES[t][1], edgecolor='black', lw=0.6))
        b.text(x + 0.085, y, lab, fontsize=8.2, va='center')
    y -= 0.065
    b.text(0.03, y, 'Current wind extent:', fontsize=8.2, va='center')
    for x, c, lab in ((0.33, WIND64, 'Hurricane'), (0.6, WIND34, 'Trop Stm')):
        b.add_patch(Rectangle((x, y - 0.02), 0.07, 0.04, facecolor=c, edgecolor='black', lw=0.6))
        b.text(x + 0.085, y, lab, fontsize=8.2, va='center')
    if cfg['cone'].get('show_inland', True) and cfg.get('inland_ww'):
        y -= 0.065
        b.text(0.03, y, 'Inland warnings (local NWS offices) shaded by county', fontsize=7.6, va='center',
               style='italic')


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir)


if __name__ == '__main__':
    main()
