"""SPC-style Mesoscale Discussion graphic from an MCD JSON.

usage: python -m wxsim.mcd scenarios/<name>/mcd_XXXX.json [--outdir DIR]

The "satellite" background is synthesized from the scenario boundaries: clear skies west of a dryline,
a cumulus field (with cloud streets) east of it, an optional stable-air stratus deck (e.g. a CAD wedge),
optional towers and cirrus. MLCAPE is a smooth analytic field shaped by the same boundaries.
Also prints the SPC LAT...LON string for the MCD polygon so the text product can reuse it.
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon as MPoly
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, rotate
import shapely
from shapely.geometry import shape, Polygon, LineString
from pyproj import Transformer

from . import geo
from .util import latlon_string

INTENSITY = LinearSegmentedColormap.from_list('mpi', ['#fff6d5', '#ffd27a', '#ff9f3a', '#f2531d',
                                                      '#c60b0b', '#8e0000', '#ff1fe0'])
sig = lambda x: 1 / (1 + np.exp(-x))


def synth_fields(cfg, LO, LA, is_land, rng):
    env = cfg['environment']
    dl = env['dryline']
    dry_lon = np.interp(LA, [p[1] for p in dl], [p[0] for p in dl])
    d_dry = LO - dry_lon
    ny, nx = LA.shape

    def noise(s):
        n = gaussian_filter(rng.standard_normal(LA.shape), s)
        return n / n.std()

    if env.get('stable_front'):
        sf = env['stable_front']
        poly = Polygon(sf['line'] + sf['close_via'])
        dw = shapely.distance(shapely.points(LO.ravel(), LA.ravel()), LineString(sf['line'])).reshape(LA.shape)
        inside = shapely.contains_xy(poly, LO.ravel(), LA.ravel()).reshape(LA.shape)
        d_st = np.where(inside, dw, -dw)
    else:
        d_st = np.full(LA.shape, -9.0)

    base = np.where(is_land, 0.27 + 0.03 * noise(9), 0.10 + 0.01 * noise(10))
    if env.get('mountains'):
        m = env['mountains']
        base = base - sig((LA - m['lat_min']) / 0.25) * sig((-LO + m['lon_max']) / 0.3) * 0.04
    strat_edge = sig((d_st - 0.05 + 0.08 * noise(3)) / 0.04)
    stratus = strat_edge * (0.62 + 0.12 * noise(12) + 0.05 * noise(3))
    east = sig(d_dry / 0.03) * (1 - strat_edge)
    cu_band = east * (0.35 + 0.65 * np.exp(-np.clip(d_dry, 0, None) / env.get('cu_band_width', 0.55)))
    for lo, la, r in env.get('mesolows', []):
        cu_band = cu_band + 0.5 * np.exp(-(((LO - lo) / r) ** 2 + ((LA - la) / r) ** 2)) * east
    cu_band = np.clip(cu_band, 0, 1.2)
    raw = rng.standard_normal((int(ny * 1.6), int(nx * 1.6)))
    raw = rotate(gaussian_filter(raw, (1.3, 2.6)), env.get('cloud_street_angle', 18), reshape=False, mode='wrap')
    oy, ox = (raw.shape[0] - ny) // 2, (raw.shape[1] - nx) // 2
    speck = raw[oy:oy + ny, ox:ox + nx]
    speck = (speck - np.percentile(speck, 62)) / speck.std()
    speck = speck * (0.7 + 0.3 * sig(noise(10)))
    cu = np.clip(speck, 0, None) ** 0.8 * 0.36 * cu_band * is_land
    for lo, la, r in env.get('towers', []):
        cu = cu + 0.55 * np.exp(-(((LO - lo) / r) ** 2 + ((LA - la) / r) ** 2))
    cirrus = 0
    if env.get('cirrus_south_of') is not None:
        cirrus = sig((env['cirrus_south_of'] - LA) / 0.25) * np.clip(
            gaussian_filter(rng.standard_normal(LA.shape), (1.2, 14)), 0, None) * 0.35
    sat = np.maximum(base + cu + cirrus, stratus * is_land + base * (1 - strat_edge))
    sat = np.clip(sat + 0.012 * noise(1.5), 0, 1)

    c = env['cape']
    cape = (c['base'] - c['per_deg_lat'] * (LA - c['lat_ref'])) * sig((d_dry - 0.12) / 0.14)
    cape = cape * (1 - 0.93 * sig(d_st / 0.12))
    if c.get('east_taper_lon') is not None:
        cape = cape * np.clip(1 - 0.25 * sig((LO - c['east_taper_lon']) / 0.3), 0, 1)
    cape = np.where(is_land, gaussian_filter(cape + 90 * noise(20), 10), np.nan)
    return sat, np.clip(cape, 50, None)


def render(cfg, outdir):
    matplotlib.rcdefaults()
    rng = np.random.default_rng(cfg.get('seed', 417))
    lon0, lon1, lat0, lat1 = cfg['extent']
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    land = geo.land().intersection(EXT)

    nx, ny = cfg.get('grid', [900, 640])
    lons = np.linspace(lon0 - 0.3, lon1 + 0.3, nx); lats = np.linspace(lat0 - 0.3, lat1 + 0.3, ny)
    LO, LA = np.meshgrid(lons, lats)
    is_land = shapely.contains_xy(land, LO.ravel(), LA.ravel()).reshape(LA.shape)
    sat, cape = synth_fields(cfg, LO, LA, is_land, rng)

    fig = plt.figure(figsize=(14.4, 10.5), facecolor='#1b1b1b')
    SB = 0.235
    ax = fig.add_axes([SB, 0.0, 1 - SB, 1.0])
    X, Y = tr.transform(LO, LA)
    ax.pcolormesh(X, Y, sat, cmap='gray', vmin=0, vmax=1, shading='auto', zorder=0, rasterized=True)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#5b5b5b', county_lw=0.3, county_alpha=0.7,
                state_color='black', state_lw=1.6, z_county=2, z_state=3)
    for pts in cfg.get('interstates', {}).values():
        xs, ys = tr.transform(*zip(*pts))
        ax.plot(xs, ys, color='#2350d6', lw=1.5, alpha=0.85, zorder=4, solid_capstyle='round')

    lvls = cfg.get('cape_levels', [500, 1000, 1500, 2000, 2500, 3000])
    ccols = ['#f7c46c', '#f5a142', '#f07e1f', '#e25a12', '#c9380c', '#a8200a', '#8a1208'][:len(lvls)]
    cs = ax.contour(X, Y, cape, levels=lvls, colors=ccols, linewidths=1.6, zorder=5)
    for t in ax.clabel(cs, fmt='%d', fontsize=8, inline=True, inline_spacing=4):
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])

    sh = cfg['environment'].get('shear', {'base': 52, 'per_deg_lat': 5.5, 'dir': 248, 'dir_per_deg_lat': -4})
    bx, by, bu, bv = [], [], [], []
    for la in np.arange(lat0 + 0.35, lat1 - 0.1, 0.42):
        for lo in np.arange(lon0 + 0.3, lon1 - 0.1, 0.5):
            i = np.argmin(abs(lats - la)); j = np.argmin(abs(lons - lo))
            if not is_land[i, j] or cape[i, j] < 150:
                continue
            spd = sh['base'] + sh['per_deg_lat'] * (la - lat0) + rng.normal(0, 2)
            wd = np.deg2rad(sh['dir'] + sh['dir_per_deg_lat'] * (la - lat0) + rng.normal(0, 3))
            bu.append(-spd * np.sin(wd)); bv.append(-spd * np.cos(wd))
            x, y = tr.transform(lo, la); bx.append(x); by.append(y)
    ax.barbs(bx, by, bu, bv, length=6.2, color='#1e7be0', linewidth=1.2, zorder=6, sizes=dict(emptybarb=0.0))

    for stn, la, lo, t, td, wd, ws in cfg.get('obs', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 'o', ms=5, mfc='white', mec='black', mew=0.9, zorder=8)
        ax.barbs([x], [y], [-ws * np.sin(np.deg2rad(wd))], [-ws * np.cos(np.deg2rad(wd))], length=6.0,
                 color='black', linewidth=1.0, zorder=7, sizes=dict(emptybarb=0.0))
        kw = dict(fontsize=8.2, weight='bold', zorder=9, ha='right',
                  path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.text(x - 5500, y + 4500, f'{t}', color='#d61f1f', va='bottom', **kw)
        ax.text(x - 5500, y - 4500, f'{td}', color='#15a02a', va='top', **kw)
    for name, la, lo in cfg.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.text(x + 8000, y - 16000, name, fontsize=9.2, weight='bold', color='#111', zorder=10,
                path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])

    poly = cfg['polygon']  # [lat, lon], closed
    mx, my = tr.transform([q[1] for q in poly], [q[0] for q in poly])
    ax.plot(mx, my, color='#ff1fe0', lw=4.5, zorder=11, solid_joinstyle='round')
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    for s in ax.spines.values():
        s.set_visible(False)
    bpos = cfg.get('banner_pos', [0.55, 0.955])
    ax.text(*bpos, cfg['banner'], transform=ax.transAxes, ha='center', va='top', fontsize=13.5, weight='bold',
            color='#ff1fe0', zorder=12, linespacing=1.25,
            bbox=dict(boxstyle='square,pad=0.45', facecolor='white', edgecolor='#ff1fe0', lw=1.2))
    ax.text(0.995, 0.006, cfg.get('scenario_label', 'SIMULATED'), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color='white', style='italic', zorder=12, bbox=dict(facecolor='#000000aa', edgecolor='none', pad=2))

    # sidebar
    sb = fig.add_axes([0, 0, SB, 1]); sb.set_xlim(0, 1); sb.set_ylim(0, 1); sb.axis('off')
    sb.add_patch(plt.Rectangle((0, 0), 1, 1, color='#232323', zorder=0))
    W = dict(color='white', ha='center', zorder=2)
    sb.text(0.5, 0.965, 'Storm Prediction Center', fontsize=12, style='italic', color='#ff6b5b', ha='center', weight='bold')
    sb.text(0.5, 0.945, 'Norman, Oklahoma', fontsize=8.5, color='#c9c9c9', ha='center')
    sb.text(0.5, 0.875, f"Mesoscale\nDiscussion\n#{cfg['number']}", fontsize=25, weight='bold', va='center', linespacing=1.05, **W)
    sb.text(0.5, 0.768, f"Valid Until:\n{cfg['valid_until']}", fontsize=10.5, color='#d5d5d5', ha='center', va='center', linespacing=1.3)
    sb.text(0.5, 0.69, 'Concerning:\n' + '\n'.join(cfg['concerning']) + f"\nWatch Probability: {cfg['watch_probability']}",
            fontsize=11.5, weight='bold', va='center', linespacing=1.3, **W)
    ints = cfg.get('intensities', [])
    if ints:
        sb.text(0.5, 0.612, 'Most Probable Peak Intensity', fontsize=9.5, weight='bold', **W)
        y = 0.575
        for it in ints:
            sb.add_patch(MPoly([[0.04, y - 0.021], [0.33, y - 0.021], [0.37, y], [0.33, y + 0.021], [0.04, y + 0.021]],
                               closed=True, facecolor='#4a4a4a', edgecolor='#9a9a9a', lw=0.8, zorder=2))
            sb.text(0.07, y, it['label'], color='white', fontsize=11, weight='bold', va='center', zorder=3)
            sb.add_patch(FancyBboxPatch((0.38, y - 0.021), 0.58, 0.042, boxstyle='square,pad=0',
                                        facecolor=INTENSITY(it['level']), edgecolor='#9a9a9a', lw=0.8, zorder=2))
            sb.text(0.67, y, it['value'], color='white' if it['level'] > 0.62 else '#1b1b1b', fontsize=11,
                    weight='bold', va='center', ha='center', zorder=3)
            y -= 0.055
        sb.imshow(np.linspace(0, 1, 256)[None, :], extent=(0.14, 0.9, 0.395, 0.41), cmap=INTENSITY, aspect='auto', zorder=2)
        sb.text(0.1, 0.402, 'Min', color='white', fontsize=8, ha='right', va='center', weight='bold')
        sb.text(0.94, 0.402, 'Max', color='white', fontsize=8, ha='left', va='center', weight='bold')
    sb.text(0.5, 0.355, 'Fields Plotted', fontsize=15, weight='bold', **W)
    sb.plot([0.2, 0.8], [0.341, 0.341], color='white', lw=1)
    sb.text(0.5, 0.29, '\n'.join(cfg['fields_plotted']), fontsize=9.2, weight='bold', va='center', linespacing=1.35, **W)

    ins = fig.add_axes([0.012, 0.015, SB - 0.024, 0.2]); ins.set_facecolor('#dfe9f2')
    ix0, ix1, iy0, iy1 = cfg.get('inset_extent', [-92.5, -76.0, 28.2, 37.8])
    bx_ = geo.extent_box(ix0, ix1, iy0, iy1)
    tri = Transformer.from_crs('EPSG:4326', f'+proj=lcc +lat_1=33 +lat_2=45 +lat_0={(iy0 + iy1) / 2} '
                               f'+lon_0={(ix0 + ix1) / 2} +units=m', always_xy=True)
    for f in geo.states():
        g = shape(f['geometry']).buffer(0).intersection(bx_)
        for pg in ([g] if isinstance(g, Polygon) else list(getattr(g, 'geoms', []))):
            if isinstance(pg, Polygon) and not pg.is_empty:
                xs, ys = tri.transform(*pg.exterior.xy)
                ins.fill(xs, ys, facecolor='white', edgecolor='#444', lw=0.6)
    xs, ys = tri.transform([q[1] for q in poly], [q[0] for q in poly])
    ins.fill(xs, ys, facecolor='#ff7df0', edgecolor='#ff1fe0', lw=1.6, alpha=0.9)
    for ab, la, lo in cfg.get('inset_labels', []):
        x, y = tri.transform(lo, la)
        ins.text(x, y, ab, fontsize=7.5, ha='center', va='center', color='#333', weight='bold')
    a0 = tri.transform(ix0 + 1.5, iy0); a1 = tri.transform(ix1 - 0.5, iy1)
    ins.set_xlim(a0[0], a1[0]); ins.set_ylim(a0[1], a1[1]); ins.set_xticks([]); ins.set_yticks([])

    out = Path(outdir) / cfg.get('output', f"MCD{cfg['number']}_graphic.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print('saved', out)
    print('LAT...LON for the text product:\n' + latlon_string(poly))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir)


if __name__ == '__main__':
    main()
