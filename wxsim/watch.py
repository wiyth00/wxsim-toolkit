"""SPC-style watch graphic (county-based) from a watch JSON, plus the axis sentence and WWP corner points.

usage: python -m wxsim.watch scenarios/<name>/watch_XXXX.json [--outdir DIR]

The watch area is defined the way SPC words it: "along and N statute miles north and south (or east and
west) of a line from A miles DIR of CITY to B miles DIR of CITY". Counties whose interior point falls in
that parallelogram are included, then trimmed by optional per-state filters.
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MPoly, Rectangle
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

from . import geo
from .util import offset_point, COMPASS, DIR_WORDS, latlon_code

OUTLINE = {'Tornado': '#e8232a', 'Severe Thunderstorm': '#1f5fd6'}


def prob_color(v):
    if v < 5: return '#fff9b0'
    if v < 20: return '#ffd24d'
    if v < 40: return '#ff9a1a'
    if v < 70: return '#ff1a1a'
    return '#ff00ff'


def axis_points(ax_cfg):
    pts = []
    for end in (ax_cfg['from'], ax_cfg['to']):
        lat, lon = offset_point(end['lat'], end['lon'], end['miles'], COMPASS[end['dir']])
        pts.append((lon, lat))
    return pts


def watch_counties(cfg):
    (lo1, la1), (lo2, la2) = axis_points(cfg['axis'])
    hw = cfg['axis']['half_width_mi']
    if cfg['axis'].get('orientation', 'north_south') == 'north_south':
        d = hw / 69.0
        para = Polygon([(lo1, la1 - d), (lo1, la1 + d), (lo2, la2 + d), (lo2, la2 - d)])
    else:
        import math
        d1 = hw / (69.0 * math.cos(math.radians(la1))); d2 = hw / (69.0 * math.cos(math.radians(la2)))
        para = Polygon([(lo1 - d1, la1), (lo1 + d1, la1), (lo2 + d2, la2), (lo2 - d2, la2)])
    filt = cfg.get('state_filters', {})
    keep = []
    for f in geo.counties():
        fid = str(f['id']).zfill(5)
        if fid[:2] not in cfg['state_fips']:
            continue
        g = shape(f['geometry']).buffer(0)
        c = g.representative_point()
        if not para.contains(c):
            continue
        r = filt.get(fid[:2], {})
        if ('lon_min' in r and c.x < r['lon_min']) or ('lon_max' in r and c.x > r['lon_max']) or \
           ('lat_min' in r and c.y < r['lat_min']) or ('lat_max' in r and c.y > r['lat_max']):
            continue
        keep.append(g)
    return keep, para


def render(cfg, outdir):
    matplotlib.rcdefaults()
    counties_in, para = watch_counties(cfg)
    wu = unary_union(counties_in).buffer(0.004).buffer(-0.004)
    lon0, lon1, lat0, lat1 = cfg['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.5)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    wtype = cfg.get('type', 'Tornado')

    fig = plt.figure(figsize=(12.8, 7.2), facecolor='#2d3238')
    ban = fig.add_axes([0.018, 0.862, 0.964, 0.118]); ban.axis('off')
    ban.add_patch(Rectangle((0, 0), 1, 1, transform=ban.transAxes, facecolor='#ffff8c',
                            edgecolor=OUTLINE[wtype], lw=3))
    ban.text(0.5, 0.47, f'{wtype} Watch', ha='center', va='center', fontsize=40, color='black')
    if cfg.get('pds'):
        fig.text(0.635, 0.832, 'THIS IS A PARTICULARLY DANGEROUS SITUATION', ha='center', va='center',
                 fontsize=17.5, weight='bold', color='#ff1fe0')

    ax = fig.add_axes([0.28, 0.018, 0.705, 0.785])
    ax.set_facecolor('#77a1d8')
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#fdf8e1', edgecolor='none', zorder=1)
    geo.draw(ax, wu, proj, facecolor='#ffff8c', edgecolor=OUTLINE[wtype], lw=5.5, zorder=2, joinstyle='round')
    geo.draw(ax, wu, proj, facecolor='#ffff8c', edgecolor='none', zorder=3)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#8f8f8f', county_lw=0.35, county_alpha=0.8,
                state_color='black', state_lw=1.8, z_county=4, z_state=5)
    for name, la, lo in cfg.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, '+', ms=6, mew=1.3, color='black', zorder=6)
        ax.text(x + 7000, y + 5000, name, fontsize=11, weight='bold', color='black', zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    for s in ax.spines.values():
        s.set_color('black'); s.set_linewidth(1.2)
    ax.text(0.5, 0.012, f"Updated: {cfg['updated']}", transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
            zorder=10, bbox=dict(facecolor='white', edgecolor='black', lw=0.8, pad=2.5))
    ax.text(0.99, 0.012, cfg.get('scenario_label', 'SIMULATED'), transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, style='italic', color='#8a1c1c', zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

    lp = fig.add_axes([0.0, 0.0, 0.275, 0.84]); lp.set_xlim(0, 1); lp.set_ylim(0, 1); lp.axis('off')
    lp.text(0.5, 0.905, 'Watch Valid Until:\n' + cfg['valid_until'], ha='center', va='center', color='white',
            fontsize=15, linespacing=1.15)
    lp.text(0.5, 0.775, 'Hazard Probabilities', ha='center', va='center', color='white', fontsize=15)
    y = 0.705; h = 0.074
    for lab, val, v in cfg['probabilities']:
        lp.add_patch(MPoly([[0.07, y - h / 2], [0.62, y - h / 2], [0.67, y], [0.62, y + h / 2], [0.07, y + h / 2]],
                           closed=True, facecolor='#1c2a8c', edgecolor='white', lw=1.2))
        lp.text(0.36, y, lab.replace('\\n', '\n'), ha='center', va='center', color='white', fontsize=13.5, linespacing=1.0)
        lp.add_patch(Rectangle((0.665, y - h / 2), 0.29, h, facecolor=prob_color(v), edgecolor='white', lw=1.2))
        lp.text(0.81, y, val, ha='center', va='center', color='black', fontsize=17, weight='bold')
        y -= h + 0.019
    lp.text(0.07, 0.125, f"Max Expected Hail: {cfg['max_hail']}", color='white', fontsize=13.5, va='center')
    lp.text(0.07, 0.065, f"Max Expected Wind:  {cfg['max_wind']}", color='white', fontsize=13.5, va='center')
    fig.text(0.14, 0.022, f"{wtype} Watch #{cfg['number']}  •  Storm Prediction Center", ha='center', va='center',
             color='white', fontsize=9.5, bbox=dict(facecolor='black', edgecolor='white', lw=0.8, pad=3))

    out = Path(outdir) / cfg.get('output', f"WW{cfg['number']}_graphic.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print('saved', out, f'({len(counties_in)} counties)')

    a = cfg['axis']; f_, t_ = a['from'], a['to']
    ns = 'north and south' if a.get('orientation', 'north_south') == 'north_south' else 'east and west'
    print(f"SEL axis: approximately along and {a['half_width_mi']} statute miles {ns} of a line from "
          f"{f_['miles']} miles {DIR_WORDS[f_['dir']]} of {f_['city']} to {t_['miles']} miles {DIR_WORDS[t_['dir']]} of {t_['city']}.")
    corners = list(para.exterior.coords)[:4]
    print('WWP LAT...LON ' + ' '.join(latlon_code(la, lo) for lo, la in corners))


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
