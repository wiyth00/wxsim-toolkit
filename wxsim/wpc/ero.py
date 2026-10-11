"""WPC-style Excessive Rainfall Outlook graphic + discussion from an ero_*.json.

usage: python -m wxsim.wpc.ero scenarios/<name>/ero_XXXX.json [--outdir DIR]

Risk comes from the same rainfall field as the QPF map (wxsim.wpc.rain):
  ratio = (rain inside the valid window) / (flash flood guidance)
  FFG   = max(ffg_min, ffg_base - ffg_per_inch_antecedent * antecedent rain)   <- soaked ground lowers FFG
The ratio is smoothed and contoured at `ratio_thresholds` for Marginal/Slight/Moderate/High (WPC thresholds
are 5/15/30/50% probability of exceeding FFG within 25 mi; the ratio levels are the illustrative proxy).
Optional `manual` polygons ([lon, lat] rings per category) replace the computed areas.
The discussion text is hand-written in the JSON and wrapped with nws_text.wrap_product().
"""
import argparse, json, warnings
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon

from .. import geo
from ..nws_text import wrap_product
from ..tropical.storm import parse_z
from . import rain

CATS = ['MRGL', 'SLGT', 'MDT', 'HIGH']
FILL = {'MRGL': '#7fe07f', 'SLGT': '#ffff66', 'MDT': '#ff4d4d', 'HIGH': '#ff66ff'}
EDGE = {'MRGL': '#1e8c1e', 'SLGT': '#c8b400', 'MDT': '#b30000', 'HIGH': '#b300b3'}
LABEL = {'MRGL': 'Marginal (at least 5%)', 'SLGT': 'Slight (at least 15%)', 'MDT': 'Moderate (at least 30%)',
         'HIGH': 'High (at least 50%)'}


def risk_field(cfg, base):
    """Window rainfall and antecedent rainfall -> ratio to FFG.
    window rain  = track rain inside [start, end] + pre_in_window * (predecessor rain), times orography
    antecedent   = antecedent_pre * (predecessor rain) + track rain from the track start to the window start
    (older configs: 'antecedent': 'pre' means antecedent_pre = 1)."""
    rcfg = json.load(open(Path(base) / cfg['rain_config']))
    t0, t1 = parse_z(cfg['valid']['start']), parse_z(cfg['valid']['end'])
    LON, LAT, trk, _ = rain.build(rcfg, base, window=(t0, t1), parts=('track',))
    _, _, pre, _ = rain.build(rcfg, base, parts=('pre',))
    orog = rain.orog_factor(LON, LAT, rcfg.get('orog', []))
    fin = cfg.get('pre_in_window', 0.0)
    qpf = trk * orog + fin * pre * orog
    a_pre = cfg.get('antecedent_pre', 1.0 if cfg.get('antecedent', 'pre') == 'pre' else 0.0)
    ts = parse_z(rcfg['track']['start'])
    ante = a_pre * pre
    if t0 > ts:
        _, _, trk_before, _ = rain.build(rcfg, base, window=(ts, t0), parts=('track',))
        ante = ante + trk_before * orog
    f = cfg.get('ffg', {})
    ffg = np.maximum(f.get('min', 0.8), f.get('base', 3.5) - f.get('per_inch_antecedent', 0.25) * ante)
    ratio = gaussian_filter(qpf / ffg, cfg.get('smooth', 4))
    return LON, LAT, qpf, ffg, ratio


def render(cfg, outdir):
    matplotlib.rcdefaults()
    base = cfg.get('_base_dir', '.')
    LON, LAT, qpf, ffg, ratio = risk_field(cfg, base)
    # optional "exclude": [lon, lat] rings where no risk is drawn (e.g. where the precipitation falls as snow or ice,
    # or where the rain in the field falls outside this window); edges are feathered so contours stay smooth
    if cfg.get('exclude'):
        from shapely import contains_xy
        mask = np.zeros(LON.shape)
        for ring in cfg['exclude']:
            mask = np.maximum(mask, contains_xy(Polygon(ring), LON, LAT).astype(float))
        ratio = ratio * (1 - np.clip(gaussian_filter(mask, 3), 0, 1))
    lon0, lon1, lat0, lat1 = cfg['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    fig = plt.figure(figsize=(12.8, 9.6), facecolor='white')
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.86])
    ax.set_facecolor('#ffffff')
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#f7f7f2', edgecolor='none', zorder=1)
    th = cfg.get('ratio_thresholds', {'MRGL': 0.3, 'SLGT': 0.55, 'MDT': 0.85, 'HIGH': 1.2})
    X, Y = tr.transform(LON, LAT)
    land_proj = proj(geo.land().intersection(EXT))
    if cfg.get('manual'):
        geoms = geo.nest([cfg['manual'][c] for c in CATS if c in cfg['manual']], geo.land())
        for c, g in zip([c for c in CATS if c in cfg['manual']], geoms):
            geo.draw(ax, g, proj, facecolor=FILL[c], edgecolor=EDGE[c], lw=1.6, zorder=2 + CATS.index(c))
    else:
        for k, c in enumerate(CATS):
            cs = ax.contourf(X, Y, ratio, levels=[th[c], 1e9], colors=[FILL[c]], zorder=2 + k)
            ax.contour(X, Y, ratio, levels=[th[c]], colors=[EDGE[c]], linewidths=1.6, zorder=2.5 + k)
        # mask water: draw the ocean back over the contours
        ocean = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0).difference(geo.land())
        geo.draw(ax, ocean, proj, facecolor='#ffffff', edgecolor='none', zorder=7)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#777', county_lw=0.2, county_alpha=0.4,
                state_color='black', state_lw=1.1, z_county=8, z_state=9,
                county_box=geo.extent_box(*cfg.get('county_extent', cfg['extent']), pad=0))
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='none', edgecolor='#333', lw=0.8, zorder=9)
    for name, la, lo in cfg.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 'o', ms=3, color='black', zorder=10)
        ax.text(x + 6000, y + 5000, name, fontsize=9, zorder=10,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    t0, t1 = parse_z(cfg['valid']['start']), parse_z(cfg['valid']['end'])
    fig.text(0.012, 0.965, f"Excessive Rainfall Outlook — Day {cfg.get('day', 1)}", fontsize=19, weight='bold',
             va='top')
    fig.text(0.012, 0.918, f"Valid {t0:%HZ %a %b} {t0.day} {t0.year} – {t1:%HZ %a %b} {t1.day} {t1.year}   •   "
             f"Issued {cfg['issued_local']}", fontsize=11, va='top')
    fig.text(0.988, 0.965, 'SIMULATED', fontsize=18, weight='bold', va='top', ha='right', color='#b00000')
    handles = [Patch(facecolor=FILL[c], edgecolor=EDGE[c], label=LABEL[c]) for c in CATS]
    lg = ax.legend(handles=handles, loc='lower right', fontsize=10, framealpha=0.95,
                   title='Probability of rainfall exceeding flash flood\nguidance within 25 miles of a point',
                   title_fontsize=9.5)
    lg.set_zorder(20)
    ax.text(0.008, 0.01, cfg.get('note', 'Risk areas derived from the simulated rainfall field vs. simulated FFG.'),
            transform=ax.transAxes, fontsize=7.5, style='italic', zorder=21,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    out = Path(outdir) / cfg.get('output', f"ERO_day{cfg.get('day', 1)}.png")
    fig.savefig(out, dpi=150, facecolor='white'); plt.close(fig)
    areas = {c: int((ratio >= th[c]).sum()) for c in CATS}
    print('saved', out, '| grid cells per category:', areas)

    txt = cfg.get('text')
    if txt:
        day = cfg.get('day', 1)
        L = ['000', f"FOUS30 KWBC {cfg['wmo_time']}", 'QPFERD', '', 'Excessive Rainfall Discussion',
             'NWS Weather Prediction Center College Park MD', cfg['issued_local'], f'Day {day}',
             f"Valid {t0:%HZ %a %b %d %Y} - {t1:%HZ %a %b %d %Y}".replace(' 0', ' '), '']
        for h in txt['headlines']:
            L += [f'...{h}...', '']
        if txt.get('update'):
            L += [f"...{txt['update']}...", '']
        for region, paras in txt['regions']:
            L += [f'...{region}...', ''] + sum(([p, ''] for p in paras), [])
        L += [f"..{txt['forecaster']}..", '',
              f"Day {day} threat area: www.wpc.ncep.noaa.gov/qpf/{ {1: '94e', 2: '98e', 3: '99e'}[day] }points.txt",
              '']
        tout = Path(outdir) / cfg.get('text_output', f'QPFERD_day{day}.txt')
        tout.write_text(wrap_product('\n'.join(L)))
        print('saved', tout)
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
