"""SPC-style convective outlook maps (categorical + tornado/wind/hail probabilities) from an outlook JSON.

usage: python -m wxsim.outlook scenarios/<name>/outlook_XXX.json [--outdir DIR]

Polygons are lists of [lon, lat] control points drawn roughly by hand; they are smoothed (Chaikin),
nested (each inner level forced inside its outer level), and clipped to land. The script also checks
that every probability contour sits inside the categorical risk it implies, and warns if not.
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon

from . import geo

plt.rcParams['hatch.linewidth'] = 1.1
CAT_ORDER = ['TSTM', 'MRGL', 'SLGT', 'ENH', 'MDT', 'HIGH']
CAT_FILL = {'TSTM': '#c1e9c1', 'MRGL': '#7dc580', 'SLGT': '#f6f67f', 'ENH': '#e6c27f', 'MDT': '#e67f7f', 'HIGH': '#fe7ffe'}
CAT_EDGE = {'TSTM': '#55bb55', 'MRGL': '#005500', 'SLGT': '#dddd00', 'ENH': '#ff9600', 'MDT': '#cc0000', 'HIGH': '#ff00ff'}
CAT_LABEL = {'TSTM': 'General Thunder', 'MRGL': '1 – Marginal', 'SLGT': '2 – Slight', 'ENH': '3 – Enhanced',
             'MDT': '4 – Moderate', 'HIGH': '5 – High'}
PCOL = {  # (fill, edge)
    'tornado': {2: ('#80c580', '#008b00'), 5: ('#c9a086', '#8b4726'), 10: ('#ffe680', '#e6b400'),
                15: ('#ff8080', '#e00000'), 30: ('#ff80ff', '#d400d4'), 45: ('#c68ef0', '#912cee'),
                60: ('#7fa0c8', '#104e8b')},
    'other': {5: ('#c9a086', '#8b4726'), 15: ('#ffe680', '#e6b400'), 30: ('#ff8080', '#e00000'),
              45: ('#ff80ff', '#d400d4'), 60: ('#c68ef0', '#912cee')},
}
# Day 1 probability -> minimum categorical risk it implies
IMPLIES = {'tornado': {2: 'MRGL', 5: 'SLGT', 10: 'ENH', 15: 'MDT', 30: 'HIGH'},
           'wind': {5: 'MRGL', 15: 'SLGT', 30: 'ENH', 45: 'MDT', 60: 'HIGH'},
           'hail': {5: 'MRGL', 15: 'SLGT', 30: 'ENH', 45: 'MDT', 60: 'MDT'}}
TITLES = {'tornado': 'Tornado Probability', 'wind': 'Damaging Wind Probability', 'hail': 'Large Hail Probability'}


def _figure(cfg, title):
    ext = cfg['extent']
    tr = geo.transformer(lat_0=cfg.get('proj_lat0', 34.0), lon_0=cfg.get('proj_lon0', -84.0))
    proj = geo.projector(tr)
    fig = plt.figure(figsize=(14, 10.2), facecolor='white')
    ax = fig.add_axes([0.02, 0.1, 0.96, 0.8])
    fig.text(0.02, 0.955, f"{cfg.get('day_label', 'Day 1')} {title}", fontsize=20, weight='bold', color='#111')
    fig.text(0.02, 0.925, f"NWS Storm Prediction Center  •  Issued {cfg['issued']}  •  Valid {cfg['valid']}",
             fontsize=11, color='#333')
    fig.text(0.98, 0.955, f"Forecaster: {cfg['forecaster']}", fontsize=10, color='#333', ha='right')
    fig.text(0.98, 0.93, cfg.get('scenario_label', 'Simulated scenario'), fontsize=9.5, color='#8a1c1c',
             ha='right', style='italic')
    return fig, ax, tr, proj, ext


def _finish(fig, ax, tr, proj, cfg, clip_ext, handles, note, out):
    ext = cfg['extent']
    cbox = geo.extent_box(ext[0], ext[1], ext[2], ext[3], pad=-0.3)
    geo.basemap(ax, proj, clip_ext, county_box=cbox, draw_land=False)
    for name, lat, lon in cfg.get('cities', geo.CITIES_SE):
        x, y = tr.transform(lon, lat)
        ax.plot(x, y, 'o', ms=3.6, color='#1d2227', zorder=12)
        ax.text(x + 9000, y + 7000, name, fontsize=8.6, color='#1d2227', zorder=12)
    geo.set_extent(ax, tr, *ext)
    for s in ax.spines.values():
        s.set_color('#4a4f55'); s.set_linewidth(1.2)
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.045), handlelength=2.2, handleheight=1.2, columnspacing=2.2)
    if note:
        fig.text(0.5, 0.025, note, ha='center', fontsize=9.5, color='#444')
    fig.savefig(out, dpi=150, facecolor='white'); plt.close(fig)
    print('saved', out)


def render(cfg, outdir):
    matplotlib.rcdefaults()
    plt.rcParams['hatch.linewidth'] = 1.1
    ce = cfg.get('clip_extent', [cfg['extent'][0] - 2, cfg['extent'][1] + 0.6, cfg['extent'][2] - 0.6, cfg['extent'][3] + 0.6])
    clip_ext = geo.extent_box(ce[0], ce[1], ce[2], ce[3], pad=0)
    land_c = geo.land().intersection(clip_ext)
    outs = cfg.get('outputs', {})

    # ---- categorical
    cats = [c for c in CAT_ORDER if c in cfg['categorical']]
    cat_geoms = dict(zip(cats, geo.nest([cfg['categorical'][c] for c in cats], land_c)))
    fig, ax, tr, proj, _ = _figure(cfg, 'Convective Outlook')
    geo.draw(ax, land_c, proj, facecolor='#fbfbf8', edgecolor='none', zorder=1)
    for i, c in enumerate(cats):
        geo.draw(ax, cat_geoms[c], proj, facecolor=CAT_FILL[c], edgecolor=CAT_EDGE[c], lw=1.6, zorder=2 + i)
    handles = [Patch(facecolor=CAT_FILL[c], edgecolor=CAT_EDGE[c], lw=1.5, label=CAT_LABEL[c]) for c in CAT_ORDER]
    _finish(fig, ax, tr, proj, cfg, clip_ext, handles, cfg.get('categorical_note'),
            Path(outdir) / outs.get('categorical', 'outlook_categorical.png'))

    # ---- probabilistic
    for hz in ('tornado', 'wind', 'hail'):
        spec = cfg.get('probabilistic', {}).get(hz)
        if not spec:
            continue
        levels = sorted(int(k) for k in spec['levels'])
        geoms = geo.nest([spec['levels'][str(l)] for l in levels], land_c)
        pal = PCOL['tornado' if hz == 'tornado' else 'other']
        fig, ax, tr, proj, _ = _figure(cfg, TITLES[hz])
        geo.draw(ax, land_c, proj, facecolor='#fbfbf8', edgecolor='none', zorder=1)
        for i, (lvl, g) in enumerate(zip(levels, geoms)):
            geo.draw(ax, g, proj, facecolor=pal[lvl][0], edgecolor=pal[lvl][1], lw=1.7, alpha=0.9, zorder=2 + i)
            need = IMPLIES[hz].get(lvl)
            if need and need in cat_geoms:
                spill = g.difference(cat_geoms[need].buffer(0.15)).area
                if spill > 0.02:
                    print(f'  WARNING: {hz} {lvl}% extends outside the {need} categorical area '
                          f'(~{spill:.2f} sq deg). Enlarge {need} or shrink the {lvl}% contour.')
            elif need and need not in cat_geoms:
                print(f'  WARNING: {hz} {lvl}% implies {need}, but no {need} polygon is defined.')
        handles = [Patch(facecolor=pal[l][0], edgecolor=pal[l][1], lw=1.5, label=f'{l}%') for l in levels]
        if spec.get('sig'):
            hg = Polygon(geo.chaikin(spec['sig'])).intersection(land_c)
            geo.draw(ax, hg, proj, facecolor='none', edgecolor='black', hatch='///', lw=1.4, zorder=8)
            handles.append(Patch(facecolor='white', edgecolor='black', hatch='///', lw=1.2,
                                 label='Significant severe (≥10%)'))
        _finish(fig, ax, tr, proj, cfg, clip_ext, handles, spec.get('note'),
                Path(outdir) / outs.get(hz, f'outlook_{hz}.png'))


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
