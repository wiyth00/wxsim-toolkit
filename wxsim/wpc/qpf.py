"""WPC-style rainfall map(s) from a qpf_*.json (rule-based field in wxsim.wpc.rain).

usage: python -m wxsim.wpc.qpf scenarios/<name>/qpf_XXXX.json [--outdir DIR]

`panels` draws one map per entry side by side, each choosing which parts of the field to include, e.g. the
observed predecessor rain alone, then the storm total. City values are sampled from the same grid.
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .. import geo
from . import rain

LEVELS = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
COLORS = ['#c9f0c9', '#7fd67f', '#2e9e2e', '#0d6e0d', '#86c1f0', '#3a8fd9', '#1450b4', '#fff27a', '#f5c400',
          '#f59300', '#e8531a', '#c40000', '#8c0000', '#e05ce0', '#9d2fc4', '#5e1d9c', '#d2d2d2', '#8a8a8a']
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(LEVELS, CMAP.N)


def draw_panel(ax, tr, proj, cfg, LON, LAT, field, title):
    lon0, lon1, lat0, lat1 = cfg['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    ax.set_facecolor('#dfe7ee')
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#fbfaf5', edgecolor='none', zorder=1)
    X, Y = tr.transform(LON, LAT)
    f = np.where(field >= LEVELS[0], field, np.nan)
    ax.contourf(X, Y, f, levels=LEVELS, cmap=CMAP, norm=NORM, zorder=2, extend='max')
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#555', county_lw=0.2, county_alpha=0.35,
                state_color='black', state_lw=1.0, z_county=3, z_state=4,
                county_box=geo.extent_box(*cfg.get('county_extent', cfg['extent']), pad=0))
    for name, la, lo in cfg.get('cities', []):
        v = rain.sample(LON, LAT, field, la, lo)
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 'o', ms=3, color='black', zorder=6)
        ax.text(x + 6000, y + 5000, f'{name} {v:.1f}"', fontsize=8.5, weight='bold', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    j, i = np.unravel_index(np.nanargmax(np.where(np.isnan(f), -1, f)), f.shape)
    x, y = tr.transform(LON[j, i], LAT[j, i])
    ax.plot(x, y, 'X', ms=9, color='black', mec='white', zorder=7)
    ax.text(x, y - 22000, f'max {field[j, i]:.1f}"', ha='center', va='top', fontsize=9, weight='bold', zorder=7,
            path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    ax.set_title(title, fontsize=12.5, weight='bold', loc='left')
    for s in ax.spines.values():
        s.set_linewidth(1.0)


def render(cfg, outdir):
    matplotlib.rcdefaults()
    base = cfg.get('_base_dir', '.')
    lon0, lon1, lat0, lat1 = cfg['extent']
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    panels = cfg.get('panels', [{'title': 'Storm total rainfall', 'parts': ['pre', 'track', 'orog']}])
    n = len(panels)
    fig = plt.figure(figsize=(7.6 * n, 8.4), facecolor='white')
    fig.text(0.012, 0.975, cfg.get('title', 'Rainfall'), fontsize=17, weight='bold', va='top')
    fig.text(0.012, 0.935, cfg.get('subtitle', ''), fontsize=10.5, va='top', color='#333')
    fig.text(0.988, 0.975, 'SIMULATED', fontsize=17, weight='bold', va='top', ha='right', color='#b00000')
    stats = []
    for k, p in enumerate(panels):
        ax = fig.add_axes([0.01 + k / n, 0.1, 1 / n - 0.02, 0.76])
        LON, LAT, field, _ = rain.build(cfg, base, parts=tuple(p['parts']))
        draw_panel(ax, tr, proj, cfg, LON, LAT, field, p['title'])
        stats.append((p['title'], float(np.nanmax(field))))
    cax = fig.add_axes([0.15, 0.045, 0.7, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation='horizontal',
                      ticks=LEVELS[:-1])
    cb.ax.set_xticklabels([f'{v:g}' for v in LEVELS[:-1]], fontsize=8.5)
    cb.set_label('inches', fontsize=9)
    fig.text(0.988, 0.012, 'Rule-based simulated rainfall field — not an observation or official forecast',
             fontsize=7.5, style='italic', ha='right', color='#555')
    out = Path(outdir) / cfg.get('output', 'QPF.png')
    fig.savefig(out, dpi=140, facecolor='white'); plt.close(fig)
    print('saved', out, '|', '; '.join(f'{t}: max {m:.1f} in' for t, m in stats))
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
