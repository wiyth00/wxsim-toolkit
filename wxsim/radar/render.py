"""Radar-app style viewer for simulated single-site reflectivity.

usage: python -m wxsim.radar.render scenarios/<name>/radar_XXXX.json [--outdir DIR] [--time ISO]

Renders each requested product (base 0.5 deg / composite) with county/state lines, interstates, cities,
and optional overlays pulled from a hazards_*.json (warning polygons, watch outline) plus storm IDs.
"""
import argparse, json, warnings as _pw
from datetime import timedelta
from pathlib import Path

_pw.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colorbar import ColorbarBase
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .. import geo
from . import storms as stm
from .colormaps import reflectivity, REF_LEVELS
from .field import RadarSim

BG = '#0e1116'; LAND = '#1b1f25'; WATER = '#0f1722'
PRODUCT_NAME = {'base': 'Base Reflectivity  0.5°', 'composite': 'Composite Reflectivity'}
WARN_COL = {'Tornado Emergency': '#ff3dff', 'Tornado Warning': '#ff2a2a', 'Severe Thunderstorm Warning': '#ffb000', 'Flash Flood Warning': '#1fbf3f'}


def make_sim(cfg):
    """RadarSim for a radar config (storms + boundary fine lines). Reuse it across frames of a loop."""
    base = Path(cfg.get('_base_dir', '.'))
    blob = json.load(open(base / cfg['storms_file']))
    lines_fn = None
    if blob.get('boundaries') and cfg.get('fine_lines', True):
        from ..env.boundaries import lines_at
        lines_fn = lambda w: lines_at(blob['boundaries'], w)
    return RadarSim(cfg['site'], blob['storms'], seed=cfg.get('seed', 17), lines_fn=lines_fn)


def render(cfg, outdir, when=None, sim=None):
    """Render each product/view at `when`; returns the list of saved paths."""
    matplotlib.rcdefaults()
    base = Path(cfg.get('_base_dir', '.'))
    when = when or cfg['time']
    sim = sim or make_sim(cfg)
    storms = sim.storms
    proj = geo.projector(sim.to_xy)
    lon0, lon1, lat0, lat1 = cfg['view']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    ov = cfg.get('overlays', {})
    tz = cfg.get('tz', {'offset_h': -4, 'name': 'EDT'})
    t = stm.parse_time(when)
    tloc = t + timedelta(hours=tz['offset_h'])
    cmap, norm = reflectivity()

    # overlays from hazards file
    warn_polys, watch_geom = [], None
    if ov.get('hazards_file'):
        from .. import hazards as hz, watch as watchmod
        hc = json.load(open(base / ov['hazards_file']))
        hz.normalize_warnings(hc, base)
        for w in hc.get('warnings', []):
            poly = w.get('polygon') or hz.storm_polygon(w['storm'])
            issue = w.get('issue')
            expire = w.get('expire')
            if issue and expire and not (stm.parse_time(issue) <= t <= stm.parse_time(expire)):
                continue
            warn_polys.append((hz.wlabel(w), Polygon([(lo, la) for la, lo in poly]),
                               w.get('tags', {}).get('damage') in ('CONSIDERABLE', 'DESTRUCTIVE', 'CATASTROPHIC')))
        if ov.get('watch_outline'):
            geoms = []
            cidx = None
            for wt in hc.get('watches', []):
                if wt.get('from_watch_file'):
                    ids, _ = watchmod.watch_counties(json.load(open(base / wt['from_watch_file'])))
                    geoms += [g for _, g in ids]
                if wt.get('counties'):
                    cidx = cidx or hz.county_index()
                    geoms += [cidx[c.lower()][1] for c in wt['counties'] if c.lower() in cidx]
            if geoms:
                watch_geom = unary_union(geoms).buffer(0.003).buffer(-0.003)

    outs = cfg.get('outputs', {})
    views = [(None, cfg['view'])] + [(v['name'], v['view']) for v in cfg.get('extra_views', [])]
    cache, saved = {}, []
    for product in cfg.get('products', ['base']):
        if product not in cache:
            cache[product] = sim.composite(when, clutter=cfg.get('clutter', True)) if product == 'composite' else \
                sim.base(when, clutter=cfg.get('clutter', True))
        for vname, view in views:
            saved.append(_draw(cfg, sim, proj, storms, when, t, tloc, tz, ov, warn_polys, watch_geom, product,
                               cache[product], view, vname, outs, outdir))
    return saved


def _draw(cfg, sim, proj, storms, when, t, tloc, tz, ov, warn_polys, watch_geom, product, dbz, view, vname, outs, outdir):
    cmap, norm = reflectivity()
    lon0, lon1, lat0, lat1 = view
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    if True:
        fig = plt.figure(figsize=(16, 10), facecolor=BG)
        ax = fig.add_axes([0.01, 0.075, 0.98, 0.855])
        ax.set_facecolor(WATER)
        geo.draw(ax, geo.land().intersection(EXT), proj, facecolor=LAND, edgecolor='none', zorder=1)
        X, Y = sim.grid.edges_xy()
        ax.pcolormesh(X, Y, dbz, cmap=cmap, norm=norm, shading='flat', zorder=2, rasterized=True)
        geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#48505c', county_lw=0.35,
                    county_alpha=0.9, state_color='#aab3bf', state_lw=1.2, z_county=3, z_state=4)
        for pts in cfg.get('interstates', {}).values():
            xs, ys = sim.to_xy.transform(*zip(*pts))
            ax.plot(xs, ys, color='#6c7f99', lw=0.9, alpha=0.55, zorder=3.5)
        if watch_geom is not None:
            geo.draw(ax, watch_geom, proj, facecolor='none', edgecolor='#ffe600', lw=1.6, ls=(0, (5, 3)), zorder=5)
        if ov.get('warnings', True):
            for wtype, g, heavy in sorted(warn_polys, key=lambda x: ('Tornado' in x[0], x[0] == 'Tornado Emergency')):
                col = WARN_COL.get(wtype, '#ffffff')
                lw = 2.6 + (1.2 if heavy else 0)
                geo.draw(ax, g, proj, facecolor='none', edgecolor='black', lw=lw + 1.8, zorder=6)
                geo.draw(ax, g, proj, facecolor='none', edgecolor=col, lw=lw, zorder=7)
        if ov.get('range_rings'):
            for rr in (50, 100, 150, 200):
                ax.add_patch(plt.Circle((0, 0), rr, fill=False, ec='#5d6673', lw=0.7, ls=':', zorder=4))
        ax.plot(0, 0, marker='+', ms=11, mew=2, color='#e6edf3', zorder=8)
        ax.text(3, -6, cfg['site']['id'], color='#e6edf3', fontsize=8.5, weight='bold', zorder=8, clip_on=True)
        for name, la, lo in cfg.get('cities', []):
            x, y = sim.to_xy.transform(lo, la)
            ax.plot(x, y, 'o', ms=2.8, color='#e6edf3', zorder=8)
            ax.text(x + 2.2, y + 1.6, name, fontsize=8.8, color='#e6edf3', zorder=8, clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground='#0b0d10')])
        if ov.get('storm_ids'):
            for s in storms:
                if not stm.primitives(s, when):
                    continue
                la, lo = stm.position(s, when)
                x, y = sim.to_xy.transform(lo, la)
                if np.hypot(x, y) > cfg['site'].get('max_range_km', 230) - 5:
                    continue
                th = np.deg2rad(stm.heading(s, when))
                ax.annotate('', xy=(x + 9 * np.sin(th), y + 9 * np.cos(th)), xytext=(x, y), zorder=9,
                            arrowprops=dict(arrowstyle='-|>', color='white', lw=1.3, mutation_scale=10))
                lx, ly = -np.cos(th), np.sin(th)                      # unit vector to the left of motion
                sgn = 1 if s.get('split', 'right') == 'left' else -1      # labels on the outer side of a split pair
                ax.text(x + sgn * 7 * lx - 3 * np.sin(th), y + sgn * 7 * ly - 3 * np.cos(th), s['id'], color='white',
                        fontsize=9, weight='bold', ha='center', va='center', zorder=9, clip_on=True,
                        path_effects=[pe.withStroke(linewidth=2.6, foreground='black')])
        if ov.get('tornado_markers'):
            # TVS-style marker (red inverted triangle) on storms with an ongoing tornado in the storm file
            for s in storms:
                for tor in s.get('meta', {}).get('tornadoes', []):
                    if stm.parse_time(tor['start']) <= t <= stm.parse_time(tor.get('end', tor['start'])):
                        la, lo = stm.meso_position(s, when)
                        x, y = sim.to_xy.transform(lo, la)
                        if np.hypot(x, y) > cfg['site'].get('max_range_km', 230) - 5:
                            continue
                        ax.plot(x, y, marker='v', ms=13, mfc='#ff1f1f', mec='white', mew=1.4, zorder=10, clip_on=True)
        # extent (inner box of the lat/lon view)
        x0, y0 = sim.to_xy.transform(lon0, lat0); x1, y1 = sim.to_xy.transform(lon1, lat1)
        xa, ya = sim.to_xy.transform(lon0, lat1); xb, yb = sim.to_xy.transform(lon1, lat0)
        ax.set_xlim(max(x0, xa), min(x1, xb)); ax.set_ylim(max(y0, yb), min(y1, ya))
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#30363d')

        fig.text(0.012, 0.962, f"{cfg['site']['id']}  {cfg['site'].get('name', '')}", color='white', fontsize=19, weight='bold')
        fig.text(0.012, 0.938, PRODUCT_NAME.get(product, product), color='#9fb3c8', fontsize=12)
        fig.text(0.988, 0.962, f"{t.strftime('%H%MZ')}  •  {tloc.strftime('%I:%M %p').lstrip('0')} {tz['name']}  "
                 f"{tloc.strftime('%a %b %d %Y')}", color='white', fontsize=14, ha='right', weight='bold')
        fig.text(0.988, 0.938, cfg.get('scenario_label', 'SIMULATED RADAR (not real data)'), color='#ff7b72',
                 fontsize=10, ha='right', style='italic')
        cax = fig.add_axes([0.012, 0.03, 0.42, 0.018])
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', ticks=REF_LEVELS[::2], extend='neither')
        cb.outline.set_edgecolor('#30363d'); cax.tick_params(colors='#c9d1d9', labelsize=8.5)
        fig.text(0.44, 0.039, 'dBZ', color='#c9d1d9', fontsize=9, va='center')
        present = {w[0] for w in warn_polys} if ov.get('warnings', True) else set()
        short = {'Severe Thunderstorm Warning': 'Severe Tstm Warning'}
        legend = [(short.get(k, k), WARN_COL[k]) for k in WARN_COL if k in present]
        if 'Tornado Emergency' not in present and present:
            legend = [(short.get(k, k), WARN_COL[k]) for k in ('Tornado Warning', 'Severe Thunderstorm Warning')]
        if watch_geom is not None:
            legend.append(('Tornado Watch', '#ffe600'))
        xx = 0.99
        for lab, col in reversed(legend):
            txt = fig.text(xx, 0.039, lab, color='#c9d1d9', fontsize=9.5, ha='right', va='center')
            fig.canvas.draw()
            bb = txt.get_window_extent().transformed(fig.transFigure.inverted())
            fig.add_artist(plt.Line2D([bb.x0 - 0.032, bb.x0 - 0.008], [0.039, 0.039], color=col, lw=3,
                                      transform=fig.transFigure))
            xx = bb.x0 - 0.05
        name = outs.get(product, f"{cfg['site']['id']}_{product}_{t.strftime('%Y%m%d%H%M')}.png")
        if vname:
            name = name.replace('.png', f'_{vname}.png')
        out = Path(outdir) / name
        fig.savefig(out, dpi=cfg.get('dpi', 150), facecolor=BG); plt.close(fig)
        if not cfg.get('_quiet'):
            print('saved', out, f'(max {np.nanmax(dbz):.0f} dBZ)')
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir'); ap.add_argument('--time')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg['_base_dir'] = str(Path(a.config).parent)
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir, a.time)


if __name__ == '__main__':
    main()
