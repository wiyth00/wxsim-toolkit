"""SPC-style convective outlook maps (categorical + tornado/wind/hail probabilities) from an outlook JSON.

usage: python -m wxsim.outlook scenarios/<name>/outlook_XXX.json [--outdir DIR]

Polygons are lists of [lon, lat] control points drawn roughly by hand; they are smoothed (Chaikin),
nested (each inner level forced inside its outer level), and clipped to land.

Intensity is drawn one of two ways per hazard:
  "cig": {"1": [...], "2": [...], "3": [...]}   Conditional Intensity Groups, SPC's format since March 2026.
         CIG1 = dashed hatch, CIG2 = solid hatch, CIG3 = cross-hatch. Tornado and wind use 1-3, hail 1-2.
  "sig": [...]                                   the older single "significant severe" hatch (pre-2026 events).

Consistency checks (printed as WARNINGs):
  * every probability band, split by intensity group, sits inside the categorical risk it implies, using SPC's
    2026 probability x CIG matrix below (the old two-column table for "sig" events; override with "conversion");
  * every categorical area is backed by some probability/CIG combination that implies it.
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

from . import geo

CAT_ORDER = ['TSTM', 'MRGL', 'SLGT', 'ENH', 'MDT', 'HIGH']
CAT_FILL = {'TSTM': '#c1e9c1', 'MRGL': '#7dc580', 'SLGT': '#f6f67f', 'ENH': '#e6c27f', 'MDT': '#e67f7f', 'HIGH': '#fe7ffe'}
CAT_EDGE = {'TSTM': '#55bb55', 'MRGL': '#005500', 'SLGT': '#dddd00', 'ENH': '#ff9600', 'MDT': '#cc0000', 'HIGH': '#ff00ff'}
CAT_LABEL = {'TSTM': 'General Thunder', 'MRGL': '1 – Marginal', 'SLGT': '2 – Slight', 'ENH': '3 – Enhanced',
             'MDT': '4 – Moderate', 'HIGH': '5 – High'}
RANK = {c: i for i, c in enumerate(CAT_ORDER)}
PCOL = {  # (fill, edge)
    'tornado': {2: ('#80c580', '#008b00'), 5: ('#c9a086', '#8b4726'), 10: ('#ffe680', '#e6b400'),
                15: ('#ff8080', '#e00000'), 30: ('#ff80ff', '#d400d4'), 45: ('#c68ef0', '#912cee'),
                60: ('#7fa0c8', '#104e8b')},
    'other': {5: ('#c9a086', '#8b4726'), 15: ('#ffe680', '#e6b400'), 30: ('#ff8080', '#e00000'),
              45: ('#ff80ff', '#d400d4'), 60: ('#c68ef0', '#912cee'),
              75: ('#7fa0c8', '#104e8b'), 90: ('#8f8fbf', '#2e2e6e')},   # 75/90% wind levels added in 2026
}
TITLES = {'tornado': 'Tornado Probability', 'wind': 'Damaging Wind Probability', 'hail': 'Large Hail Probability',
          'severe': 'Severe Probability'}

# Probability x Conditional Intensity Group -> categorical risk: SPC's 2026 conversion matrix.
# Each row is [<CIG1, CIG1, CIG2, CIG3]; None = combination SPC does not use. Hail stops at CIG2.
# Override per scenario with "conversion": {"hail": {"45": ["ENH", "ENH", "MDT"]}, ...}.
CONVERSION = {
    'tornado': {2: ['MRGL', 'MRGL', 'SLGT', None], 5: ['SLGT', 'SLGT', 'ENH', None],
                10: ['SLGT', 'ENH', 'ENH', 'ENH'], 15: ['ENH', 'ENH', 'MDT', 'MDT'],
                30: ['ENH', 'MDT', 'HIGH', 'HIGH'], 45: ['ENH', 'MDT', 'HIGH', 'HIGH'],
                60: ['ENH', 'HIGH', 'HIGH', 'HIGH']},
    'wind': {5: ['MRGL', 'MRGL', 'SLGT', None], 15: ['SLGT', 'SLGT', 'ENH', None],
             30: ['SLGT', 'ENH', 'ENH', None], 45: ['ENH', 'ENH', 'MDT', 'HIGH'],
             60: ['ENH', 'MDT', 'HIGH', 'HIGH'], 75: ['ENH', 'MDT', 'HIGH', 'HIGH'],
             90: ['ENH', 'MDT', 'HIGH', 'HIGH']},
    'hail': {5: ['MRGL', 'MRGL', 'SLGT'], 15: ['SLGT', 'SLGT', 'ENH'], 30: ['SLGT', 'ENH', 'ENH'],
             45: ['ENH', 'ENH', 'MDT'], 60: ['ENH', 'MDT', 'MDT']},
}
# Pre-2026 table for events drawn with the single "sig" hatch: [no hatch, hatched]. Day 3 "any severe" too.
LEGACY_CONVERSION = {
    'tornado': {2: ['MRGL', 'MRGL'], 5: ['SLGT', 'SLGT'], 10: ['ENH', 'ENH'], 15: ['ENH', 'MDT'],
                30: ['MDT', 'HIGH'], 45: ['HIGH', 'HIGH'], 60: ['HIGH', 'HIGH']},
    'wind': {5: ['MRGL', 'MRGL'], 15: ['SLGT', 'SLGT'], 30: ['ENH', 'ENH'], 45: ['ENH', 'MDT'],
             60: ['MDT', 'HIGH']},
    'hail': {5: ['MRGL', 'MRGL'], 15: ['SLGT', 'SLGT'], 30: ['ENH', 'ENH'], 45: ['ENH', 'MDT'], 60: ['MDT', 'MDT']},
    'severe': {5: ['MRGL', 'MRGL'], 15: ['SLGT', 'SLGT'], 30: ['ENH', 'ENH'], 45: ['ENH', 'MDT'], 60: ['MDT', 'MDT']},
}
CIG_LABEL = {
    'tornado': {1: 'Intensity 1: EF2+ possible', 2: 'Intensity 2: EF3+ possible', 3: 'Intensity 3: EF4+ possible'},
    'wind': {1: 'Intensity 1: 75+ mph gusts', 2: 'Intensity 2: 85+ mph, derecho possible',
             3: 'Intensity 3: 95+ mph, high-end derecho'},
    'hail': {1: 'Intensity 1: 2+ in. hail', 2: 'Intensity 2: 3.5+ in. hail'},
    'severe': {1: 'Intensity 1', 2: 'Intensity 2', 3: 'Intensity 3'},
}
CIG_MAX = {'tornado': 3, 'wind': 3, 'hail': 2, 'severe': 3}
# hatch styles: (angles, dash pattern or None for solid)
CIG_STYLE = {1: ((45,), (0, (5, 3.5))), 2: ((45,), None), 3: ((45, -45), None), 'sig': ((45,), None)}


# ---------------------------------------------------------------------------------------------- hatching
def hatch(ax, g_proj, level, spacing=16000, lw=1.25, zorder=8, color='black'):
    """Draw diagonal hatch lines (dashed / solid / crossed) clipped to a projected (Multi)Polygon."""
    if g_proj.is_empty:
        return
    g_proj = g_proj.buffer(0)
    angles, dashes = CIG_STYLE[level]
    x0, y0, x1, y1 = g_proj.bounds
    cx, cy, R = (x0 + x1) / 2, (y0 + y1) / 2, np.hypot(x1 - x0, y1 - y0)
    for ang in angles:
        a = np.radians(ang)
        d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
        for off in np.arange(-R, R + spacing, spacing):
            p0 = np.array([cx, cy]) + n * off - d * R; p1 = np.array([cx, cy]) + n * off + d * R
            seg = LineString([p0, p1]).intersection(g_proj)
            for part in getattr(seg, 'geoms', [seg]):
                if part.is_empty or part.geom_type != 'LineString':
                    continue
                xs, ys = part.xy
                ax.add_line(Line2D(xs, ys, color=color, lw=lw, zorder=zorder, solid_capstyle='butt',
                                   **({'linestyle': dashes} if dashes else {})))
    for p in getattr(g_proj, 'geoms', [g_proj]):
        if p.geom_type == 'Polygon':
            xs, ys = p.exterior.xy
            ax.plot(xs, ys, color=color, lw=lw * 1.1, zorder=zorder,
                    **({'linestyle': dashes} if (dashes and level != 'sig') else {}))


class HatchKey:
    def __init__(self, level):
        self.level = level


class HatchHandler(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
        angles, dashes = CIG_STYLE[orig.level]
        box = Polygon([(xd, yd), (xd + w, yd), (xd + w, yd + h), (xd, yd + h)])
        arts = [Rectangle((xd, yd), w, h, facecolor='white', edgecolor='black', lw=1.1, transform=trans,
                          **({'linestyle': (0, (3, 2))} if (dashes and orig.level != 'sig') else {}))]
        step = w / 3.2
        for ang in angles:
            sgn = 1 if ang > 0 else -1
            for x0 in np.arange(xd - w, xd + 2 * w, step):
                seg = LineString([(x0, yd if sgn > 0 else yd + h), (x0 + h, yd + h if sgn > 0 else yd)])
                seg = LineString([(x0 - 2 * h, yd - 2 * h * sgn + (0 if sgn > 0 else h)),
                                  (x0 + 3 * h, yd + 3 * h * sgn + (0 if sgn > 0 else h))]).intersection(box)
                if seg.is_empty or seg.geom_type != 'LineString':
                    continue
                xs, ys = seg.xy
                arts.append(Line2D(xs, ys, color='black', lw=1.0, transform=trans,
                                   **({'linestyle': (0, (2.5, 1.8))} if dashes else {})))
        return arts


# ---------------------------------------------------------------------------------------------- figure
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


def _finish(fig, ax, tr, proj, cfg, clip_ext, handles, note, out, handler_map=None, ncol=None, labels=None):
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
    labels = labels or [h.get_label() for h in handles]
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=ncol or len(handles), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.045), handlelength=2.2, handleheight=1.2, columnspacing=2.0,
               handler_map=handler_map or {})
    if note:
        fig.text(0.5, 0.025, note, ha='center', fontsize=9.5, color='#444')
    fig.savefig(out, dpi=150, facecolor='white'); plt.close(fig)
    print('saved', out)


def _conversion(cfg, style):
    base = LEGACY_CONVERSION if style == 'sig' else CONVERSION
    table = {hz: {k: list(v) for k, v in rows.items()} for hz, rows in base.items()}
    if 'severe' not in table:
        table['severe'] = {k: list(v) for k, v in LEGACY_CONVERSION['severe'].items()}
    for hz, over in cfg.get('conversion', {}).items():
        for lvl, row in over.items():
            table.setdefault(hz, {})[int(lvl)] = list(row)
    return table


def _intensity(spec, land_c):
    """-> ({level: geom}, style) where style is 'cig' or 'sig'."""
    if spec.get('cig'):
        return {int(k): geo.smooth_area(v).intersection(land_c).buffer(0) for k, v in spec['cig'].items()}, 'cig'
    if spec.get('sig'):
        return {1: geo.smooth_area(spec['sig']).intersection(land_c)}, 'sig'
    return {}, None


# ---------------------------------------------------------------------------------------------- render
def render(cfg, outdir):
    matplotlib.rcdefaults()
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
            Path(outdir) / outs.get(cfg.get('categorical_output_key', 'categorical'), 'outlook_categorical.png'))

    # ---- probabilistic
    implied = {c: [] for c in CAT_ORDER}          # what the probabilities support, per category
    for hz in ('tornado', 'wind', 'hail', 'severe'):
        spec = cfg.get('probabilistic', {}).get(hz)
        if not spec:
            continue
        levels = sorted(int(k) for k in spec['levels'])
        geoms = geo.nest([spec['levels'][str(l)] for l in levels], land_c)
        inten, style = _intensity(spec, land_c)
        pal = PCOL['tornado' if hz == 'tornado' else 'other']
        fig, ax, tr, proj, _ = _figure(cfg, TITLES[hz])
        geo.draw(ax, land_c, proj, facecolor='#fbfbf8', edgecolor='none', zorder=1)
        for i, (lvl, g) in enumerate(zip(levels, geoms)):
            geo.draw(ax, g, proj, facecolor=pal[lvl][0], edgecolor=pal[lvl][1], lw=1.7, alpha=0.9, zorder=2 + i)
        # consistency: split each probability band by intensity group and check the category each piece implies
        conv = _conversion(cfg, style).get(hz, {})
        tops = sorted(inten)
        for i, (lvl, g) in enumerate(zip(levels, geoms)):
            band = g.difference(geoms[i + 1]) if i + 1 < len(geoms) else g
            row = conv.get(lvl)
            if row is None:
                print(f'  WARNING: no conversion row for {hz} {lvl}%.')
                continue
            for k in [0] + tops:
                if k == 0:
                    piece = band.difference(unary_union([inten[j] for j in tops])) if tops else band
                else:
                    piece = band.intersection(inten[k]).difference(inten[k + 1]) if k + 1 in inten else band.intersection(inten[k])
                if piece.is_empty or piece.area < 0.01:
                    continue
                tag = 'no intensity' if k == 0 else ('sig hatch' if style == 'sig' else f'CIG{k}')
                need = row[k] if k < len(row) else None
                if need is None:
                    print(f'  WARNING: {hz} {lvl}% with {tag} is a combination SPC does not use.')
                    continue
                implied[need].append(piece)
                if need not in cat_geoms:
                    print(f'  WARNING: {hz} {lvl}% with {tag} implies {need}, but no {need} polygon is defined.')
                    continue
                spill = piece.difference(cat_geoms[need].buffer(0.15)).area
                if spill > 0.02:
                    print(f'  WARNING: {hz} {lvl}% with {tag} implies {need}, but ~{spill:.2f} sq deg of it lies '
                          f'outside the {need} area.')
        if style == 'cig':
            for k in inten:
                if k > CIG_MAX[hz]:
                    print(f'  WARNING: {hz} has no CIG{k} (max is CIG{CIG_MAX[hz]}).')
            for k in sorted(inten):
                outer = inten.get(k - 1)
                if outer is not None and inten[k].difference(outer.buffer(0.1)).area > 0.02:
                    print(f'  WARNING: {hz} CIG{k} extends outside CIG{k - 1}.')
                if inten[k].difference(geoms[0].buffer(0.1)).area > 0.02:
                    print(f'  WARNING: {hz} CIG{k} extends outside the lowest probability contour.')
        handles = [Patch(facecolor=pal[l][0], edgecolor=pal[l][1], lw=1.5, label=f'{l}%') for l in levels]
        hmap = {HatchKey: HatchHandler()}
        if style == 'cig':
            for k in sorted(inten):
                inner = inten.get(k + 1)
                ring = inten[k] if inner is None else inten[k].difference(inner)
                hatch(ax, proj(ring), k, zorder=8 + k)
                key = HatchKey(k); handles.append(key)
            labels = [f'{l}%' for l in levels] + [CIG_LABEL[hz].get(k, f'Intensity {k}') for k in sorted(inten)]
        elif style == 'sig':
            hatch(ax, proj(inten[1]), 'sig')
            handles.append(HatchKey('sig'))
            labels = [f'{l}%' for l in levels] + ['Significant severe']
        else:
            labels = [f'{l}%' for l in levels]
        ncol = len(handles) if len(handles) <= 6 else (len(levels) if len(levels) >= len(handles) - len(levels) else 4)
        _finish(fig, ax, tr, proj, cfg, clip_ext, handles, spec.get('note'),
                Path(outdir) / outs.get(hz, f'outlook_{hz}.png'), handler_map=hmap, ncol=ncol, labels=labels)

    # ---- is every categorical area backed by the probabilities?
    for c in cats:
        if c in ('TSTM',):
            continue
        support = unary_union([p for cc in CAT_ORDER[RANK[c]:] for p in implied[cc]]) if any(
            implied[cc] for cc in CAT_ORDER[RANK[c]:]) else Polygon()
        extra = cat_geoms[c].difference(support.buffer(0.2)).area
        if extra > 0.05:
            ctr = cat_geoms[c].difference(support.buffer(0.2)).representative_point()
            print(f'  WARNING: ~{extra:.2f} sq deg of the {c} area (near {ctr.y:.1f}N {-ctr.x:.1f}W) has no '
                  f'probability/intensity combination that implies {c}.')


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
