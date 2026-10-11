"""NWS-style active hazards map: storm-based warning polygons over county-shaded watches/advisories.

usage: python -m wxsim.hazards scenarios/<name>/hazards_XXXX.json [--outdir DIR]

Warnings can be given as explicit polygons ([lat, lon] points) or generated from a storm:
  "storm": {"lat", "lon", "dir_from", "speed_kt", "minutes", "back_mi", "front_mi"}
which builds the usual trapezoid (narrow behind the storm, wide downstream at the expiration time).
Counties are shaded by their highest-priority active watch/advisory. For every warning the script also
prints the UGC county line, LAT...LON and TIME...MOT...LOC lines for the text product.
"""
import argparse, json, math, warnings as _pywarnings
from pathlib import Path

_pywarnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

from . import geo, watch as watchmod
from .util import offset_point

# Standard NWS hazard-map colors
COLORS = {
    'Tornado Emergency': '#C000C0', 'Tornado Warning': '#FF0000', 'Severe Thunderstorm Warning': '#FFA500', 'Flash Flood Warning': '#8B0000',
    'Special Marine Warning': '#FFA500', 'Tornado Watch': '#FFFF00', 'Severe Thunderstorm Watch': '#DB7093',
    'Flash Flood Watch': '#2E8B57', 'Flood Watch': '#2E8B57', 'Red Flag Warning': '#FF1493',
    'Fire Weather Watch': '#FFDEAD', 'Wind Advisory': '#D2B48C', 'High Wind Warning': '#DAA520',
    'Special Weather Statement': '#FFE4B5', 'Heat Advisory': '#FF7F50',
    'Hurricane Warning': '#DC143C', 'Tropical Storm Warning': '#B22222', 'Hurricane Watch': '#FF00FF',
    'Tropical Storm Watch': '#F08080', 'Flash Flood Emergency': '#FF1FA0',
    # wind, winter and coastal hazards (standard NWS map colors)
    'Extreme Wind Warning': '#FF8C00', 'High Wind Watch': '#B8860B',
    'Blizzard Warning': '#FF4500', 'Ice Storm Warning': '#8B008B', 'Winter Storm Warning': '#FF69B4',
    'Winter Storm Watch': '#4682B4', 'Winter Weather Advisory': '#7B68EE',
    'Coastal Flood Warning': '#228B22', 'Coastal Flood Watch': '#66CDAA', 'Coastal Flood Advisory': '#7CFC00',
}
TC_TYPES = {'HU.W': 'Hurricane Warning', 'TR.W': 'Tropical Storm Warning', 'HU.A': 'Hurricane Watch',
            'TR.A': 'Tropical Storm Watch'}
# County-shading priority (lower index wins)
# New winter/coastal types slot in by NWS rank relative to their neighbors; the original order is unchanged.
PRIORITY = ['Extreme Wind Warning', 'Hurricane Warning', 'Blizzard Warning', 'Ice Storm Warning', 'Winter Storm Warning',
            'Tropical Storm Warning', 'Hurricane Watch', 'Tropical Storm Watch', 'Tornado Watch',
            'Severe Thunderstorm Watch', 'Flash Flood Watch', 'Flood Watch', 'High Wind Warning',
            'Coastal Flood Warning', 'Winter Storm Watch', 'High Wind Watch', 'Coastal Flood Watch',
            'Red Flag Warning', 'Fire Weather Watch', 'Heat Advisory', 'Winter Weather Advisory', 'Wind Advisory',
            'Coastal Flood Advisory', 'Special Weather Statement']
STATE_FIPS = {'AL': '01', 'FL': '12', 'GA': '13', 'KY': '21', 'MS': '28', 'NC': '37', 'SC': '45', 'TN': '47',
              'VA': '51', 'TX': '48', 'OK': '40', 'KS': '20', 'AR': '05', 'LA': '22', 'MO': '29', 'IL': '17',
              'IN': '18', 'OH': '39', 'WV': '54', 'NE': '31', 'IA': '19', 'CO': '08', 'NM': '35'}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}
DIR16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def county_index():
    idx = {}
    for f in geo.counties():
        fid = str(f['id']).zfill(5)
        st = FIPS_STATE.get(fid[:2])
        if st:
            idx[f"{f['properties']['name']} {st}".lower()] = (fid, shape(f['geometry']).buffer(0))
    return idx


def hazard_counties(h, cidx):
    """County keys for a watch/hazard entry: its "counties" list plus every county whose interior point falls
    inside its optional "polygon" ([lat, lon] points, like warning polygons), minus any "exclude" names.
    A polygon makes large regional hazards (a winter storm warning across three states) quick to define."""
    keys = [c.lower() for c in h.get('counties', [])]
    if h.get('polygon'):
        area = Polygon([(lo, la) for la, lo in h['polygon']])
        keys += [k for k, (_fid, g) in cidx.items() if area.contains(g.representative_point())]
    drop = {c.lower() for c in h.get('exclude', [])}
    return [k for k in dict.fromkeys(keys) if k not in drop]


def storm_polygon(s):
    """Trapezoid: back edge through a point just behind the storm, front edge at the expiration position."""
    toward = (s['dir_from'] + 180) % 360
    dist = s['speed_kt'] * 1.15078 * s['minutes'] / 60.0 + s.get('front_pad_mi', 3)
    blat, blon = offset_point(s['lat'], s['lon'], s.get('back_pad_mi', 3), (toward + 180) % 360)
    flat, flon = offset_point(s['lat'], s['lon'], dist, toward)
    left, right = (toward - 90) % 360, (toward + 90) % 360
    pts = [offset_point(blat, blon, s['back_mi'] / 2, left), offset_point(flat, flon, s['front_mi'] / 2, left),
           offset_point(flat, flon, s['front_mi'] / 2, right), offset_point(blat, blon, s['back_mi'] / 2, right)]
    return [[round(la, 2), round(lo, 2)] for la, lo in pts]


def ugc_line(fids, expire_ddhhmm):
    by_state = {}
    for fid in sorted(fids):
        by_state.setdefault(FIPS_STATE[fid[:2]], []).append(int(fid[2:]))
    parts = []
    for st, codes in by_state.items():
        runs, start, prev = [], codes[0], codes[0]
        for c in codes[1:] + [None]:
            if c is not None and c == prev + 2:   # county FIPS codes are odd numbers, consecutive = +2
                prev = c; continue
            runs.append(f'{start:03d}' if start == prev else f'{start:03d}>{prev:03d}')
            if c is not None:
                start = prev = c
        parts.append(f'{st}C' + '-'.join(runs))
    return '-'.join(parts) + f'-{expire_ddhhmm}-'


def local_time_str(iso, tz):
    from datetime import timedelta
    from .radar.storms import parse_time
    t = parse_time(iso) + timedelta(hours=tz.get('offset_h', -4))
    return f"{t.strftime('%I:%M %p').lstrip('0')} {tz.get('name', 'EDT')}"


def wcolor(w):
    """Warnings flagged "emergency": true draw in their emergency color (Tornado / Flash Flood Emergency)."""
    if w.get('emergency'):
        return COLORS['Flash Flood Emergency' if w['type'] == 'Flash Flood Warning' else 'Tornado Emergency']
    return COLORS[w['type']]


def wlabel(w):
    if w.get('emergency'):
        lab = 'Flash Flood Emergency' if w['type'] == 'Flash Flood Warning' else 'Tornado Emergency'
    else:
        lab = w['type']
    return ('PDS ' + lab) if w.get('pds') else lab


def legend_color(label):
    return COLORS.get(label.replace('PDS ', ''), '#ccc')


def normalize_warnings(cfg, base):
    """Fill legacy fields for warnings defined by storm_id + issue/expire times (storms file)."""
    if not cfg.get('storms_file'):
        return
    from .radar import storms as stm
    storms = {s['id']: s for s in stm.load(base / cfg['storms_file'])}
    tz = cfg.get('tz', {'offset_h': -4, 'name': 'EDT'})
    for w in cfg.get('warnings', []):
        if 'storm_id' not in w:
            continue
        s = storms[w['storm_id']]
        # "loc": "meso" puts the warning's storm location at the mesocyclone/hook (where radar shows the tornado)
        lat, lon = (stm.meso_position if w.get('loc') == 'meso' else stm.position)(s, w['issue'])
        shp = w.get('shape', {})
        mdir, mspd = stm.motion_at(s, w['issue'])
        w['storm'] = {'lat': round(lat, 3), 'lon': round(lon, 3), 'dir_from': int(round(mdir)),
                      'speed_kt': int(round(mspd)), 'minutes': stm.minutes_between(w['issue'], w['expire']),
                      'back_mi': shp.get('back_mi', 6), 'front_mi': shp.get('front_mi', 16)}
        w.setdefault('expires', local_time_str(w['expire'], tz))
        w.setdefault('expire_ddhhmm', stm.parse_time(w['expire']).strftime('%d%H%M'))
        w.setdefault('issue_z', stm.parse_time(w['issue']).strftime('%H%MZ'))
        if cfg.get('valid'):
            w.setdefault('elapsed_min', stm.minutes_between(w['issue'], cfg['valid']))


def radar_underlay(ax, tr, cfg, base):
    from .radar.field import RadarSim
    from .radar import storms as stm
    from .radar.colormaps import reflectivity
    ru = cfg['radar_underlay']
    rc = json.load(open(base / ru['config']))
    from .radar.render import make_sim
    rc['_base_dir'] = str(base)
    sim = make_sim(rc)                        # includes boundary fine lines for generated storms files
    when = ru.get('time', rc['time'])
    dbz = sim.composite(when, clutter=False) if ru.get('product') == 'composite' else sim.base(when, clutter=False)
    X, Y = sim.grid.edges_xy()
    lon, lat = sim.to_ll.transform(X, Y)
    px, py = tr.transform(lon, lat)
    cmap, norm = reflectivity()
    ax.pcolormesh(px, py, dbz, cmap=cmap, norm=norm, shading='flat', alpha=ru.get('alpha', 0.85), zorder=2.5,
                  rasterized=True)


def render(cfg, outdir):
    matplotlib.rcdefaults()
    COLORS.update(cfg.get('colors', {}))        # per-map overrides, e.g. a blue Tropical Storm Warning for contrast
    normalize_warnings(cfg, Path(cfg.get('_base_dir', '.')))
    lon0, lon1, lat0, lat1 = cfg['extent']
    EXT = geo.extent_box(lon0, lon1, lat0, lat1, pad=1.0)
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    cidx = county_index()
    base = Path(cfg.get('_base_dir', '.'))

    # ---- county-based hazards
    county_haz = {}   # fid -> (priority, type)
    legend_types = []

    def add(fid, htype):
        p = PRIORITY.index(htype) if htype in PRIORITY else 99
        if fid not in county_haz or p < county_haz[fid][0]:
            county_haz[fid] = (p, htype)
        if htype not in legend_types:
            legend_types.append(htype)

    for w in cfg.get('watches', []):
        if w.get('from_watch_file'):
            wcfg = json.load(open(base / w['from_watch_file']))
            ids, _ = watchmod.watch_counties(wcfg)
            for fid, _g in ids:
                add(fid, w['type'])
        for c in hazard_counties(w, cidx):
            add(cidx[c][0], w['type'])
    hatched = []      # (label, color, [geoms]) drawn as overlays, e.g. a Flood Watch under hurricane warnings
    for h in cfg.get('county_hazards', []):
        if h.get('from_tc_file'):
            tcc = json.load(open(base / h['from_tc_file']))
            for blk in tcc.get('inland_ww', []):
                for c in blk['counties']:
                    if c.lower() in cidx:
                        add(cidx[c.lower()][0], TC_TYPES[blk['type']])
            continue
        if h.get('style') == 'hatch':
            gs = [cidx[c][1] for c in hazard_counties(h, cidx) if c in cidx]
            label = ('PDS ' if h.get('pds') else '') + h['type']
            hatched.append((label, COLORS.get(h['type'], '#2E8B57'), gs))
            continue
        for c in hazard_counties(h, cidx):
            key = c
            if key not in cidx:
                print(f'  WARNING: county "{c}" not found (use "Name ST", e.g. "Troup GA")')
                continue
            add(cidx[key][0], h['type'])
    geoms_by_fid = {fid: g for fid, g in cidx.values()}

    fig = plt.figure(figsize=(16, 9.6), facecolor='#1f2329')
    MAPW = 0.70
    ax = fig.add_axes([0.012, 0.075, MAPW, 0.84])
    ax.set_facecolor('#a9c6e8')
    geo.draw(ax, geo.land().intersection(EXT), proj, facecolor='#f4f1e8', edgecolor='none', zorder=1)
    for fid, (_, htype) in county_haz.items():
        geo.draw(ax, geoms_by_fid[fid], proj, facecolor=COLORS.get(htype, '#cccccc'), edgecolor='none',
                 alpha=0.55, zorder=2)
    for label, col, gs in hatched:
        u = unary_union(gs).buffer(0.003).buffer(-0.003)
        geo.draw(ax, u, proj, facecolor='none', edgecolor=col, hatch='///', lw=0, zorder=2.6)
        geo.draw(ax, u, proj, facecolor='none', edgecolor=col, lw=3.0, zorder=2.7)
    if cfg.get('radar_underlay'):
        radar_underlay(ax, tr, cfg, base)
    geo.basemap(ax, proj, EXT, ocean=None, draw_land=False, county_color='#6f6f6f', county_lw=0.35,
                county_alpha=0.8, state_color='black', state_lw=1.6, z_county=3, z_state=4)
    for pts in cfg.get('interstates', {}).values():
        xs, ys = tr.transform(*zip(*pts))
        ax.plot(xs, ys, color='#3b5ba8', lw=1.1, alpha=0.6, zorder=4)

    # ---- warnings
    warns = cfg.get('warnings', [])
    order = sorted(range(len(warns)), key=lambda i: 0 if warns[i]['type'] != 'Tornado Warning' else 1)
    for i in order:
        w = warns[i]
        poly = w.get('polygon') or storm_polygon(w['storm'])
        w['_poly'] = poly
        g = Polygon([(lo, la) for la, lo in poly])
        col = wcolor(w)
        lw = 3.4 if w['type'] == 'Tornado Warning' or w.get('emergency') else 2.6
        if w.get('tags', {}).get('damage') in ('CONSIDERABLE', 'DESTRUCTIVE', 'CATASTROPHIC'):
            lw += 1.4
        geo.draw(ax, g, proj, facecolor=col, edgecolor='none', alpha=0.28, zorder=5)
        geo.draw(ax, g, proj, facecolor='none', edgecolor='black', lw=lw + 1.6, zorder=6, joinstyle='round')
        geo.draw(ax, g, proj, facecolor='none', edgecolor=col, lw=lw, zorder=7, joinstyle='round')
        if wlabel(w) not in legend_types:
            legend_types.insert(0, wlabel(w))
        s = w.get('storm')
        if s:
            now_min = w.get('elapsed_min', 0)
            toward = (s['dir_from'] + 180) % 360
            clat, clon = offset_point(s['lat'], s['lon'], s['speed_kt'] * 1.15078 * now_min / 60, toward)
            x, y = tr.transform(clon, clat)
            alat, alon = offset_point(clat, clon, 9, toward)
            ax2, ay2 = tr.transform(alon, alat)
            ax.annotate('', xy=(ax2, ay2), xytext=(x, y), zorder=9,
                        arrowprops=dict(arrowstyle='-|>', color='black', lw=2.2, mutation_scale=16))
            ax.plot(x, y, marker='o', ms=9, mfc=col, mec='black', mew=1.6, zorder=10)
            ax.text(x, y - 9000, str(i + 1), ha='center', va='top', fontsize=10, weight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    for name, la, lo in cfg.get('cities', []):
        x, y = tr.transform(lo, la)
        ax.plot(x, y, 's', ms=3.5, color='black', zorder=11)
        ax.text(x + 6000, y + 4000, name, fontsize=9.5, weight='bold', color='#111', zorder=11,
                path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)
    for sp in ax.spines.values():
        sp.set_color('black'); sp.set_linewidth(1.2)
    ax.text(0.995, 0.008, cfg.get('scenario_label', 'SIMULATED'), transform=ax.transAxes, ha='right',
            va='bottom', fontsize=8.5, style='italic', color='#8a1c1c', zorder=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

    # ---- header + legend
    fig.text(0.012, 0.955, cfg.get('title', 'Active Watches & Warnings'), fontsize=21, weight='bold', color='white')
    fig.text(0.012, 0.927, f"{cfg.get('subtitle', '')}  •  Valid {cfg['valid_time']}", fontsize=11, color='#c9d1d9')
    handles = [Patch(facecolor=legend_color(t), edgecolor='black', lw=0.8, label=t) for t in legend_types]
    handles += [Patch(facecolor='none', edgecolor=col, hatch='///', lw=1.5, label=label) for label, col, _ in hatched]
    if any(l.startswith('PDS') for l, _, _ in hatched) or any(w.get('pds') for w in cfg.get('warnings', [])):
        fig.text(0.988, 0.955, 'PARTICULARLY DANGEROUS SITUATION', fontsize=15, weight='bold', color='#ff4fd8',
                 ha='right')
    leg = fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.012, 0.005), ncol=min(len(handles), 5),
                     frameon=False, fontsize=10, handlelength=1.8, handleheight=1.1, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color('white')

    # ---- right panel: warning cards
    pn = fig.add_axes([MAPW + 0.022, 0.075, 1 - MAPW - 0.032, 0.84]); pn.set_xlim(0, 1); pn.set_ylim(0, 1)
    pn.axis('off')
    pn.text(0.0, 0.985, f'ACTIVE WARNINGS ({len(warns)})', color='white', fontsize=13, weight='bold', va='top')
    n = max(len(warns), 1)
    ch = min(0.19, 0.94 / n - 0.012)
    y = 0.945
    for i, w in enumerate(warns):
        col = wcolor(w)
        pn.add_patch(FancyBboxPatch((0, y - ch), 1, ch, boxstyle='round,pad=0,rounding_size=0.015',
                                    facecolor='#2b313a', edgecolor='#444c56', lw=0.8))
        pn.add_patch(Rectangle((0, y - ch), 0.035, ch, facecolor=col, edgecolor='none'))
        tx = 0.06
        tcol = {'#FFFF00': 'white', '#8B0000': '#ff6b6b'}.get(col, col)
        pn.text(tx, y - 0.012, f"{i + 1}  {wlabel(w).upper()}", color=tcol,
                fontsize=10.5, weight='bold', va='top')
        pn.text(0.985, y - 0.012, w.get('office', ''), color='#9aa4af', fontsize=8.5, ha='right', va='top')
        # counties touched
        g = Polygon([(lo, la) for la, lo in w['_poly']])
        hit = []
        for key, (fid, cg) in cidx.items():
            if cg.intersects(g) and cg.intersection(g).area / cg.area > 0.03:
                hit.append((fid, key))
        names = sorted({k.rsplit(' ', 1)[0].title() for _, k in hit})
        sts = sorted({k.rsplit(' ', 1)[1].upper() for _, k in hit})
        w['_fids'] = [f for f, _ in hit]
        line2 = f"Until {w['expires']}  •  {', '.join(names)} ({'/'.join(sts)})"
        import textwrap
        wrapped = textwrap.wrap(line2, 50)[:2]
        yy = y - 0.043
        for ln in wrapped:
            pn.text(tx, yy, ln, color='#e6edf3', fontsize=8.6, va='top'); yy -= 0.024
        s = w.get('storm')
        if s:
            toward = (s['dir_from'] + 180) % 360
            pn.text(tx, yy, f"Moving {DIR16[int((toward + 11.25) // 22.5) % 16]} at "
                            f"{round(s['speed_kt'] * 1.15078 / 5) * 5:.0f} mph", color='#9aa4af', fontsize=8.6, va='top')
            yy -= 0.026
        tags = w.get('tags', {})
        bx = tx
        for k in ('tornado', 'flood', 'damage', 'hail', 'wind'):
            if k not in tags:
                continue
            label = {'tornado': 'TORNADO', 'flood': 'FLASH FLOOD', 'damage': 'DAMAGE THREAT', 'hail': 'HAIL',
                     'wind': 'WIND'}[k]
            txt = f'{label}: {tags[k]}'
            hot = tags[k] in ('OBSERVED', 'RADAR CONFIRMED', 'CONSIDERABLE', 'DESTRUCTIVE', 'CATASTROPHIC')
            t = pn.text(bx, yy, txt, fontsize=7.8, weight='bold', va='top', color='white' if hot else '#1b1f24',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#b3121f' if hot else '#e6edf3', edgecolor='none'))
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(pn.transData.inverted())
            bx = bb.x1 + 0.02
            if bx > 0.8:
                bx = tx; yy -= 0.03
        y -= ch + 0.012

    out = Path(outdir) / cfg.get('output', 'hazards_map.png')
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print('saved', out)

    # ---- text-product helpers
    for i, w in enumerate(warns):
        poly = w['_poly']
        ll = ' '.join(f'{int(round(la * 100)):04d} {int(round(abs(lo) * 100)) % 10000:04d}' for la, lo in poly)
        print(f"\n[{i + 1}] {wlabel(w)} ({w.get('office', '')}) until {w['expires']}")
        if w.get('expire_ddhhmm'):
            print('  UGC: ' + ugc_line(w['_fids'], w['expire_ddhhmm']))
        print('  LAT...LON ' + ll)
        s = w.get('storm')
        if s and w.get('issue_z'):
            print(f"  TIME...MOT...LOC {w['issue_z']} {s['dir_from']:03d}DEG {s['speed_kt']}KT "
                  f"{int(round(s['lat'] * 100)):04d} {int(round(abs(s['lon']) * 100)) % 10000:04d}")


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
