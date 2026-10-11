"""SPC-style preliminary storm reports: map + CSV, built from a storms file.

usage: python -m wxsim.reports scenarios/<name>/reports_XXX.json [--outdir DIR]

Where the hazards come from
  * a generated storms file (storms_gen_seedN.json): every storm carries "hazards" rows
    [time, lat, lon, hail_in, peak_gust_kt, tornado_EF or -1] written by wxsim.env.generator;
  * a hand-written storms.json: give each storm a "storm_hazards" entry in the reports config
    ({"S2": {"start", "end", "hail_in", "gust_kt", "tornadoes": [{"start", "end", "ef"}]}}) and rows are
    built every 10 min along the storm's anchor/motion track;
  * several files at once: "storms_file": [{"file": "storms.json"}, {"file": "storms_gen_seed1.json", "ids": ["G03"]}];
  * "extra" reports typed in by hand (type, time, lat, lon, mag, comment).

How reports are drawn from the hazards
  Real report counts follow people, not storms. Each 10-minute storm step can produce a Poisson number of hail
  (>= 1") and wind (>= 50 kt) reports, scaled by how many towns lie within ~12 km ("population" from the bundled
  GeoNames list, data/places_us.csv). Hail sizes are drawn below the storm's peak and snapped to the usual
  reported sizes; most wind reports are damage reports ("UNK"), some are measured. Tornadoes are always
  reported (start, every 10 min along the path, end). Near-duplicates (same type, <3 km, <15 min) are dropped,
  much like SPC's filtered reports.

Outputs
  * <name>.png  SPC-style map: red tornado, blue wind, green hail; black triangles = hail >= 2", black squares =
    wind >= 65 kt. An outlook JSON can be overlaid ("outlook_overlay") to verify the forecast.
  * <name>.csv  SPC today.csv layout (three blocks: tornado / wind / hail; times UTC; hail size in 1/100 in.;
    wind speed in mph or UNK). Locations read like "3 NW Athens", with county and state from the bundled maps.
  The WFO tag in each comment comes from the nearest office in "wfo_offices" (a rough stand-in for CWAs).
"""
import argparse, csv, json, math, warnings
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

from . import geo
from .hazards import FIPS_STATE
from .radar import storms as stormlib
from .radar.storms import parse_time

DIR16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
HAIL_SIZES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 2.75, 3.00, 3.50, 4.00, 4.25, 4.50, 5.00]
HAIL_NAMES = {1.00: 'Quarter', 1.25: 'Half dollar', 1.50: 'Ping pong ball', 1.75: 'Golf ball', 2.00: 'Hen egg',
              2.50: 'Tennis ball', 2.75: 'Baseball', 3.00: 'Teacup', 3.50: 'Large', 4.00: 'Softball',
              4.25: 'Softball', 4.50: 'Grapefruit', 5.00: 'Grapefruit'}
COL = {'torn': '#e60000', 'wind': '#1f5bff', 'hail': '#00a000'}


# ------------------------------------------------------------------------------------------ gazetteer
@lru_cache(maxsize=None)
def places():
    rows = list(csv.DictReader(open(geo.DATA / 'places_us.csv', encoding='utf-8')))
    lat = np.array([float(r['lat']) for r in rows]); lon = np.array([float(r['lon']) for r in rows])
    xy = np.c_[lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57]
    return rows, lat, lon, cKDTree(xy)


def _xy(lat, lon):
    return (lon * 111.32 * math.cos(math.radians(lat)), lat * 110.57)


def pop_factor(lat, lon, radius_km=12):
    n = len(places()[3].query_ball_point(_xy(lat, lon), radius_km))
    return 1 - math.exp(-(n + 0.3) / 2.5)


def bearing(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)); dy = lat2 - lat1
    return math.degrees(math.atan2(dx, dy)) % 360


def location_name(lat, lon):
    rows, plat, plon, tree = places()
    d, i = tree.query(_xy(lat, lon))
    r = rows[i]
    mi = d / 1.609344
    if mi < 1:
        return r['name'], r['state']
    return f"{int(round(mi))} {DIR16[int((bearing(plat[i], plon[i], lat, lon) + 11.25) // 22.5) % 16]} {r['name']}", r['state']


class CountyFinder:
    def __init__(self, ext_box):
        self.geoms, self.meta = [], []
        for f in geo.counties():
            g = shape(f['geometry']).buffer(0)
            if g.intersects(ext_box):
                fid = str(f['id']).zfill(5)
                self.geoms.append(g); self.meta.append((f['properties']['name'], FIPS_STATE.get(fid[:2], '??')))
        self.tree = STRtree(self.geoms)

    def __call__(self, lat, lon):
        p = Point(lon, lat)
        for i in self.tree.query(p):
            if self.geoms[i].contains(p):
                return self.meta[i]
        return None, None


# ------------------------------------------------------------------------------------------ hazard rows
def hazard_rows(cfg, base):
    """-> list of dicts {storm, t, lat, lon, hail, gust, ef, hb} from the storms file (+ storm_hazards)."""
    srcs = cfg['storms_file']
    srcs = [{'file': srcs}] if isinstance(srcs, str) else srcs     # or [{"file": ..., "ids": [...]}, ...]
    storms = []
    for src in srcs:
        sf = json.load(open(base / src['file']))
        storms += [s for s in (sf['storms'] if isinstance(sf, dict) else sf) if not src.get('ids') or s['id'] in src['ids']]
    over = cfg.get('storm_hazards', {})
    rows = []
    for s in storms:
        if s.get('hazards') and s['id'] not in over:
            kf = {k['at']: k['params'] for k in s.get('keyframes', [])}
            for t, la, lo, hail, gust, ef in s['hazards']:
                rows.append(dict(storm=s['id'], t=parse_time(t), lat=la, lon=lo, hail=hail, gust=gust, ef=ef,
                                 hb=kf.get(t, {}).get('hb', 0.0)))
        elif s['id'] in over:
            h = over[s['id']]
            t0, t1 = parse_time(h['start']), parse_time(h['end'])
            tors = [(parse_time(x['start']), parse_time(x['end']), x['ef']) for x in h.get('tornadoes', [])]
            t = t0
            while t <= t1:
                la, lo = stormlib.position(s, t.strftime('%Y-%m-%dT%H:%MZ'))
                ef = next((e for a, b, e in tors if a <= t <= b), -1)
                rows.append(dict(storm=s['id'], t=t, lat=la, lon=lo, hail=h.get('hail_in', 0), gust=h.get('gust_kt', 0),
                                 ef=ef, hb=h.get('high_based', 1.0 if s.get('archetype') == 'high_based_cell' else 0.0)))
                t += timedelta(minutes=10)
            for a, b, e in tors:                         # make sure the tornado's exact start/end are sampled
                for tt in (a, b):
                    la, lo = stormlib.position(s, tt.strftime('%Y-%m-%dT%H:%MZ'))
                    rows.append(dict(storm=s['id'], t=tt, lat=la, lon=lo, hail=0, gust=0, ef=e, hb=0.0))
    return rows


# ------------------------------------------------------------------------------------------ reports
def snap_hail(x):
    return max([v for v in HAIL_SIZES if v <= x + 1e-6] or [1.00])


def _wind_comment(kt, hb, rng):
    if kt >= 70:
        c = rng.choice(['Roof blown off a home and several large trees uprooted.',
                        'Numerous large trees and power poles snapped. Metal outbuilding destroyed.'])
    elif kt >= 60:
        c = rng.choice(['Numerous trees and power lines down across the area.',
                        'Several trees uprooted. Minor roof damage to a few homes.'])
    else:
        c = rng.choice(['Several trees down.', 'Trees and power lines down.', 'Large limbs down on area roads.'])
    if hb > 0.5 and rng.random() < 0.6:
        c += ' Little or no rain with the damaging winds.'
    return c


def build_reports(cfg, base):
    rng = np.random.default_rng(cfg.get('seed', 11))
    rates = {'hail': 0.55, 'wind': 0.45} | cfg.get('report_rate', {})
    w0, w1 = (parse_time(x) for x in cfg['window'])
    out = []
    land_only = cfg.get('land_only', True)          # nobody reports damage from the open ocean
    if land_only:
        import shapely
        land = geo.land()
    for r in sorted(hazard_rows(cfg, base), key=lambda r: r['t']):
        if not (w0 <= r['t'] < w1):
            continue
        if land_only and not shapely.contains_xy(land, r['lon'], r['lat']):
            continue
        pf = pop_factor(r['lat'], r['lon'])
        if r['ef'] >= 0:
            out.append(dict(type='torn', t=r['t'], lat=r['lat'], lon=r['lon'], mag=r['ef'], storm=r['storm']))
        if r['hail'] >= 1.0:
            for _ in range(rng.poisson(rates['hail'] * pf * (0.6 + 0.4 * min(r['hail'] / 2, 2)))):
                dx, dy = rng.normal(0, 4, 2)
                size = snap_hail(max(1.0, r['hail'] * rng.uniform(0.55, 1.0)))
                out.append(dict(type='hail', t=r['t'] + timedelta(minutes=float(rng.uniform(0, 10))),
                                lat=r['lat'] + dy / 110.57, lon=r['lon'] + dx / (111.32 * math.cos(math.radians(r['lat']))),
                                mag=size, storm=r['storm']))
        if r['gust'] >= 50:
            for _ in range(rng.poisson(rates['wind'] * pf * (r['gust'] - 45) / 15)):
                dx, dy = rng.normal(0, 5, 2)
                measured = rng.random() < cfg.get('measured_wind_frac', 0.3)
                mph = int(round(r['gust'] * 1.15078 * rng.uniform(0.85, 1.0)))
                out.append(dict(type='wind', t=r['t'] + timedelta(minutes=float(rng.uniform(0, 10))),
                                lat=r['lat'] + dy / 110.57, lon=r['lon'] + dx / (111.32 * math.cos(math.radians(r['lat']))),
                                mag=mph if measured else None, est=r['gust'], hb=r['hb'], storm=r['storm']))
    for e in cfg.get('extra', []):
        out.append(dict(type={'tornado': 'torn'}.get(e['type'], e['type']), t=parse_time(e['time']), lat=e['lat'],
                        lon=e['lon'], mag=e.get('mag'), est=e.get('mag', 50), hb=0, comment=e.get('comment'),
                        storm='extra', keep=True))
    # de-duplicate (like SPC's filtered reports); tornado reports along one path are kept
    out.sort(key=lambda r: r['t'])
    kept = []
    for r in out:
        win, dist = (240, 1.5) if r['type'] == 'torn' else (900, 3)   # tornado paths keep one report per step
        if not r.get('keep') and any(
                k['type'] == r['type'] and abs((k['t'] - r['t']).total_seconds()) < win and
                math.hypot(*np.subtract(_xy(k['lat'], k['lon']), _xy(r['lat'], r['lon']))) < dist for k in kept):
            continue
        kept.append(r)
    # names, counties, offices, comments
    ext = cfg['extent']
    cf = CountyFinder(geo.extent_box(*ext, pad=1.0))
    offices = cfg.get('wfo_offices', {})
    for r in kept:
        r['loc'], pst = location_name(r['lat'], r['lon'])
        r['county'], r['state'] = cf(r['lat'], r['lon'])
        r['state'] = r['state'] or pst
        r['wfo'] = min(offices, key=lambda k: math.hypot(*np.subtract(_xy(*offices[k]), _xy(r['lat'], r['lon'])))) \
            if offices else None
        if r.get('comment'):
            continue
        if r['type'] == 'hail':
            r['comment'] = f"{HAIL_NAMES[r['mag']]} size hail reported." if r['mag'] >= 1.75 else 'Hail reported.'
            if r['mag'] >= 3.0:
                r['comment'] += ' ' + rng.choice(['Windshields broken.', 'Roof and siding damage reported.',
                                                  'Photo shared on social media.'])
        elif r['type'] == 'wind':
            r['comment'] = _wind_comment(r['est'], r.get('hb', 0), rng) if r['mag'] is None else \
                rng.choice(['Measured by a home weather station.', 'Measured gust at a mesonet site.',
                            'Measured by an ASOS.'])
        else:
            r['comment'] = rng.choice(['Tornado reported by storm chasers.', 'Tornado confirmed by trained spotter.',
                                       'Law enforcement reported a tornado on the ground.',
                                       'Damage survey pending.'])
            if r['mag'] >= 2:
                r['comment'] += ' Significant damage to homes reported.'
    return kept


# ------------------------------------------------------------------------------------------ outputs
def write_csv(reports, path, rated=False):
    blocks = [('torn', 'Time,F_Scale,Location,County,State,Lat,Lon,Comments'),
              ('wind', 'Time,Speed,Location,County,State,Lat,Lon,Comments'),
              ('hail', 'Time,Size,Location,County,State,Lat,Lon,Comments')]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for typ, header in blocks:
            f.write(header + '\n')
            for r in [r for r in reports if r['type'] == typ]:
                if typ == 'torn':
                    mag = f"EF{r['mag']}" if rated and r['mag'] is not None else 'UNK'
                elif typ == 'wind':
                    mag = r['mag'] if r['mag'] is not None else 'UNK'
                else:
                    mag = int(round(r['mag'] * 100))
                cm = r['comment'] + (f" ({r['wfo']})" if r['wfo'] else '')
                w.writerow([r['t'].strftime('%H%M'), mag, r['loc'], r['county'] or '', r['state'],
                            f"{r['lat']:.2f}", f"{r['lon']:.2f}", cm])
    print('saved', path)


def render_map(cfg, reports, path, base):
    ext = cfg['extent']
    tr = geo.transformer(lat_0=cfg.get('proj_lat0', (ext[2] + ext[3]) / 2), lon_0=cfg.get('proj_lon0', (ext[0] + ext[1]) / 2))
    proj = geo.projector(tr)
    clip = geo.extent_box(*ext, pad=1.5)
    fig = plt.figure(figsize=(14, 10.2), facecolor='white')
    ax = fig.add_axes([0.02, 0.1, 0.96, 0.8])
    w0, w1 = (parse_time(x) for x in cfg['window'])
    fig.text(0.02, 0.955, cfg.get('title', 'Storm Reports'), fontsize=20, weight='bold', color='#111')
    fig.text(0.02, 0.925, f"NWS Storm Prediction Center  •  {w0:%H%M} UTC {w0:%b %-d} – "
                          f"{(w1 - timedelta(minutes=1)):%H%M} UTC {(w1 - timedelta(minutes=1)):%b %-d %Y}  •  "
                          f"{cfg.get('subtitle', 'Preliminary, unfiltered by quality control')}", fontsize=11, color='#333')
    fig.text(0.98, 0.955, cfg.get('scenario_label', 'Simulated scenario'), fontsize=9.5, color='#8a1c1c', ha='right',
             style='italic')
    geo.basemap(ax, proj, clip, county_box=geo.extent_box(*ext, pad=0.2))
    # optional outlook overlay (verification)
    ov = cfg.get('outlook_overlay')
    if ov:
        from .outlook import CAT_ORDER, CAT_EDGE, CAT_LABEL
        oc = json.load(open(base / ov))
        land_c = geo.land().intersection(clip)
        cats = [c for c in CAT_ORDER if c in oc['categorical'] and c != 'TSTM']
        for c, g in zip(cats, geo.nest([oc['categorical'][c] for c in cats], land_c)):
            geo.draw(ax, g, proj, facecolor='none', edgecolor=CAT_EDGE[c], lw=2.0, zorder=11, alpha=0.9)
    for name, lat, lon in cfg.get('cities', []):
        x, y = tr.transform(lon, lat)
        ax.text(x + 8000, y + 6000, name, fontsize=8.4, color='#333', zorder=13)
        ax.plot(x, y, 's', ms=2.5, color='#333', zorder=13)
    # dots: hail under wind under tornado, the way SPC stacks them
    for typ, ms in (('hail', 6.5), ('wind', 6.5), ('torn', 8)):
        rs = [r for r in reports if r['type'] == typ]
        if not rs:
            continue
        xs, ys = tr.transform([r['lon'] for r in rs], [r['lat'] for r in rs])
        ax.plot(xs, ys, 'o', ms=ms, color=COL[typ], mec='black', mew=0.5, zorder=14 + (typ == 'torn'), ls='none')
    big_h = [r for r in reports if r['type'] == 'hail' and r['mag'] >= 2.0]
    big_w = [r for r in reports if r['type'] == 'wind' and (r['mag'] or 0) >= 75]
    for rs, mk in ((big_h, '^'), (big_w, 's')):
        if rs:
            xs, ys = tr.transform([r['lon'] for r in rs], [r['lat'] for r in rs])
            ax.plot(xs, ys, mk, ms=5.2, color='black', zorder=16, ls='none')
    geo.set_extent(ax, tr, *ext)
    for s in ax.spines.values():
        s.set_color('#4a4f55'); s.set_linewidth(1.2)
    n = {t: sum(r['type'] == t for r in reports) for t in COL}
    handles = [Line2D([], [], marker='o', ls='none', ms=9, color=COL['torn'], mec='black', label=f"Tornado ({n['torn']})"),
               Line2D([], [], marker='o', ls='none', ms=9, color=COL['wind'], mec='black', label=f"Wind ({n['wind']})"),
               Line2D([], [], marker='o', ls='none', ms=9, color=COL['hail'], mec='black', label=f"Hail ({n['hail']})"),
               Line2D([], [], marker='^', ls='none', ms=8, color='black', label=f'Hail 2"+ ({len(big_h)})'),
               Line2D([], [], marker='s', ls='none', ms=7, color='black', label=f'Wind 65 kt+ ({len(big_w)})')]
    if ov:
        handles.append(Line2D([], [], color='#888', lw=2, label=f"Outline: {oc.get('day_label', 'Day 1')} outlook "
                                                                  f"{oc.get('issued', '').split(' ')[0]}"))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.5, 0.045), columnspacing=1.8)
    if cfg.get('note'):
        fig.text(0.5, 0.025, cfg['note'], ha='center', fontsize=9.5, color='#444')
    fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)
    print('saved', path)


def render(cfg, outdir):
    matplotlib.rcdefaults()
    base = Path(cfg.get('_base_dir', '.'))
    reps = build_reports(cfg, base)
    stem = cfg.get('output', 'storm_reports')
    render_map(cfg, reps, Path(outdir) / f'{stem}.png', base)
    write_csv(reps, Path(outdir) / f'{stem}.csv', rated=cfg.get('rated', False))
    n = {t: sum(r['type'] == t for r in reps) for t in ('torn', 'wind', 'hail')}
    big = max([r['mag'] for r in reps if r['type'] == 'hail'], default=0)
    print(f"  {n['torn']} tornado, {n['wind']} wind, {n['hail']} hail reports; largest hail {big:.2f} in.")


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
