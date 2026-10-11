"""NHC-style text products from a storm JSON: TCM forecast/advisory and TCP public advisory.

usage: python -m wxsim.tropical.products scenarios/<name>/tc_XXXX.json [--outdir DIR] [--no-seas]

Writes TCMATx_<atcf>_advNN.txt and TCPATx_<atcf>_advNN.txt (x = wmo_slot) to <scenario>/output/.
Both are built as one-paragraph-per-line drafts and wrapped with nws_text.wrap_product().
"""
import argparse, json
from datetime import timedelta
from pathlib import Path

from ..nws_text import wrap_product
from . import storm as S

DIR_WORDS = {'N': 'north', 'NNE': 'north-northeast', 'NE': 'northeast', 'ENE': 'east-northeast', 'E': 'east',
             'ESE': 'east-southeast', 'SE': 'southeast', 'SSE': 'south-southeast', 'S': 'south',
             'SSW': 'south-southwest', 'SW': 'southwest', 'WSW': 'west-southwest', 'W': 'west',
             'WNW': 'west-northwest', 'NW': 'northwest', 'NNW': 'north-northwest'}
SUMMARY_ORDER = [('HU.W', 'A Hurricane Warning is in effect for...'),
                 ('HU.A', 'A Hurricane Watch is in effect for...'),
                 ('TR.W', 'A Tropical Storm Warning is in effect for...'),
                 ('TR.A', 'A Tropical Storm Watch is in effect for...')]


def ddhhmm(t):
    return t.strftime('%d%H%M')


def lat_s(lat, w=4):
    return f"{abs(lat):{w}.1f}{'N' if lat >= 0 else 'S'}"


def lon_s(lon, w=5):
    return f"{abs(lon):{w}.1f}{'W' if lon < 0 else 'E'}"


def local_hdr(st, t=None):
    t = t or st.adv_time
    lt = t + timedelta(hours=st.tz['offset_h'])
    h12 = lt.hour % 12 or 12
    return f"{h12}{lt.minute:02d} {'AM' if lt.hour < 12 else 'PM'} {st.tz['name']} {lt.strftime('%a %b %d %Y')}"


def local_short(st, t):
    lt = t + timedelta(hours=st.tz['offset_h'])
    h12 = lt.hour % 12 or 12
    return f"{h12}{lt.minute:02d} {'AM' if lt.hour < 12 else 'PM'} {st.tz['name']}"


def ww_segments(cfg, t):
    return [s for s in cfg.get('coastal_ww', []) if s['type'] == t]


def radii_lines(p, keys=('64', '50', '34'), seas=False, upper=True):
    out = []
    for k in keys:
        q = p.radii.get(k)
        if not q or not any(q.values()):
            continue
        vals = ' '.join(f"{q.get(d, 0):3d}{d}" for d in S.QUADS)
        out.append(f"{k} KT{'.' * 7}{vals}.")
    if seas and p.radii.get('12ft'):
        q = p.radii['12ft']
        out.append('12 FT SEAS..' + ' '.join(f"{q.get(d, 0):3d}{d}" for d in S.QUADS) + '.')
    return out


def fcst_status_word(p):
    if p.status == 'PT':
        return 'POST-TROP/EXTRATROP'
    if p.status == 'LO':
        return 'POST-TROP/REMNT LOW'
    return 'INLAND' if p.inland else ''


# ----------------------------------------------------------------- TCM
def build_tcm(st, cfg, seas=True):
    cur = st.current
    adv = cfg['advisory']
    bdir, bkt = st.motion()
    kind = st.title.split(' ' + st.name)[0].upper()
    L = ['000', f'WTNT2{st.slot} KNHC {ddhhmm(st.adv_time)}', f'TCMAT{st.slot}', '',
         f'{kind} {st.name.upper()} FORECAST/ADVISORY NUMBER{st.adv_num:>4}',
         f'NWS NATIONAL HURRICANE CENTER MIAMI FL       {st.atcf}',
         st.adv_time.strftime('%H%M UTC %a %b %d %Y').upper(), '']
    if cfg.get('changes'):
        L += ['CHANGES IN WATCHES AND WARNINGS WITH THIS ADVISORY...', '']
        L += [c.upper() + '\n' for c in cfg['changes']]
    L += ['SUMMARY OF WATCHES AND WARNINGS IN EFFECT...', '']
    for t, head in SUMMARY_ORDER:
        segs = ww_segments(cfg, t)
        if segs:
            L += [head.upper()] + [f"* {s['from']} TO {s['to']}".upper() for s in segs] + ['']
    L += ['INTERESTS ELSEWHERE ALONG THE GULF COAST AND THE SOUTHEASTERN UNITED STATES SHOULD MONITOR THE '
          'PROGRESS OF DELPHINE.'.replace('DELPHINE', st.name.upper()), '']
    L += [f"{kind} CENTER LOCATED NEAR {lat_s(cur.lat)} {lon_s(cur.lon)} AT {st.adv_time:%d/%H%M}Z",
          f"POSITION ACCURATE WITHIN {adv.get('position_accuracy_nmi', 20):3d} NM", '',
          f"PRESENT MOVEMENT TOWARD THE {DIR_WORDS[S.compass16(bdir)].upper()} OR {int(round(bdir)):3d} DEGREES AT "
          f"{int(round(bkt)):3d} KT", '',
          f"ESTIMATED MINIMUM CENTRAL PRESSURE {cur.mslp:4d} MB"]
    if adv.get('eye_diameter_nmi') and cur.status in ('HU', 'MH'):
        L.append(f"EYE DIAMETER {adv['eye_diameter_nmi']:3d} NM")
    L += [f"MAX SUSTAINED WINDS {cur.vmax:3d} KT WITH GUSTS TO {cur.gust:3d} KT."]
    L += radii_lines(cur, seas=seas)
    L += ['WINDS AND SEAS VARY GREATLY IN EACH QUADRANT.  RADII IN NAUTICAL',
          'MILES ARE THE LARGEST RADII EXPECTED ANYWHERE IN THAT QUADRANT.', '',
          f"REPEAT...CENTER LOCATED NEAR {lat_s(cur.lat)} {lon_s(cur.lon)} AT {st.adv_time:%d/%H%M}Z"]
    prev = [r for r in st.past if r['time'] < st.adv_time]
    if prev:
        r = prev[-1]
        L.append(f"AT {r['time']:%d/%H%M}Z CENTER WAS LOCATED NEAR {lat_s(r['lat'])} {lon_s(r['lon'])}")
    L.append('')
    ext_note = False
    for p in st.points[1:]:
        word = 'FORECAST' if p.tau <= 72 else 'OUTLOOK'
        if p.tau > 72 and not ext_note:
            L += ['EXTENDED OUTLOOK. NOTE...ERRORS FOR TRACK HAVE AVERAGED NEAR 125 NM ON DAY 4 AND 175 NM ON DAY 5...'
                  'AND FOR INTENSITY NEAR 15 KT EACH DAY', '']
            ext_note = True
        if p.status == 'DS':
            L += [f"{word} VALID {p.time:%d/%H%M}Z...{p.note if hasattr(p, 'note') and p.note else 'DISSIPATED'}", '']
            break                                   # NHC lists the first dissipated/absorbed time only
        L.append(f"{word} VALID {p.time:%d/%H%M}Z {lat_s(p.lat)} {lon_s(p.lon)}")
        L.append(f"MAX WIND {p.vmax:3d} KT...GUSTS {p.gust or p.vmax + 10:3d} KT.")
        L += [ln.replace(' KT' + '.' * 7, ' KT' + '.' * 3) for ln in radii_lines(p)]
        w = fcst_status_word(p)
        if w:
            L.append(w)
        L.append('')
    L += [f"REQUEST FOR 3 HOURLY SHIP REPORTS WITHIN 300 MILES OF {lat_s(cur.lat).strip()} {lon_s(cur.lon).strip()}", '',
          f"NEXT ADVISORY AT {S.parse_z(adv['next_complete']):%d/%H%M}Z", '', '$$',
          f"FORECASTER {adv.get('forecaster', 'SIMULATED').upper()}", '']
    return wrap_product('\n'.join(L))


# ----------------------------------------------------------------- TCP
def reference_line(st, lat, lon):
    rows = []
    for name, state, la, lo in st.cfg.get('reference_points', []):
        d, b = S.dist_bearing(la, lo, lat, lon)
        rows.append((d, name, state, b))
    rows.sort()
    d, name, state, b = rows[0]
    mi = int(5 * round(d / 5)); km = int(5 * round(d * 1.609344 / 5))
    return f"ABOUT {mi} MI...{km} KM {S.compass16(b)} OF {name.upper()} {state.upper()}"


def build_tcp(st, cfg):
    cur = st.current
    adv = cfg['advisory']
    txt = cfg.get('text', {})
    bdir, bkt = st.motion()
    lh = local_hdr(st)
    utc = st.adv_time.strftime('%H%M')
    summ = f"SUMMARY OF {local_short(st, st.adv_time)}...{utc} UTC...INFORMATION"
    cat = S.saffir_simpson(cur.vmax)
    wpc = adv.get('center', 'NHC') == 'WPC'
    L = ['000', f"WTNT3{st.slot} {'KWNH' if wpc else 'KNHC'} {ddhhmm(st.adv_time)}", f'TCPAT{st.slot}', '', 'BULLETIN',
         f"{st.title} Advisory Number{st.adv_num:>4}",
         f"NWS {'Weather Prediction Center College Park MD' if wpc else 'National Hurricane Center Miami FL'}"
         f"       {st.atcf}", lh, '']
    L += [f"...{h}..." for h in txt.get('headlines', [])]
    L += ['', '', summ, '-' * len(summ),
          f"LOCATION...{abs(cur.lat):.1f}{'N' if cur.lat >= 0 else 'S'} {abs(cur.lon):.1f}{'W' if cur.lon < 0 else 'E'}",
          reference_line(st, cur.lat, cur.lon),
          f"MAXIMUM SUSTAINED WINDS...{S.kt_to_mph(cur.vmax)} MPH...{S.kt_to_kmh(cur.vmax)} KM/H",
          f"PRESENT MOVEMENT...{S.compass16(bdir)} OR {int(round(bdir))} DEGREES AT {S.motion_mph(bkt)} MPH..."
          f"{S.motion_kmh(bkt)} KM/H",
          f"MINIMUM CENTRAL PRESSURE...{cur.mslp} MB...{S.mb_to_in(cur.mslp):.2f} INCHES", '', '',
          'WATCHES AND WARNINGS', '-' * 20]
    if cfg.get('changes'):
        L += ['CHANGES WITH THIS ADVISORY:', ''] + [c + '\n' for c in cfg['changes']]
    else:
        L += ['CHANGES WITH THIS ADVISORY:', '', 'None.', '']
    if cfg.get('coastal_ww'):
        L += ['SUMMARY OF WATCHES AND WARNINGS IN EFFECT:', '']
    elif not cfg.get('changes'):
        L += ['There are no coastal watches or warnings in effect.', '']
    for t, head in SUMMARY_ORDER:
        segs = ww_segments(cfg, t)
        if segs:
            L += [head] + [f"* {s['from']} to {s['to']}" for s in segs] + ['']
    defs = {'HU.W': 'A Hurricane Warning means that hurricane conditions are expected somewhere within the warning '
                    'area. Preparations to protect life and property should be rushed to completion.',
            'HU.A': 'A Hurricane Watch means that hurricane conditions are possible within the watch area.',
            'TR.W': 'A Tropical Storm Warning means that tropical storm conditions are expected somewhere within the '
                    'warning area.',
            'TR.A': 'A Tropical Storm Watch means that tropical storm conditions are possible within the watch area.'}
    for t, _ in SUMMARY_ORDER:
        if ww_segments(cfg, t):
            L += [defs[t], '']
    if txt.get('watch_extra'):
        L += [txt['watch_extra'], '']

    # discussion
    p12 = next((p for p in st.points if p.tau == 12), None)
    L += ['', 'DISCUSSION AND OUTLOOK', '-' * 22,
          f"At {local_short(st, st.adv_time)} ({utc} UTC), the center of {st.title} was located near latitude "
          f"{abs(cur.lat):.1f} {'North' if cur.lat >= 0 else 'South'}, longitude {abs(cur.lon):.1f} "
          f"{'West' if cur.lon < 0 else 'East'}. {st.name} is moving toward the {DIR_WORDS[S.compass16(bdir)]} near "
          f"{S.motion_mph(bkt)} mph ({S.motion_kmh(bkt)} km/h). {txt.get('track', '')}".strip(), '']
    wtxt = (f"Maximum sustained winds are near {S.kt_to_mph(cur.vmax)} mph ({S.kt_to_kmh(cur.vmax)} km/h) with higher "
            f"gusts.")
    if cat:
        wtxt += (f" {st.name} is a {'major hurricane' if cat >= 3 else 'hurricane'} -- a category {cat} on the "
                 f"Saffir-Simpson Hurricane Wind Scale.")
    if p12 and S.saffir_simpson(p12.vmax) > cat:
        wtxt += f" {st.name} is forecast to become a category {S.saffir_simpson(p12.vmax)} hurricane before landfall."
    if txt.get('intensity_extra'):
        wtxt += ' ' + txt['intensity_extra']
    L += [wtxt, '']
    r64 = max(cur.radii.get('64', {}).values() or [0]); r34 = max(cur.radii.get('34', {}).values() or [0])
    if r34:
        s = ''
        if r64:
            s = (f"Hurricane-force winds extend outward up to {int(5 * round(r64 * 1.15078 / 5))} miles "
                 f"({int(5 * round(r64 * 1.852 / 5))} km) from the center and ")
        s += (f"{'t' if s else 'T'}ropical-storm-force winds extend outward up to "
              f"{int(5 * round(r34 * 1.15078 / 5))} miles ({int(5 * round(r34 * 1.852 / 5))} km).")
        L += [s, '']
    L += [f"The estimated minimum central pressure is {cur.mslp} mb ({S.mb_to_in(cur.mslp):.2f} inches).", '']

    # hazards
    L += ['', 'HAZARDS AFFECTING LAND', '-' * 22]
    hz = txt.get('hazards', [])
    if not hz:
        L += ['None', '']
    for block in hz:
        head, first, *rest = block
        L.append(f"{head}:  {first}")
        L += rest
        L.append('')

    rr = txt.get('rain_reports')
    if rr:
        from ..wpc import rain as R
        base = cfg.get('_base_dir', '.')
        rcfg = json.load(open(Path(base) / rr['rain_config'])); 
        L += [rr.get('intro', 'The following are selected storm total rainfall amounts in inches, including the '
              'predecessor rain event, through the time of this advisory...'), '']
        for state, rows in R.top_places(rcfg, base, rr['states'], rr.get('n_per_state', 4),
                                        rr.get('force', []),
                                        window=(S.parse_z(rcfg['track']['start']), st.adv_time)).items():
            L.append(f"...{rr.get('state_names', {}).get(state, state).upper()}...")
            L += [f"{nm:<30}{v:6.2f}" for nm, v in rows]
            L.append('')
    nxt_i, nxt_c = adv.get('next_intermediate'), adv.get('next_complete')
    L += ['', 'NEXT ADVISORY', '-' * 13]
    if nxt_i:
        L.append(f"Next intermediate advisory at {local_short(st, S.parse_z(nxt_i))}.")
    if nxt_c:
        L.append(f"Next complete advisory at {local_short(st, S.parse_z(nxt_c))}.")
    L += ['', '$$', f"Forecaster {adv.get('forecaster', 'Simulated')}", '']
    return wrap_product('\n'.join(L))


def render(cfg, outdir, seas=True):
    st = S.Storm(cfg)
    outs = []
    kinds = (('TCM', lambda: build_tcm(st, cfg, seas)), ('TCP', lambda: build_tcp(st, cfg)))
    if cfg['advisory'].get('center', 'NHC') == 'WPC':      # WPC issues the public advisory only
        kinds = kinds[1:]
    for kind, fn in kinds:
        out = Path(outdir) / f"{kind}AT{st.slot}_{st.atcf}_adv{st.adv_num}.txt"
        out.write_text(fn())
        print('saved', out)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--outdir'); ap.add_argument('--no-seas', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); cfg['_base_dir'] = str(Path(a.config).parent)
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir, seas=not a.no_seas)


if __name__ == '__main__':
    main()
