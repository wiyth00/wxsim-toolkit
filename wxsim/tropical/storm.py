"""Storm JSON loading, validation and the shared unit/time/geometry helpers for the tropical module.

A storm file (tc_*.json) is one NHC advisory: identity, advisory time, past track, forecast points, wind
radii, coastal (and optional inland) watches/warnings, and the illustrative cone-radii table. See the `_doc`
keys in scenarios/delphine_2027/tc_delphine_adv18.json for the full format.

Everything downstream (cone graphic, TCM, TCP, and later the Phase 2 wind-probability ensemble) reads the
same Storm object, so new products only need a new consumer module.
"""
import json, math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

STATUSES = ('TD', 'TS', 'HU', 'MH', 'PT', 'EX', 'LO', 'DS')    # PT = post-tropical, EX = extratropical, LO = remnant low, DS = dissipated
TROPICAL = ('TD', 'TS', 'HU', 'MH')
STATUS_WORDS = {'TD': 'Tropical Depression', 'TS': 'Tropical Storm', 'HU': 'Hurricane', 'MH': 'Hurricane',
                'PT': 'Post-Tropical Cyclone', 'EX': 'Extratropical Low', 'LO': 'Remnant Low', 'DS': 'Dissipated'}
QUADS = ('NE', 'SE', 'SW', 'NW')
QUAD_BEARINGS = {'NE': (0, 90), 'SE': (90, 180), 'SW': (180, 270), 'NW': (270, 360)}
FCST_TAUS = (0, 12, 24, 36, 48, 60, 72, 96, 120)
# Watch/warning codes -> (label, NHC color). Order = drawing priority (last drawn on top).
WW_TYPES = {'TR.A': ('Tropical Storm Watch', '#ffe600'), 'TR.W': ('Tropical Storm Warning', '#1f4de0'),
            'HU.A': ('Hurricane Watch', '#ff9ecf'), 'HU.W': ('Hurricane Warning', '#e8141c')}
WW_ORDER = ('TR.A', 'HU.A', 'TR.W', 'HU.W')


# ----------------------------------------------------------------- units (NHC rounding conventions)
def kt_to_mph(kt):
    """NHC public-advisory convention: nearest 5 mph."""
    return int(5 * round(kt * 1.15078 / 5))


def motion_mph(kt):
    """Forward speed: nearest whole mph (not the 5-mph wind rounding)."""
    return int(round(kt * 1.15078))


def motion_kmh(kt):
    return int(round(kt * 1.852))


def kt_to_kmh(kt):
    return int(5 * round(kt * 1.852 / 5))


def mb_to_in(mb):
    return mb * 0.0295300


def saffir_simpson(kt):
    for cat, lo in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64)):
        if kt >= lo:
            return cat
    return 0


def status_from_wind(kt):
    return 'MH' if kt >= 96 else 'HU' if kt >= 64 else 'TS' if kt >= 34 else 'TD'


def marker_letter(status, kt):
    """Cone marker letter. Post-tropical/remnant points keep the wind letter but are drawn hollow."""
    return {'TD': 'D', 'TS': 'S', 'HU': 'H', 'MH': 'M'}.get(status, {'TD': 'D', 'TS': 'S', 'HU': 'H',
                                                                     'MH': 'M'}[status_from_wind(kt)])


# ----------------------------------------------------------------- geometry
def dist_bearing(lat1, lon1, lat2, lon2):
    """Great-circle distance (statute miles) and initial bearing (deg) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    d = 2 * 3958.76 * math.asin(math.sqrt(a))
    b = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return d, b % 360


def compass16(bearing):
    pts = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return pts[int((bearing % 360) / 22.5 + 0.5) % 16]


# ----------------------------------------------------------------- time
def parse_z(s):
    """'2027-09-24T21:00Z' -> aware UTC datetime."""
    return datetime.strptime(s.replace('Z', ''), '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc)


def local_label(t, tz):
    """NHC cone label style: '2 PM Sat'."""
    lt = t + timedelta(hours=tz['offset_h'])
    h = lt.hour % 12 or 12
    return f"{h} {'AM' if lt.hour < 12 else 'PM'} {lt.strftime('%a')}"


@dataclass
class Point:
    tau: int
    time: datetime
    lat: float
    lon: float
    vmax: int
    gust: int = None
    status: str = 'TS'
    mslp: int = None
    radii: dict = field(default_factory=dict)     # {'34': {'NE':..}, '50': {...}, '64': {...}} in nmi
    inland: bool = False
    note: str = ''                                 # e.g. 'ABSORBED BY FRONT' for a DS point

    @property
    def tropical(self):
        return self.status in TROPICAL


class Storm:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg['name']
        self.number = int(cfg['number'])
        self.basin = cfg.get('basin', 'AL')
        self.year = int(cfg.get('year', cfg['advisory']['time'][:4]))
        self.atcf = f"{self.basin}{self.number:02d}{self.year}"
        adv = cfg['advisory']
        self.adv_num = str(adv['number'])
        self.adv_time = parse_z(adv['time'])
        # forecast taus count from the synoptic time (adv time rounded down to 00/06/12/18Z), as NHC does
        self.synoptic = self.adv_time.replace(minute=0, hour=self.adv_time.hour - self.adv_time.hour % 6)
        self.tz = cfg.get('tz', {'offset_h': -4, 'name': 'EDT'})
        self.slot = int(cfg.get('wmo_slot', 1))
        self.past = [dict(time=parse_z(r[0]), lat=r[1], lon=r[2], vmax=r[3], mslp=r[4], status=r[5])
                     for r in cfg['past_track']]
        self.points = []
        for f in cfg['forecast']:
            tau = int(f['tau'])
            t = self.adv_time if tau == 0 else self.synoptic + timedelta(hours=tau)
            st = f.get('status') or status_from_wind(f['vmax_kt'])
            self.points.append(Point(tau, t, f.get('lat'), f.get('lon'), f.get('vmax_kt', 0),
                                     f.get('gust_kt'), st, f.get('mslp'), f.get('radii', {}), f.get('inland', False),
                                     f.get('note', '')))
        self.cone_radii = {int(k): v for k, v in cfg['cone']['radii_nmi'].items()}
        self.validate()

    # --------------------------------------------------------- convenience
    @property
    def current(self):
        return self.points[0]

    @property
    def live_points(self):
        """Forecast points that still exist (not dissipated) -> used for the cone and markers."""
        return [p for p in self.points if p.status != 'DS' and p.lat is not None]

    @property
    def title(self):
        p = self.current
        if p.status in ('HU', 'MH'):
            return f'Hurricane {self.name}'
        return f'{STATUS_WORDS[p.status]} {self.name}'

    def motion(self):
        """(bearing deg, kt): explicit advisory.motion, else from the last past fix to the current position."""
        m = self.cfg['advisory'].get('motion')
        if m:
            return m['dir_deg'], m['kt']
        last = self.past[-1] if self.past and self.past[-1]['time'] < self.adv_time else self.past[-2]
        d, b = dist_bearing(last['lat'], last['lon'], self.current.lat, self.current.lon)
        hrs = (self.adv_time - last['time']).total_seconds() / 3600
        return b, d / 1.15078 / hrs

    def validate(self):
        errs = []
        for p in self.points:
            if p.status not in STATUSES:
                errs.append(f'tau {p.tau}: unknown status {p.status}')
            if p.status in TROPICAL and p.status != status_from_wind(p.vmax):
                errs.append(f'tau {p.tau}: status {p.status} does not match {p.vmax} kt')
            for k, q in p.radii.items():
                if k not in ('34', '50', '64', '12ft'):
                    errs.append(f'tau {p.tau}: radii key {k}')
                if k != '12ft' and int(k) > p.vmax and any(q.values()):
                    errs.append(f'tau {p.tau}: {k}-kt radii but vmax {p.vmax} kt')
            if p.radii.get('64') and p.radii.get('34'):
                for qd in QUADS:
                    if p.radii['64'].get(qd, 0) > p.radii['34'].get(qd, 0):
                        errs.append(f'tau {p.tau}: 64-kt radius > 34-kt radius in {qd}')
        for p in self.live_points:
            if p.tau not in self.cone_radii:
                errs.append(f'cone radii table has no entry for tau {p.tau}')
        if errs:
            raise ValueError('storm JSON problems:\n  ' + '\n  '.join(errs))


def load(path):
    cfg = json.load(open(path))
    return Storm(cfg), cfg
