"""Boundaries that evolve on their own, and storm-generated cold pools / outflow boundaries.

DynamicBoundary
    A line of nodes (km coordinates) that move normal to the line each step. Positive normal speed = toward the
    RIGHT side of the line as drawn (the side opposite its `left` air mass). Rules by kind:
      dryline      synoptic push + daytime mixing "jump" (solar-driven, lagged) that is locally enhanced under a
                   mid-level speed max (dryline bulge) and suppressed under storm anvils; nocturnal retreat
                   once mixing stops.
      wedge_front  (cold-air-damming edge, wedge on the left) erodes toward the wedge by daytime mixing and
                   evening warm advection; rain-cooled outflow from storms nearby reinforces it (it stalls or
                   pushes back).
      cold_front   moves toward the warm (right) side at a prescribed, time-varying speed.
      warm_front   retreats toward the cold (left) side at a prescribed, time-varying speed.
    The line is lightly smoothed and resampled every step so it stays well-behaved.

ColdPools
    Each precipitating surface-based storm feeds a cold pool just behind it (so pools trail storms as swaths).
    Pools spread outward (gust-front speed ~ sqrt(strength)), drift with the low-level flow, merge when they
    touch, and weaken once no longer fed. Inside: capped/stable (storms there go elevated or die). Along the edge:
    convergent lift for new cells, backed winds (SRH boost) and lower LCLs.
"""
import math
from datetime import timedelta

import numpy as np
import shapely
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

from ..radar.storms import parse_time

KT2KMH = 1.852


def solar_sin_elev(t, lat, lon):
    doy = t.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(math.radians(360 / 365 * (284 + doy)))
    hours = t.hour + t.minute / 60 + np.asarray(lon) / 15.0
    H = np.radians(15 * (hours - 12))
    phi = np.radians(lat)
    return np.sin(phi) * math.sin(decl) + np.cos(phi) * math.cos(decl) * np.cos(H)


def mixing(t, lat, lon, lag_h=2.0):
    """0..1 boundary-layer mixing strength: solar heating lagged ~2 h (the PBL keeps deepening into the afternoon)."""
    return np.clip(solar_sin_elev(t - timedelta(hours=lag_h), lat, lon) / 0.8, 0, 1)


def _pw(tv, t):
    if tv is None:
        return None
    if isinstance(tv, (int, float)):
        return float(tv)
    ks = sorted(tv, key=parse_time)
    return float(np.interp(t.timestamp(), [parse_time(k).timestamp() for k in ks], [tv[k] for k in ks]))


def _resample_xy(P, spacing):
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(8, int(s[-1] / spacing) + 1)
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]


class DynamicBoundary:
    def __init__(self, b, env):
        self.b, self.env = b, env
        d = b['dynamic']
        self.kind = b.get('kind', 'boundary')
        self.rules = d
        x, y = env.km(*np.asarray(d['initial_line'], float).T)
        self.P = _resample_xy(np.c_[x, y], d.get('spacing_km', 12))
        b['track'] = [{'time': d['initial_time'], 'line': self._lonlat()}]

    def _lonlat(self):
        lo, la = self.env.lonlat(self.P[:, 0], self.P[:, 1])
        return [[round(float(a), 4), round(float(c), 4)] for a, c in zip(lo, la)]

    def normals(self):
        P = self.P
        d = np.gradient(P, axis=0)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1])[:, None], 1e-9)
        return np.c_[d[:, 1], -d[:, 0]]            # right-hand normal

    def speed_kt(self, t, lon, lat, storms_xy):
        r = self.rules
        k = self.kind
        if k == 'dryline':
            mix = mixing(t, lat, lon)
            bulge = np.zeros_like(lon)
            for j in r.get('speed_max', []):
                jl, jo = _track_point(j['track'], t)
                jx, jy = self.env.km(jo, jl)
                x, y = self.env.km(lon, lat)
                bulge += j.get('amp', 0.7) * np.exp(-((x - jx) ** 2 + (y - jy) ** 2) / j.get('radius_km', 80) ** 2)
            shade = 1 - r.get('anvil_shading', 0.45) * _near(storms_xy, self.env.km(lon, lat), 45)
            return (_pw(r.get('synoptic_kt', 5), t) + r.get('jump_kt', 16) * mix * (1 + bulge) * shade
                    - r.get('retreat_kt', 8) * (1 - mix))
        if k == 'wedge_front':
            mix = mixing(t, lat, lon) * r.get('cloud_factor', 0.6)
            erode = r.get('erode_day_kt', 5) * mix + (_pw(r.get('erode_night_kt', 2), t) or 0) * (1 - mix)
            reinforce = r.get('reinforce_kt', 7) * _near(storms_xy, self.env.km(lon, lat), 40)
            return -erode + reinforce + (_pw(r.get('push_kt', 0), t) or 0)
        if k == 'cold_front':
            # QLCS cold pools surge the front ahead where convection is strongest (bows / LEWPs)
            surge = r.get('surge_kt', 0) * _near(storms_xy, self.env.km(lon, lat), r.get('surge_radius_km', 50))
            # optional latitude profile of the front's speed, e.g. a trailing southern end that lags behind
            lat_f = np.ones_like(lon)
            if r.get('speed_by_lat'):
                la_, f_ = np.array(sorted(r['speed_by_lat'])).T
                lat_f = np.interp(lat, la_, f_)
            return _pw(r.get('speed_kt', 15), t) * lat_f + surge
        if k == 'warm_front':
            return -_pw(r.get('speed_kt', 10), t) * np.ones_like(lon)
        return _pw(r.get('speed_kt', 0), t) * np.ones_like(lon)

    def step(self, t, dt_min, storms_xy):
        lon, lat = self.env.lonlat(self.P[:, 0], self.P[:, 1])
        v = self.speed_kt(t, lon, lat, storms_xy) * KT2KMH * dt_min / 60      # km this step
        self.P = self.P + self.normals() * v[:, None]
        # smooth (keep endpoints) and resample
        # Taubin smoothing (shrink-free): a lambda pass followed by a negative-mu inflation pass
        for _ in range(self.rules.get('smooth_passes', 1)):
            for f in (0.33, -0.34):
                lap = 0.5 * (self.P[:-2] + self.P[2:]) - self.P[1:-1]
                self.P[1:-1] = self.P[1:-1] + f * lap
        self.P = _resample_xy(self.P, self.rules.get('spacing_km', 12))
        self.b['track'].append({'time': (t + timedelta(minutes=dt_min)).strftime('%Y-%m-%dT%H:%MZ'),
                                'line': self._lonlat()})


def _track_point(track, t):
    ts = [parse_time(p['time']).timestamp() for p in track]
    return (float(np.interp(t.timestamp(), ts, [p['lat'] for p in track])),
            float(np.interp(t.timestamp(), ts, [p['lon'] for p in track])))


def _near(storms_xy, xy, radius_km):
    """0..1 'how much precipitating storm activity is within radius' for each node."""
    x, y = xy
    out = np.zeros(np.shape(x))
    for sx, sy, w in storms_xy:
        out = np.maximum(out, w * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / radius_km ** 2))
    return out


# ------------------------------------------------------------------------------------------------ cold pools
class Pool:
    _n = 0

    def __init__(self, geom, strength, parent, t):
        Pool._n += 1
        self.id = f'P{Pool._n:02d}'
        self.geom, self.I, self.parents, self.born, self.fed = geom, strength, {parent}, t, t


class ColdPools:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg or {}
        self.pools = []
        self.history = {}           # iso -> list of (id, I, geom_km, parents)

    # -- feeding / evolution
    def feed(self, storm, t, E):
        c = self.cfg
        if storm.elev > 0.5 or storm.dz < -8 or storm.age < 25:
            return
        prec01 = (storm.prec + 1) / 2
        C = (0.35 + 0.4 * prec01) * np.clip(1.3 - E['rh_low'] / 100, 0.3, 1.2) * (1 + 0.6 * storm.hb) * (1 - 0.35 * storm.sup)
        C = float(np.clip(C * c.get('strength', 1.0), 0, 1))
        if C < 0.08:
            return
        hd = math.atan2(storm.u, storm.v) if storm.u is not None else 0.0
        x, y = self.env.km(storm.lon, storm.lat)
        back = c.get('source_back_km', 5)
        disc = Point(x - back * math.sin(hd), y - back * math.cos(hd)).buffer(c.get('source_radius_km', 7) * storm.scale,
                                                                             resolution=8)
        mine = [p for p in self.pools if storm.id in p.parents]
        if mine:
            p = mine[0]
            p.geom = p.geom.union(disc)
            p.I = max(p.I, C); p.fed = t
        else:
            self.pools.append(Pool(disc, C, storm.id, t))

    def step(self, t, dt_min):
        c = self.cfg
        for p in self.pools:
            cx, cy = p.geom.centroid.x, p.geom.centroid.y
            lo, la = self.env.lonlat(cx, cy)
            F = self.env.sample(np.array([lo]), np.array([la]), t.strftime('%Y-%m-%dT%H:%MZ'), use_pools=False)
            drift = c.get('drift_frac', 0.35) * dt_min / 60 * KT2KMH
            from shapely import affinity
            p.geom = affinity.translate(p.geom, float(F['mean_u'][0]) * drift, float(F['mean_v'][0]) * drift)
            # gravity-current speed ~ sqrt(depth * dtheta): full speed while fed, slowing as the unfed pool thins
            unfed_min = (t - p.fed).total_seconds() / 60
            taper = math.exp(-max(0.0, unfed_min - dt_min) / c.get('spread_taper_min', 40))
            spread = c.get('spread_kt', 14) * math.sqrt(max(p.I, 0)) * taper * KT2KMH * dt_min / 60
            if spread > 0.05:
                p.geom = p.geom.buffer(spread, resolution=6).simplify(1.0)
            if unfed_min > dt_min:
                p.I *= math.exp(-dt_min / c.get('decay_min', 100))
            # a big, old pool loses its sharp edge: cap area growth by eroding its fringe once it stops being fed
            # (a merged pool fed by many storms is capped too once it passes hard_area_factor x max_area)
            amax = c.get('max_area_km2', 25000)
            if (unfed_min > 90 and p.geom.area > amax) or p.geom.area > c.get('hard_area_factor', 2.0) * amax:
                p.geom = p.geom.buffer(-2.0, resolution=6).simplify(1.0)
        self.pools = [p for p in self.pools if p.I > c.get('min_strength', 0.08) and not p.geom.is_empty]
        # merge touching pools
        merged = True
        while merged:
            merged = False
            for i in range(len(self.pools)):
                for j in range(i + 1, len(self.pools)):
                    a, b = self.pools[i], self.pools[j]
                    if a.geom.intersects(b.geom):
                        a.geom = unary_union([a.geom, b.geom]).simplify(1.0)
                        a.I = max(a.I, b.I); a.parents |= b.parents; a.fed = max(a.fed, b.fed)
                        del self.pools[j]
                        merged = True
                        break
                if merged:
                    break
        self.history[t.strftime('%Y-%m-%dT%H:%MZ')] = [(p.id, p.I, p.geom, set(p.parents)) for p in self.pools]

    # -- queries
    def snapshot(self, when):
        if when in self.history:
            return self.history[when]
        keys = sorted(self.history, key=parse_time)
        t = parse_time(when)
        prior = [k for k in keys if parse_time(k) <= t]
        return self.history[prior[-1]] if prior else []

    def effects(self, x, y, when, exclude=None):
        """Multiplicative/additive modifiers for environment fields at km points (x, y)."""
        shape = np.shape(x)
        w_in = np.zeros(shape); w_edge = np.zeros(shape)
        pts = shapely.points(np.ravel(x), np.ravel(y))
        for pid, I, geom, parents in self.snapshot(when):
            if exclude and exclude in parents:
                continue
            inside = shapely.contains_xy(geom, np.ravel(x), np.ravel(y)).reshape(shape)
            d = shapely.distance(pts, geom.boundary).reshape(shape)
            w_in = np.maximum(w_in, np.where(inside, I / (1 + np.exp(-(d - 4) / 2.5)), 0))
            w_edge = np.maximum(w_edge, np.where(inside, I * np.exp(-(d / 4) ** 2), I * np.exp(-(d / 10) ** 2)))
        return w_in, w_edge

    def export(self, every_min=10):
        out = []
        for k in sorted(self.history, key=parse_time):
            polys = []
            for pid, I, geom, parents in self.history[k]:
                gs = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for g in gs:
                    lo, la = self.env.lonlat(*np.asarray(g.exterior.coords).T)
                    polys.append({'id': pid, 'strength': round(float(I), 3), 'parents': sorted(parents),
                                  'ring': [[round(float(a), 4), round(float(b), 4)] for a, b in zip(lo, la)]})
            out.append({'time': k, 'pools': polys})
        return out


def lines_at(boundaries_blob, when):
    """For radar fine lines / maps: [(kind, lons, lats, strength)] at ISO time from an exported storms file."""
    out = []
    t = parse_time(when)
    for b in boundaries_blob.get('lines', []):
        frames = b['track']
        prior = [f for f in frames if parse_time(f['time']) <= t]
        f = prior[-1] if prior else frames[0]
        L = np.asarray(f['line'])
        out.append((b.get('kind', 'boundary'), L[:, 0], L[:, 1], b.get('fine_line', 0.5)))
    snaps = [s for s in boundaries_blob.get('pools', []) if parse_time(s['time']) <= t]
    if snaps:
        for p in snaps[-1]['pools']:
            R = np.asarray(p['ring'])
            out.append(('outflow', R[:, 0], R[:, 1], p['strength']))
    return out
