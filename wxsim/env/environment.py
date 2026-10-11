"""Gridded storm environment built from air masses, moving boundaries, regions, and diurnal factors.

The environment answers one question: "what are the convective parameters at (lon, lat, time)?"
It is assembled the way a forecaster reads a mesoanalysis:
  * a base air mass (e.g. the warm sector),
  * boundaries that move with time (drylines, fronts, wedge edges, outflows). Each one puts a different
    air mass on its LEFT side (left = to the left when walking along the line in the direction drawn),
    optionally with a pooled-moisture band on its right side, and provides LIFT for initiation,
  * polygon regions that override the air mass (e.g. a capped zone),
  * time factors (diurnal CIN erosion, evening LLJ SRH ramp, lowering LCLs, CAPE decay after sunset),
  * small mesoscale noise so the fields are not perfectly smooth.
Vectors (mean wind, 0-6 km shear) are blended as u/v components.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ..radar.storms import parse_time

SCALARS = ['mlcape', 'mucape', 'mlcin', 'lcl', 'pw', 'srh01', 'srh03', 'sr_anvil', 'lapse75', 'rh_low',
           'hodo_curve', 'elevated', 'wbz_kft', 'linear', 'heating_ci']
DEFAULTS = dict(mucape=None, rh_low=50, hodo_curve=0.5, elevated=0.0, wbz_kft=9.0, linear=0.2, heating_ci=0.0,
                sr_anvil=50, lapse75=7.0, srh03=None)
KT2MS = 0.514444


def wind_uv(dir_from, spd):
    d = np.deg2rad(dir_from)
    return -spd * np.sin(d), -spd * np.cos(d)


def uv_dir(u, v):
    return (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360


def _piecewise(times_vals, when):
    """times_vals: {ISO: value}; linear interpolation in time, clamped at the ends."""
    if not times_vals:
        return 1.0
    ks = sorted(times_vals, key=lambda k: parse_time(k))
    ts = np.array([parse_time(k).timestamp() for k in ks])
    vs = np.array([times_vals[k] for k in ks], float)
    return float(np.interp(parse_time(when).timestamp(), ts, vs))


def _resample(line, n=60):
    p = np.asarray(line, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


class Environment:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.lon0, self.lon1, self.lat0, self.lat1 = cfg['domain']
        self.lonc, self.latc = (self.lon0 + self.lon1) / 2, (self.lat0 + self.lat1) / 2
        self.kx = 111.32 * np.cos(np.deg2rad(self.latc)); self.ky = 110.57
        self.air = {k: self._airmass(v) for k, v in cfg['airmasses'].items()}
        self.base = cfg.get('base_airmass', 'warm_sector')
        # mesoscale noise (fixed in space), sampled by bilinear interpolation
        rng = np.random.default_rng(seed + 991)
        self.nx, self.ny = 400, 300
        def fld(s):
            n = gaussian_filter(rng.standard_normal((self.ny, self.nx)), s)
            return n / n.std()
        self.noise = {'cape': fld(10), 'srh': fld(12), 'lcl': fld(14)}
        self.noise_amp = cfg.get('noise', {'cape': 0.10, 'srh': 0.12, 'lcl': 0.06})
        # optional marine layer: over open water, blend toward a (usually stable) marine air mass
        self.marine = cfg.get('marine')
        if self.marine:
            self.water = self._water_grid(self.marine.get('blend_km', 20))

    # ------------------------------------------------------------------ helpers
    def _water_grid(self, blend_km):
        """0 over land .. 1 over water, on the noise grid, smoothed over ~blend_km (coastal transition)."""
        import shapely
        from .. import geo
        lons = np.linspace(self.lon0, self.lon1, self.nx); lats = np.linspace(self.lat0, self.lat1, self.ny)
        LO, LA = np.meshgrid(lons, lats)
        on_land = shapely.contains_xy(geo.land(), LO.ravel(), LA.ravel()).reshape(LO.shape)
        dx = (self.lon1 - self.lon0) / (self.nx - 1) * self.kx
        g = gaussian_filter((~on_land).astype(float), max(blend_km / dx / 2, 0.5))
        return np.clip((g - 0.25) / 0.5, 0, 1)

    def _water_at(self, lon, lat):
        fx = (np.asarray(lon) - self.lon0) / (self.lon1 - self.lon0) * (self.nx - 1)
        fy = (np.asarray(lat) - self.lat0) / (self.lat1 - self.lat0) * (self.ny - 1)
        return map_coordinates(self.water, [np.ravel(fy), np.ravel(fx)], order=1, mode='nearest').reshape(np.shape(lon))

    def _airmass(self, a):
        d = dict(DEFAULTS); d.update(a)
        if d['mucape'] is None:
            d['mucape'] = d['mlcape'] * 1.15
        if d['srh03'] is None:
            d['srh03'] = d['srh01'] * 1.6
        d['mean_u'], d['mean_v'] = wind_uv(*a['mean_wind'])
        d['shr_u'], d['shr_v'] = wind_uv(*a['shear06'])
        return d

    def km(self, lon, lat):
        return (np.asarray(lon) - self.lonc) * self.kx, (np.asarray(lat) - self.latc) * self.ky

    def lonlat(self, x, y):
        return self.lonc + np.asarray(x) / self.kx, self.latc + np.asarray(y) / self.ky

    def boundary_line(self, b, when):
        """Boundary polyline (lon/lat, resampled) at `when`, interpolated between its keyframes."""
        key = (id(b), when, len(b['track']))
        cache = self.__dict__.setdefault('_bl_cache', {})
        if key not in cache:
            if len(cache) > 5000:
                cache.clear()
            cache[key] = self._boundary_line(b, when)
        return cache[key]

    def _boundary_line(self, b, when):
        frames = sorted(b['track'], key=lambda f: parse_time(f['time']))
        t = parse_time(when)
        if t <= parse_time(frames[0]['time']):
            return _resample(frames[0]['line'])
        if t >= parse_time(frames[-1]['time']):
            return _resample(frames[-1]['line'])
        for f0, f1 in zip(frames[:-1], frames[1:]):
            t0, t1 = parse_time(f0['time']), parse_time(f1['time'])
            if t0 <= t <= t1:
                w = (t - t0) / (t1 - t0)
                return (1 - w) * _resample(f0['line']) + w * _resample(f1['line'])

    def signed_distance(self, b, when, x, y):
        """km from the boundary; positive on the LEFT side of the line as drawn."""
        L = self.boundary_line(b, when)
        lx, ly = self.km(L[:, 0], L[:, 1])
        ax, ay, bx, by = lx[:-1], ly[:-1], lx[1:], ly[1:]
        dx, dy = bx - ax, by - ay
        P = np.c_[np.ravel(x), np.ravel(y)]
        px, py = P[:, :1], P[:, 1:]
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx ** 2 + dy ** 2), 0, 1)
        cx, cy = ax + t * dx, ay + t * dy
        dist = np.hypot(px - cx, py - cy)
        k = np.argmin(dist, axis=1)
        dmin = dist[np.arange(len(k)), k]
        cross = dx[k] * (P[:, 1] - ay[k]) - dy[k] * (P[:, 0] - ax[k])
        return (np.sign(cross) * dmin).reshape(np.shape(x))

    def _noise_at(self, key, lon, lat):
        fx = (np.asarray(lon) - self.lon0) / (self.lon1 - self.lon0) * (self.nx - 1)
        fy = (np.asarray(lat) - self.lat0) / (self.lat1 - self.lat0) * (self.ny - 1)
        return map_coordinates(self.noise[key], [np.ravel(fy), np.ravel(fx)], order=1, mode='nearest').reshape(np.shape(lon))

    # ------------------------------------------------------------------ main entry
    def sample(self, lon, lat, when, exclude=None, use_pools=True):
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        x, y = self.km(lon, lat)
        keys = SCALARS + ['mean_u', 'mean_v', 'shr_u', 'shr_v']
        F = {k: np.full(lon.shape, float(self.air[self.base][k])) for k in keys}

        def blend(name, w):
            A = self.air[name]
            for k in keys:
                F[k] = F[k] * (1 - w) + A[k] * w

        for r in self.cfg.get('regions', []):
            from shapely.geometry import Polygon
            import shapely
            poly = Polygon(r['polygon'])
            inside = shapely.contains_xy(poly, lon.ravel(), lat.ravel()).reshape(lon.shape)
            d = shapely.distance(shapely.points(lon.ravel(), lat.ravel()), poly.exterior).reshape(lon.shape) * self.ky
            sd = np.where(inside, d, -d)
            blend(r['airmass'], 1 / (1 + np.exp(-sd / (r.get('blend_km', 40) / 2))))

        if self.marine:
            blend(self.marine['airmass'], self.marine.get('weight', 1.0) * self._water_at(lon, lat))

        lift = np.zeros(lon.shape)
        bnear = np.zeros(lon.shape)          # proximity to any lifting boundary (for OFB collisions)
        for b in self.cfg.get('boundaries', []):
            sd = self.signed_distance(b, when, x, y)
            if b.get('pool'):
                pw_ = np.exp(-((np.clip(-sd, 0, None) - b['pool'].get('offset_km', 15)) / b['pool']['width_km']) ** 2)
                blend(b['pool']['airmass'], pw_ * (sd < 0))
            if b.get('left'):
                blend(b['left'], 1 / (1 + np.exp(-sd / (b.get('width_km', 20) / 2))))
            if b.get('lift'):
                lf = b['lift'] * _piecewise(b.get('lift_times'), when)
                lift = np.maximum(lift, lf * np.exp(-((sd - b.get('lift_offset_km', 0)) / b.get('lift_width_km', 12)) ** 2))
                bnear = np.maximum(bnear, np.exp(-(sd / 22) ** 2))
        for m in self.cfg.get('mesolows', []):
            if parse_time(m['start']) <= parse_time(when) <= parse_time(m['end']):
                mx, my = self.km(m['lon'], m['lat'])
                lift = np.maximum(lift, m['lift'] * np.exp(-((x - mx) ** 2 + (y - my) ** 2) / m['radius_km'] ** 2))
        F['lift'] = lift

        tf = self.cfg.get('time_factors', {})
        for k, tv in tf.items():
            F[k] = F[k] * _piecewise(tv, when)
        amp = self.noise_amp
        F['mlcape'] *= 1 + amp['cape'] * self._noise_at('cape', lon, lat)
        F['mucape'] *= 1 + amp['cape'] * self._noise_at('cape', lon, lat)
        F['srh01'] *= 1 + amp['srh'] * self._noise_at('srh', lon, lat)
        F['srh03'] *= 1 + amp['srh'] * self._noise_at('srh', lon, lat)
        F['lcl'] *= 1 + amp['lcl'] * self._noise_at('lcl', lon, lat)
        pools = getattr(self, 'pools', None)
        if use_pools and pools is not None:
            w_in, w_edge = pools.effects(x, y, when, exclude)
            pc = pools.cfg
            F['mlcape'] *= (1 - 0.85 * w_in) * (1 + 0.10 * w_edge)
            F['mlcin'] = F['mlcin'] - pc.get('interior_cin', 180) * w_in
            F['elevated'] = np.maximum(F['elevated'], 0.95 * w_in)
            F['lcl'] *= (1 - 0.40 * w_in) * (1 - 0.15 * w_edge)
            F['srh01'] += pc.get('edge_srh01', 90) * w_edge
            F['srh03'] += pc.get('edge_srh01', 90) * 0.66 * w_edge
            F['lift'] = np.maximum(F['lift'], pc.get('edge_lift', 0.7) * w_edge * (1 - w_in))
            # OFB colliding with a synoptic boundary: lift adds rather than maxes -> intersection hot spots
            F['lift'] = F['lift'] + pc.get('collision_lift', 0.8) * w_edge * (1 - w_in) * bnear
            F['pool_in'], F['pool_edge'] = w_in, w_edge
        return self.composites(F)

    @staticmethod
    def composites(F):
        shr = np.hypot(F['shr_u'], F['shr_v'])                 # kt
        F['shear_kt'] = shr
        s_ms = shr * KT2MS
        # Bunkers right/left movers: mean wind +- 7.5 m/s (14.6 kt) normal to the shear vector
        nx, ny = F['shr_v'] / np.maximum(shr, 1e-3), -F['shr_u'] / np.maximum(shr, 1e-3)
        F['rm_u'], F['rm_v'] = F['mean_u'] + 14.6 * nx, F['mean_v'] + 14.6 * ny
        F['lm_u'], F['lm_v'] = F['mean_u'] - 14.6 * nx, F['mean_v'] - 14.6 * ny
        shr_t = np.where(s_ms < 12.5, 0, np.clip(s_ms, 0, 30) / 20)
        lcl_t = np.clip((2000 - F['lcl']) / 1000, 0, 1)
        cin_t = np.clip((200 + F['mlcin']) / 150, 0, 1)
        sb = 1 - np.clip(F['elevated'], 0, 1)
        F['stp'] = sb * (F['mlcape'] / 1500) * lcl_t * (F['srh01'] / 150) * shr_t * cin_t
        F['scp'] = (F['mucape'] / 1000) * (F['srh03'] / 50) * np.where(s_ms < 10, 0, np.clip(s_ms / 20, 0, 1))
        hail_melt = np.clip((13 - F['wbz_kft']) / 4, 0.4, 1.2)
        F['ship'] = 1.6 * (F['mucape'] / 2000) * np.clip((F['lapse75'] - 5.5) / 2.8, 0, 1.6) * \
            (np.clip(shr, 14, 50) / 40) * hail_melt
        return F

    def boundaries_at(self, when):
        return [(b['name'], b.get('kind', 'boundary'), self.boundary_line(b, when)) for b in self.cfg.get('boundaries', [])]
