"""Build simulated reflectivity for one radar site at one time.

Each storm's primitives are evaluated in its own storm-relative frame at every gate the beam samples,
weighted by whether the beam (at that range and tilt) intersects the primitive's vertical extent, and
multiplied by storm-attached fractal texture (so texture moves with the storm). Contributions are summed
in linear Z, notches removed in dB, then smoothed across the beam (azimuth) and range.
"""
import zlib

import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from . import primitives as prim
from . import storms as st
from .sampling import PolarGrid, beam_height, beam_half_depth, ground_clutter, VCP212

TEX_N, TEX_HALF = 512, 110.0     # storm texture: 512 px spanning +-110 km


def _texture(seed):
    rng = np.random.default_rng(seed)
    def oct_(s, w):
        n = gaussian_filter(rng.standard_normal((TEX_N, TEX_N)), s, mode='wrap')
        return w * n / n.std()
    detail = oct_(0.8, 0.35) + oct_(2.0, 0.45) + oct_(5.0, 0.6) + oct_(12.0, 0.5)
    wu, wv = oct_(6.0, 1.0), oct_(6.0, 1.0)
    return detail / detail.std(), wu, wv


def _sample(tex, u, v):
    px = (u + TEX_HALF) / (2 * TEX_HALF) * (TEX_N - 1)
    py = (v + TEX_HALF) / (2 * TEX_HALF) * (TEX_N - 1)
    return map_coordinates(tex, [py, px], order=1, mode='wrap')


def _vweight(h, half, z):
    zb, zt = z
    lo = 1.0 if zb <= 0 else 1 / (1 + np.exp(-(h + half * 0.6 - zb) / 0.35))
    hi = 1 / (1 + np.exp(-(zt - (h - half * 0.6)) / 0.6))
    return lo * hi


class RadarSim:
    def __init__(self, site, storms, seed=17, texture_db=4.0, warp_km=0.9, lines_fn=None):
        self.site = site
        self.lines_fn = lines_fn      # when -> [(kind, lons, lats, strength)] for boundary fine lines
        self.grid = PolarGrid(site.get('max_range_km', 230.0))
        self.to_xy = Transformer.from_crs('EPSG:4326', f"+proj=aeqd +lat_0={site['lat']} +lon_0={site['lon']} +units=km",
                                          always_xy=True)
        self.to_ll = Transformer.from_crs(f"+proj=aeqd +lat_0={site['lat']} +lon_0={site['lon']} +units=km",
                                          'EPSG:4326', always_xy=True)
        self.storms = storms
        self.seed = seed
        self.texture_db, self.warp_km = texture_db, warp_km
        self._tex = {}

    def tex(self, sid):
        if sid not in self._tex:
            self._tex[sid] = _texture(zlib.crc32(f'{self.seed}:{sid}'.encode()))
        return self._tex[sid]

    @staticmethod
    def _peak_dbz(prims):
        """Return the strongest additive peak advertised by a primitive list."""
        peak = -np.inf
        for P in prims:
            if P.get('kind') in prim.SUBTRACTIVE:
                continue
            peak = max(peak, P.get('peak', -np.inf), P.get('p0', -np.inf), P.get('p1', -np.inf))
        return peak

    def _hail_spike(self, x, y, h, sx, sy, core_peak):
        """Weak three-body scatter return extending radially behind a hail core.

        This is deliberately evaluated in radar coordinates rather than as a
        storm primitive: the spike points away from this radar site and is
        therefore absent from the site-independent MRMS mosaic.  Smooth radial
        and cross-radial tapers avoid a synthetic pixel-sharp streak.
        """
        sr = float(np.hypot(sx, sy))
        if core_peak < 64.5 or sr < 8.0 or sr >= self.grid.R.max() - 4.0:
            return None
        qx, qy = sx / sr, sy / sr
        along = x * qx + y * qy - sr
        cross = np.abs(x * qy - y * qx)
        length = min(42.0, max(16.0, 0.20 * sr + 13.0))
        # Logistic end tapers keep the return continuous at its start/end.
        start = 1.0 / (1.0 + np.exp(-(along - 2.5) / 0.8))
        stop = 1.0 / (1.0 + np.exp((along - length) / 2.0))
        width = 0.55 + 0.012 * np.clip(along, 0, length)
        lateral = np.exp(-0.5 * (cross / width) ** 2)
        vertical = np.exp(-np.clip(h - 0.6, 0, None) / 3.8)
        peak = min(34.0, 20.0 + 0.38 * (core_peak - 64.5))
        db = peak - 0.20 * np.clip(along, 0, None) + 3.0 * lateral
        return 10.0 ** ((db - 8.0 * (1.0 - vertical)) / 10.0) * start * stop * lateral

    def tilt(self, elev, when):
        g = self.grid
        h_all = beam_height(g.R, elev)
        half_all = beam_half_depth(g.R)
        lin = np.zeros(g.R.shape)
        for s in self.storms:
            if s.get('archetype') == 'dynamic':
                # Dynamic storms return dBZ directly instead of a primitive
                # list.  Keep lifecycle, track, positioning, texture, and
                # linear-Z accumulation consistent with the legacy path.
                from . import dynamic
                if not st.active(s, when):
                    continue
                age_min = st.age_minutes(s, when)
                params = st.dynamic_params(s, when)
                lat, lon = st.position(s, when)
                sx, sy = self.to_xy.transform(lon, lat)
                dist = np.hypot(g.X - sx, g.Y - sy)
                try:
                    reach = float(dynamic.reach_km(age_min, params))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not np.isfinite(reach) or reach <= 0:
                    continue
                site_range = float(np.hypot(sx, sy))
                try:
                    hail_hint = float(params.get('hail', 58.0)) + float(params.get('dz', 0.0)) >= 64.5
                except (TypeError, ValueError):
                    hail_hint = False
                spike_possible = hail_hint and 8.0 <= site_range < self.grid.R.max() - 4.0
                if spike_possible:
                    reach += min(42.0, max(16.0, 0.20 * site_range + 13.0))
                m = dist < reach
                if not m.any():
                    continue
                dx, dy = g.X[m] - sx, g.Y[m] - sy
                th = np.deg2rad(st.heading(s, when))
                fx, fy = np.sin(th), np.cos(th)
                lx, ly = -np.cos(th), np.sin(th)
                u, v = dx * fx + dy * fy, dx * lx + dy * ly
                if s.get('split', 'right') == 'left':
                    v = -v
                h = h_all[m]
                d = dynamic.reflectivity(u, v, h, age_min, params,
                                         seed=self.seed, storm_id=s.get('id', ''))
                with np.errstate(invalid='ignore'):
                    # The dynamic engine owns its seeded texture; its return
                    # is already the final storm dBZ value.
                    sdb = np.asarray(d, dtype=float)
                    power = np.where(np.isfinite(sdb), 10 ** (sdb / 10), 0)
                    if spike_possible:
                        finite_db = sdb[np.isfinite(sdb)]
                        if finite_db.size:
                            spike = self._hail_spike(g.X[m], g.Y[m], h, sx, sy, float(finite_db.max()))
                            if spike is not None:
                                power += spike
                    lin[m] += power
                continue
            prims = st.primitives(s, when)
            if not prims:
                continue
            lat, lon = st.position(s, when)
            sx, sy = self.to_xy.transform(lon, lat)
            th = np.deg2rad(st.heading(s, when))
            fx, fy = np.sin(th), np.cos(th)
            lx, ly = -np.cos(th), np.sin(th)
            dist = np.hypot(g.X - sx, g.Y - sy)
            core_peak = self._peak_dbz(prims)
            spike_ok = core_peak >= 64.5 and np.hypot(sx, sy) >= 8.0 and np.hypot(sx, sy) < g.R.max() - 4.0
            reach = max(prim.reach(P) for P in prims) + 3
            if spike_ok:
                reach += min(42.0, max(16.0, 0.20 * np.hypot(sx, sy) + 13.0))
            m = dist < reach
            if not m.any():
                continue
            dx, dy = g.X[m] - sx, g.Y[m] - sy
            u, v = dx * fx + dy * fy, dx * lx + dy * ly
            detail, wu, wv = self.tex(s['id'])
            u2 = u + self.warp_km * _sample(wu, u, v)
            v2 = v + self.warp_km * _sample(wv, u, v)
            h, half = h_all[m], half_all[m]
            slin = np.zeros(u.shape)
            sub = np.zeros(u.shape)
            for P in prims:
                w = _vweight(h, half, P['z'])
                if P['kind'] in prim.SUBTRACTIVE:
                    sub += prim.SUBTRACTIVE[P['kind']](u2, v2, P) * w
                else:
                    d = prim.ADDITIVE[P['kind']](u2, v2, P)
                    with np.errstate(over='ignore', invalid='ignore'):
                        slin += np.where(np.isfinite(d), 10 ** (d / 10), 0) * w
            if spike_ok:
                spike = self._hail_spike(g.X[m], g.Y[m], h, sx, sy, core_peak)
                if spike is not None:
                    slin += spike
            tex_db = self.texture_db * _sample(detail, u, v)
            with np.errstate(divide='ignore'):
                sdb = 10 * np.log10(slin) - sub + tex_db
            lin[m] += np.where(np.isfinite(sdb), 10 ** (sdb / 10), 0)
        # beam smoothing: ~0.95 deg across azimuth (constant in angle), a gate or two in range
        lin = gaussian_filter1d(lin, 0.8, axis=0, mode='wrap')
        lin = gaussian_filter1d(lin, 0.6, axis=1, mode='nearest')
        with np.errstate(divide='ignore'):
            dbz = 10 * np.log10(lin)
        # soft ceiling: real reflectivity rarely exceeds ~75 dBZ even in giant hail
        return np.where(dbz > 66, 66 + 10 * np.tanh((dbz - 66) / 10), dbz)

    def _fl_noise(self, x, y):
        """Smooth noise (std ~1, 5-20 km features) on a 1 km grid over +-250 km, sampled at km points."""
        if not hasattr(self, '_fln'):
            rng = np.random.default_rng(self.seed + 7)
            n = gaussian_filter(rng.standard_normal((501, 501)), 4.0) + 0.6 * gaussian_filter(rng.standard_normal((501, 501)), 9.0)
            self._fln = n / n.std()
        return map_coordinates(self._fln, [np.asarray(y) + 250, np.asarray(x) + 250], order=1, mode='nearest')

    def fine_lines(self, when, elev):
        """Weak (~8-18 dBZ) thin lines from insects/refractivity gradients along boundaries, seen only while the beam
        is still in the boundary layer (roughly within 80-120 km of the radar)."""
        import shapely
        from shapely.geometry import LineString
        g = self.grid
        h = beam_height(g.R, elev)
        near = h < 1.6
        out = np.full(g.R.shape, -np.inf)
        if not self.lines_fn or not near.any():
            return out
        pts = shapely.points(g.X[near], g.Y[near])
        rng = np.random.default_rng(zlib.crc32(when.encode()))
        best = np.full(pts.shape, -np.inf)
        for k, (kind, lons, lats, s) in enumerate(self.lines_fn(when)):
            x, y = self.to_xy.transform(np.asarray(lons), np.asarray(lats))
            line = LineString(np.c_[x, y])
            d = shapely.distance(pts, line)
            m = d < 4
            if not m.any():
                continue
            # texture: a fixed, smooth 2-D noise field in space (steady scan to scan) -> gaps, beads, width changes
            mod = self._fl_noise(g.X[near][m], g.Y[near][m])
            amp = np.clip(0.55 + 0.6 * mod, 0, 1.15)                      # ~20% of the line drops out
            width = 0.65 + 0.45 * np.clip(0.5 + mod, 0, 1)                   # 0.65-1.1 km e-folding half-width
            db = (6 + 10 * s) * amp * np.exp(-(d[m] / width) ** 2) + rng.normal(0, 3.0, mod.shape)
            db = np.where(rng.random(mod.shape) < 0.22, -np.inf, db)          # grainy insect returns
            cur = np.full(pts.shape, -np.inf); cur[m] = db
            best = np.fmax(best, cur)
        best = best - 8 * np.clip(h[near] - 0.9, 0, None)          # fade as the beam climbs out of the PBL
        out[near] = best
        return out

    def base(self, when, elev=0.5, clutter=True):
        dbz = self.tilt(elev, when)
        if self.lines_fn:
            dbz = np.fmax(dbz, self.fine_lines(when, elev))
        if clutter:
            c = ground_clutter(self.grid, np.random.default_rng(self.seed + 1))
            dbz = np.fmax(dbz, c)
        return np.where(dbz >= 5, dbz, np.nan)

    def composite(self, when, tilts=VCP212, clutter=True):
        out = np.full(self.grid.R.shape, -np.inf)
        for e in tilts:
            out = np.fmax(out, self.tilt(e, when))
        if self.lines_fn:
            out = np.fmax(out, self.fine_lines(when, tilts[0]))
        if clutter:
            out = np.fmax(out, ground_clutter(self.grid, np.random.default_rng(self.seed + 1)))
        return np.where(out >= 5, out, np.nan)
