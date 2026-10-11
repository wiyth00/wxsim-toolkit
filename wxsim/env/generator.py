"""Environment-driven storm generator ("bootleg CM1").

usage: python -m wxsim.env.generator scenarios/<name>/env_XXX.json [--seed N] [--outdir DIR] [--no-maps]

Time-steps storms through an Environment:
  initiation   stochastic, where boundary/mesolow lift (or heating) can overcome CIN, with spacing suppression
  organization shear, CAPE and SRH set how quickly a cell becomes a supercell (sup 0..1)
  structure    LP <-> classic <-> HP from storm-relative anvil flow, PW and LCL (plus mergers);
               hook/notch prominence from 0-1 km SRH and LCL; elevated over stable air; high-based at high LCL
  motion       Bunkers right/left-mover from the local mean wind and shear (ordinary cells move with the mean wind)
  splitting    straighter hodographs spawn left-moving splits
  interactions mergers (the survivor grows and turns HP); clusters in line-favorable environments grow upscale
  decay        storms die when their inflow runs out of usable CAPE or they reach their natural lifetime
  hazards      tornado episodes from inflow STP, peak hail from SHIP
Writes a storms file the radar / hazards tools read directly, plus a storm log.
It is rule-based (conceptual models + parameter climatology), not a numerical cloud model.
"""
import argparse, json, math
from datetime import timedelta
from pathlib import Path

import numpy as np

from .environment import Environment, uv_dir
from .boundaries import DynamicBoundary, ColdPools, Pool
from ..radar.storms import parse_time

KT_KMH = 1.852


def iso(t):
    return t.strftime('%Y-%m-%dT%H:%MZ')


def relax(x, target, dt, tau):
    return x + (target - x) * (1 - math.exp(-dt / tau))


class Storm:
    _n = 0

    def __init__(self, lat, lon, t, rng, split='right', parent=None, **kw):
        Storm._n += 1
        self.id = kw.get('sid') or f'G{Storm._n:02d}'
        self.lat, self.lon, self.born, self.split, self.parent = lat, lon, t, split, parent
        self.alive, self.died, self.age = True, None, 0.0
        self.u = self.v = None
        self.sup, self.prec, self.hook, self.elev, self.hb = kw.get('sup', 0.0), kw.get('prec', 0.0), 0.2, 0.0, 0.0
        self.scale, self.hail, self.dz, self.decaying = kw.get('scale', 0.9), 55.0, -18.0, False
        self.lifetime = kw.get('lifetime', rng.uniform(40, 80))
        self.did_split = split == 'left'
        self.merge_bonus = 0.0
        # Continuous morphology controls.  These are deliberately state-only: all
        # stochastic evolution remains on ``self.rng`` below, in the same order as
        # before these controls were added.
        self.line = float(np.clip(kw.get('line', 0.0), 0.0, 1.0))
        self.bow = float(np.clip(kw.get('bow', 0.0), 0.0, 1.0))
        self.tor_level = float(np.clip(kw.get('tor', 0.0), 0.0, 1.0))
        self.cycle = float(np.clip(kw.get('cycle', 0.0), 0.0, 1.0))
        self.lowtop = float(np.clip(kw.get('lowtop', 0.0), 0.0, 1.0))
        self.mlcape, self.shear_kt, self.rh_low, self.srh01 = 0.0, 0.0, 50.0, 0.0
        self.split_core = 0.0
        self.merge = 0.0
        self.merge_fade_event = False
        self._cycle_phase = 0.0
        self._cycle_initialized = False
        self._cycle_tor_start = None
        self.tor, self.tors = None, []
        self.max_hail, self.max_sup, self.modes = 0.0, 0.0, set()
        self.track, self.frames = [], []
        self.haz, self.hazards = None, []
        self.qlcs = False

    def params(self):
        return dict(prec=round(self.prec, 3), sup=round(self.sup, 3), hook=round(self.hook, 3), elev=round(self.elev, 3),
                    hb=round(self.hb, 3), scale=round(self.scale, 3), hail=round(self.hail, 1), dz=round(self.dz, 2),
                    line=round(self.line, 3), bow=round(self.bow, 3), tor=round(self.tor_level, 3),
                    cycle=round(self.cycle, 3), lowtop=round(self.lowtop, 3),
                    split_core=round(self.split_core, 3), merge=round(self.merge, 3),
                    mlcape=round(self.mlcape, 1), shear_kt=round(self.shear_kt, 1),
                    rh_low=round(self.rh_low, 1), srh01=round(self.srh01, 1))

    def mode(self):
        if self.qlcs:
            return 'qlcs'
        if self.elev > 0.6:
            return 'elevated'
        if self.hb > 0.6:
            return 'high-based'
        if self.sup > 0.6:
            return 'supercell' + (' (LP)' if self.prec < -0.45 else ' (HP)' if self.prec > 0.45 else '')
        return 'multicell' if self.sup > 0.3 else 'ordinary'


class Generator:
    def __init__(self, cfg, seed=None, morphology='procedural'):
        self.cfg = cfg
        self.seed = cfg.get('seed', 1) if seed is None else seed
        if morphology not in {'procedural', 'dynamic'}:
            raise ValueError("morphology must be 'procedural' or 'dynamic'")
        self.morphology = morphology
        self.rng = np.random.default_rng(self.seed)
        self.hrng = np.random.default_rng(self.seed + 7919)   # hazard noise only, so storm evolution per seed is unchanged
        self.env = Environment(cfg, seed=self.seed)
        self.dt = cfg.get('step_min', 10)
        self.ini = cfg.get('initiation', {})
        self.evo = cfg.get('evolution', {})
        Storm._n = 0
        Pool._n = 0
        self.storms = []
        self.dyn = [DynamicBoundary(b, self.env) for b in cfg.get('boundaries', []) if 'dynamic' in b]
        oc = cfg.get('outflows', {'enabled': True})
        self.pools = ColdPools(self.env, oc) if oc.get('enabled', True) else None
        self.env.pools = self.pools

    # ------------------------------------------------------------------ env helpers
    def _env(self, lat, lon, t, exclude=None):
        F = self.env.sample(np.array([lon]), np.array([lat]), iso(t), exclude=exclude)
        return {k: float(np.ravel(v)[0]) for k, v in F.items()}

    def _offset(self, lat, lon, km_e, km_n):
        return lat + km_n / 110.57, lon + km_e / (111.32 * math.cos(math.radians(lat)))

    def _update_morphology(self, s, E, E_here, dt):
        """Advance continuous radar morphology without touching the legacy RNG stream.

        The fields here are intentionally derived from the already sampled environment,
        storm age, and interaction state.  They are written to keyframes and consumed by
        the selected radar morphology model; none participates in track or hazard decisions.
        """
        # Preserve a compact inflow summary for the optional state-driven radar field.
        s.mlcape = float(E.get('mlcape', 0.0))
        s.shear_kt = float(E.get('shear_kt', 0.0))
        s.rh_low = float(E.get('rh_low', 50.0))
        s.srh01 = float(E.get('srh01', 0.0))
        # A local density proxy lets isolated supercells stay discrete while a cluster
        # gradually acquires embedded-line character.
        nearby = 0
        for o in self.storms:
            if o is s or not o.alive:
                continue
            if math.hypot((s.lon - o.lon) * 93, (s.lat - o.lat) * 110.6) < 45:
                nearby += 1
        density = float(np.clip(nearby / 4.0, 0.0, 1.0))
        linear = float(np.clip(E.get('linear', 0.0), 0.0, 1.0))
        age_f = float(np.clip(s.age / 90.0, 0.0, 1.0))
        line_target = float(np.clip(0.62 * linear + 0.23 * density + 0.15 * age_f, 0.0, 1.0))
        # QLCS promotion remains a smooth ramp; the old qlcs/evolve metadata is
        # retained separately for readers that still switch archetypes.
        if s.qlcs:
            line_target = max(line_target, 0.78 + 0.22 * linear)
        s.line = float(np.clip(relax(s.line, line_target, dt, 25), 0.0, 1.0))

        # Cold-pool edge strength supplies the rear-inflow forcing.  The fallback to
        # pool_in keeps this useful for runs with a very broad or newly formed pool.
        cold_pool = float(np.clip(max(E.get('pool_edge', 0.0), 0.55 * E.get('pool_in', 0.0)), 0.0, 1.0))
        shear_f = float(np.clip((E.get('shear_kt', 0.0) - 35.0) / 35.0, 0.0, 1.0))
        dry_f = float(np.clip((72.0 - E.get('rh_low', 50.0)) / 35.0, 0.0, 1.0))
        bow_target = float(np.clip(s.line * cold_pool * (0.35 + 0.65 * shear_f) * (0.55 + 0.45 * dry_f), 0.0, 1.0))
        s.bow = float(np.clip(relax(s.bow, bow_target, dt, 25), 0.0, 1.0))

        # A short e-folding ramp avoids a binary hook/debris-ball appearance when a
        # tornado starts or ends between two 10-minute generator frames.
        tor_target = 1.0 if s.tor is not None else 0.0
        s.tor_level = float(np.clip(relax(s.tor_level, tor_target, dt, 4), 0.0, 1.0))

        # Mature storms cycle on a deterministic ~15--30 minute period, lengthened
        # when needed to avoid aliasing at the generator's output cadence. A new
        # tornado episode resets phase when there is one.
        tor_start = s.tor.get('start') if s.tor is not None else None
        new_tornado_cycle = tor_start is not None and tor_start != s._cycle_tor_start
        if new_tornado_cycle:
            s._cycle_tor_start = tor_start
        # Keep phase advance below half a turn per sample. At the default
        # 10-minute cadence this yields 22--30 minute cycles, avoiding aliasing
        # between keyframes while staying within the requested approximate range.
        period_floor = min(30.0, max(15.0, 2.0 * dt + 2.0))
        period = period_floor + (30.0 - period_floor) * (1.0 - float(np.clip(s.sup, 0.0, 1.0)))
        if new_tornado_cycle or not s._cycle_initialized:
            s._cycle_phase = 0.0
            s._cycle_initialized = True
        else:
            s._cycle_phase = (s._cycle_phase + dt / period) % 1.0
        # Keep `cycle` a true circular phase; the radar scales its visible effect
        # continuously with organization. Multiplying the phase itself by maturity
        # would create a discontinuity each time the phase wraps from 1 back to 0.
        s.cycle = float(s._cycle_phase)

        cape_f = float(np.clip((1250.0 - E.get('mlcape', 0.0)) / 650.0, 0.0, 1.0))
        high_shear = float(np.clip((E.get('shear_kt', 0.0) - 45.0) / 25.0, 0.0, 1.0))
        lowtop_target = float(np.clip(cape_f * high_shear * (1.0 - 0.35 * s.elev), 0.0, 1.0))
        s.lowtop = float(np.clip(relax(s.lowtop, lowtop_target, dt, 25), 0.0, 1.0))

        # Interaction pulses are morphology-only.  The fade/growth pulse is consumed
        # by the procedural archetype while the existing merge_bonus still drives the
        # legacy lifecycle and hazard behavior exactly as before.
        s.split_core = max(0.0, s.split_core - dt / 25.0)
        s.merge = float(np.clip(relax(s.merge, 0.0, dt, 18), 0.0, 1.0))

    # ------------------------------------------------------------------ initiation
    def initiate(self, t):
        ini = self.ini
        lon0, lon1, lat0, lat1 = self.cfg['domain']
        res = ini.get('grid_deg', 0.1)
        lons, lats = np.meshgrid(np.arange(lon0 + res / 2, lon1, res), np.arange(lat0 + res / 2, lat1, res))
        F = self.env.sample(lons, lats, iso(t))
        f_cin = 1 / (1 + np.exp(-(F['mlcin'] + ini.get('cin_break', 35)) / 8))
        f_heat = 1 / (1 + np.exp(-(F['mlcin'] + ini.get('heat_cin_break', 12)) / 4))
        f_cape = 1 / (1 + np.exp(-(F['mlcape'] - ini.get('min_cape', 400)) / 200))
        sb = 1 - np.clip(F['elevated'], 0, 1)
        rate = (ini.get('ci_rate', 0.3) * F['lift'] * f_cin + F['heating_ci'] * f_heat) * f_cape * sb   # per hr per 1000 km2
        area = (res * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (res * 110.57)
        lam = rate * area / 1000 * self.dt / 60
        counts = self.rng.poisson(lam)
        new = []
        spacing = ini.get('min_spacing_km', 30)
        alive = [(s.lat, s.lon) for s in self.storms if s.alive]
        for j, i in zip(*np.nonzero(counts)):
            lat = lats[j, i] + self.rng.uniform(-res / 2, res / 2)
            lon = lons[j, i] + self.rng.uniform(-res / 2, res / 2)
            near = [(a, b) for a, b in alive + new]
            if any(math.hypot((lon - b) * 93, (lat - a) * 110.6) < spacing for a, b in near):
                continue
            new.append((lat, lon))
            life = self.rng.uniform(*ini.get('lifetime_min', [45, 90]))
            s = Storm(lat, lon, t, self.rng, lifetime=life)
            self.storms.append(s)

    # ------------------------------------------------------------------ one storm, one step
    def step_storm(self, s, t):
        dt, evo, rng = self.dt, self.evo, self.rng
        own = s.id if s.sup > 0.5 else None          # organized storms are not undercut by their own outflow
        E_here = self._env(s.lat, s.lon, t, own)
        # inflow sample: ahead and to the right (right movers) / left (left movers) of the storm
        hd = math.atan2(E_here['mean_u'], E_here['mean_v']) if s.u is None else math.atan2(s.u, s.v)   # heading, rad cw from N
        side = 1 if s.split == 'right' else -1
        ang = hd + side * math.radians(70)
        ilat, ilon = self._offset(s.lat, s.lon, 18 * math.sin(ang), 18 * math.cos(ang))
        E = self._env(ilat, ilon, t, own)
        s.last_inflow = E
        elevated_here = min(1.0, max(0.0, E_here['elevated']))
        usable = 0.55 * E['mucape'] if elevated_here > 0.5 else E['mlcape']     # elevated inflow is less efficient
        usable_cin = -20 if elevated_here > 0.5 else E['mlcin']
        # organization target
        shear_f = np.clip((E['shear_kt'] - 28) / 22, 0, 1)
        cape_f = np.clip(usable / 1000, 0, 1)
        srh_f = np.clip(E['srh03'] / 150, 0.35, 1)
        if s.split == 'left':
            srh_f = np.clip(1.1 - E['hodo_curve'], 0.2, 1)      # left movers need straight hodographs
        o_target = float(shear_f * cape_f * srh_f)
        age_cap = min(1.0, s.age / evo.get('organize_min', 35))
        s.sup = relax(s.sup, min(o_target, age_cap), dt, 20)
        # structure targets
        p_target = 0.9 * (E['pw'] - 0.95) / 0.35 - 0.8 * (E['sr_anvil'] - 45) / 20 + 0.35 * (1400 - E['lcl']) / 800 \
            + s.merge_bonus
        s.prec = relax(s.prec, float(np.clip(p_target, -1, 1)), dt, 30)
        h_target = float(np.clip(E['srh01'] / 250, 0, 1) * np.clip((2300 - E['lcl']) / 1300, 0.15, 1))
        s.hook = relax(s.hook, h_target, dt, 20)
        s.elev = relax(s.elev, elevated_here, dt, 15)
        s.hb = relax(s.hb, float(np.clip((E['lcl'] - 1900) / 1100, 0, 1)) * (1 - elevated_here), dt, 20)
        s.scale = relax(s.scale, 0.85 + 0.35 * min(usable / 2500, 1.3) + 0.25 * s.sup + 0.12 * s.merge_bonus, dt, 30)
        s.hail = relax(s.hail, float(np.clip(55 + 5.5 * E['ship'] * (0.5 + 0.5 * s.sup), 50, 72)), dt, 15)
        # intensity / lifecycle
        if s.age < 30:
            s.dz = -18 * (1 - s.age / 30) ** 1.5
        elif not s.decaying:
            s.dz = relax(s.dz, 0.0, dt, 15)
        starving = usable < evo.get('min_usable_cape', 350) or usable_cin < -120
        longevity = 1 + 7.0 * max(0.0, s.sup - 0.5) * (1 - 0.6 * s.elev)   # mature supercells live much longer
        too_old = s.age > min(s.lifetime * longevity,
                              evo.get('max_qlcs_lifetime_min', 300) if s.qlcs
                              else evo.get('max_discrete_lifetime_min', float('inf')))
        if (starving or too_old) and s.age > 20:
            s.decaying = True
        if s.decaying:
            s.dz -= 3.5 * dt / 10
            if s.dz < -24:
                s.alive, s.died = False, t
        # motion: ordinary cells ride the mean wind, supercells deviate (Bunkers)
        mu, mv = E_here['mean_u'], E_here['mean_v']
        du, dv = (E_here['rm_u'] - mu, E_here['rm_v'] - mv) if s.split == 'right' else (E_here['lm_u'] - mu, E_here['lm_v'] - mv)
        tu = mu + s.sup * du + rng.normal(0, 1.5)
        tv = mv + s.sup * dv + rng.normal(0, 1.5)
        s.u = tu if s.u is None else relax(s.u, tu, dt, 20)
        s.v = tv if s.v is None else relax(s.v, tv, dt, 20)
        km = dt / 60 * KT_KMH
        s.lat, s.lon = self._offset(s.lat, s.lon, s.u * km, s.v * km)
        # splitting (straight hodographs)
        if (not s.did_split and s.split == 'right' and s.sup > 0.5 and 15 < s.age < 70
                and rng.random() < evo.get('split_rate', 0.3) * max(0.0, 0.6 - E['hodo_curve']) / 0.6):
            s.did_split = True
            s.split_core = 1.0
            lm = Storm(s.lat, s.lon, t, rng, split='left', parent=s.id, sup=s.sup * 0.8, prec=s.prec,
                       scale=s.scale * 0.85, lifetime=rng.uniform(40, 90) * (1.2 - E['hodo_curve']),
                       sid=f'{s.id}L')
            lm.age, lm.dz, lm.hail = 10, -8, s.hail - 1
            lm.split_core = 1.0
            self.storms.append(lm)
        # tornadoes: start probability scales with inflow STP; a refractory period follows each tornado
        # (mesocyclone cycling); EF rating drawn from an exponential tail that fattens with STP
        stp = E['stp']
        if s.tor is None:
            ok = s.sup > 0.7 and s.hook > 0.5 and s.elev < 0.3 and s.split == 'right' and not s.decaying
            if ok and s.age >= getattr(s, 'tor_ready', 0):
                p = evo.get('tor_rate', 0.012) * min(stp, 10) * (0.35 if s.qlcs else 1.0)
                if rng.random() < min(0.3, p):
                    lam = (0.55 + 0.12 * min(stp, 10)) * (0.6 if s.qlcs else 1.0) * evo.get('ef_scale', 1.0)
                    x = rng.exponential(lam)
                    if evo.get('ef_overflow', 'clip') == 'resample':    # draws past EF5 are redrawn, not piled onto EF5
                        for _ in range(3):
                            if x < 6:
                                break
                            x = rng.exponential(lam)
                        x = x if x < 6 else rng.uniform(3, 5)
                    ef = int(min(5, np.floor(x)))
                    s.tor = dict(start=iso(t), lat=round(s.lat, 3), lon=round(s.lon, 3), ef=ef)
        else:
            p_end = evo.get('tor_end_base', 0.6) - evo.get('tor_end_per_ef', 0.07) * s.tor['ef']   # stronger tornadoes last longer
            p_end = max(evo.get('tor_end_min', 0.08), p_end)
            dur = (t - parse_time(s.tor['start'])).total_seconds() / 60
            if rng.random() < p_end or stp < 0.5 or s.decaying or dur >= evo.get('tor_max_min', 1e9):
                s.tor.update(end=iso(t), end_lat=round(s.lat, 3), end_lon=round(s.lon, 3))
                s.tors.append(s.tor); s.tor = None
                s.tor_ready = s.age + evo.get('tor_refractory_min', 45)
        self._update_morphology(s, E, E_here, dt)
        hail_in = float(np.clip(0.6 + 0.9 * E['ship'] * s.sup + rng.normal(0, 0.25), 0.25, 5.5)) if s.dz > -8 else 0
        s.max_hail = max(s.max_hail, hail_in)
        # peak surface gust (kt): dry subcloud air and high bases (downbursts), organized rotation (RFD),
        # cold-pool-driven lines and mergers; elevated storms rarely mix severe gusts down
        dry = float(np.clip(1.3 - E['rh_low'] / 100, 0.3, 1.2))
        gust = (32 + 30 * dry * (0.4 + 0.6 * s.hb) + 10 * s.sup * np.clip(E['shear_kt'] / 60, 0, 1)
                + evo.get('qlcs_gust_bonus', 22) * (1.0 if s.qlcs else 0.0) + 6 * s.merge_bonus + 8 * (s.prec + 1) / 2 * (1 - s.hb)
                + self.hrng.normal(0, 6)) * (1 - 0.8 * s.elev)
        gust = float(gust) if s.dz > -8 else 0.0
        s.haz = [round(hail_in, 2), round(max(gust, 0.0), 1), s.tor['ef'] if s.tor is not None else -1]
        s.max_sup = max(s.max_sup, s.sup)
        s.modes.add(s.mode())
        s.age += dt

    # ------------------------------------------------------------------ interactions
    def mergers(self, t):
        live = [s for s in self.storms if s.alive]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if not (a.alive and b.alive):
                    continue
                if (a.parent == b.id or b.parent == a.id) and min(a.age, b.age) < 90:
                    continue                                   # fresh splits diverge instead of merging
                if math.hypot((a.lon - b.lon) * 93, (a.lat - b.lat) * 110.6) < self.evo.get('merge_km', 11):
                    keep, lose = (a, b) if (a.sup, a.age) >= (b.sup, b.age) else (b, a)
                    keep.merge_bonus = min(1.0, keep.merge_bonus + 0.35)
                    # Expose a one-frame interaction pulse to the morphology reader.
                    # The survivor grows; the absorbed core gets a final fade marker.
                    keep.merge = min(1.0, keep.merge + 0.85)
                    lose.merge = min(1.0, lose.merge + 0.85)
                    lose.merge_fade_event = True
                    if keep.frames:
                        keep.frames[-1]['params']['merge'] = round(keep.merge, 3)
                    if lose.frames:
                        lose.frames[-1]['params']['merge'] = round(lose.merge, 3)
                        # The radar archetype interprets larger merge_fade as more
                        # attenuation, so the physical merger frame starts at 0.
                        lose.frames[-1]['params']['merge_fade'] = 0.0
                    lose.alive, lose.died = False, t
                    lose.merged_into = keep.id

    def upscale(self, t):
        """Nearby cells in a line-favoring environment consolidate into a QLCS segment."""
        evo = self.evo
        live = [s for s in self.storms if s.alive and not s.qlcs and s.age > evo.get('qlcs_min_age', 40)]
        for s in live:
            if not s.alive:
                continue
            group = [o for o in live if o.alive and math.hypot((o.lon - s.lon) * 93, (o.lat - s.lat) * 110.6) < evo.get('qlcs_group_radius_km', 45)]
            if len(group) >= evo.get('qlcs_min_cells', 3):
                E = self._env(s.lat, s.lon, t)
                if E['linear'] > evo.get('qlcs_linear_threshold', 0.5) and self.rng.random() < evo.get('qlcs_probability', 0.3) * E['linear']:
                    lead = max(group, key=lambda o: o.sup)
                    lead.qlcs, lead.lifetime = True, lead.age + 180
                    for o in group:
                        if o is not lead:
                            o.alive, o.died, o.merged_into = False, t, lead.id

    # ------------------------------------------------------------------ run
    def run(self):
        t = parse_time(self.cfg['start']); end = parse_time(self.cfg['end'])
        ci_start = parse_time(self.ini.get('window', [self.cfg['start'], self.cfg['end']])[0])
        ci_end = parse_time(self.ini.get('window', [self.cfg['start'], self.cfg['end']])[1])
        while t <= end:
            if ci_start <= t <= ci_end:
                self.initiate(t)
            for s in [s for s in self.storms if s.alive]:
                self.step_storm(s, t)
                s.track.append([iso(t), round(s.lat, 4), round(s.lon, 4)])
                if s.haz is not None:
                    s.hazards.append([iso(t), round(s.lat, 4), round(s.lon, 4)] + s.haz)
                s.frames.append({'at': iso(t), 'params': s.params() | {'qlcs': 1.0 if s.qlcs else 0.0}})
                if self.pools is not None and s.alive:
                    self.pools.feed(s, t, s.last_inflow)
            self.mergers(t)
            self.upscale(t)
            if self.pools is not None:
                self.pools.step(t, self.dt)
            sxy = [self.env.km(s.lon, s.lat) + (float(np.clip((s.dz + 12) / 12, 0, 1)),)
                   for s in self.storms if s.alive]
            for db in self.dyn:
                db.step(t, self.dt, sxy)
            t += timedelta(minutes=self.dt)
        for s in self.storms:
            if s.tor is not None:
                s.tor.update(end=s.track[-1][0], end_lat=s.track[-1][1], end_lon=s.track[-1][2]); s.tors.append(s.tor)
            if s.died is None:
                s.died = parse_time(s.track[-1][0])
        return self.storms

    def to_storms_file(self):
        out = []
        for s in self.storms:
            if len(s.track) < 2:
                continue
            kf = [{'at': f['at'], 'params': {k: v for k, v in f['params'].items() if k != 'qlcs'}} for f in s.frames]
            # A nearby merger ends the absorbed storm's physical track immediately,
            # but its terminal radar echo should fade into the survivor over two more
            # frames.  These are serialization-only frames: no track, hazard, tornado,
            # or RNG state is changed.  Upscale grouping uses a larger radius and does
            # not set ``merge_fade_event``, avoiding artificial long-distance fades.
            radar_end = parse_time(s.track[-1][0])
            if getattr(s, 'merge_fade_event', False) and kf:
                base = dict(kf[-1]['params'])
                base_merge = float(base.get('merge', 0.0))
                base_dz = float(base.get('dz', 0.0))
                for n, fade in enumerate((0.55, 1.0), 1):
                    p = dict(base)
                    growth = (0.55, 0.15)[n - 1]
                    p['merge'] = round(base_merge * growth, 3)
                    p['merge_fade'] = round(fade, 3)
                    # Fade the terminal echo in dBZ as well as through the
                    # primitive attenuation, so the endpoint is faint before
                    # the extended radar life expires.
                    p['dz'] = round(base_dz - 8.0 * fade, 2)
                    at = radar_end + timedelta(minutes=self.dt * n)
                    kf.append({'at': iso(at), 'params': p})
                radar_end = radar_end + timedelta(minutes=self.dt * 2)
            d = {'id': s.id, 'archetype': self.morphology, 'split': s.split, 'track': s.track, 'keyframes': kf,
                 'life': {'start': s.track[0][0], 'end': iso(radar_end)},
                 'hazards': s.hazards,
                 'motion': {'dir_from': 0, 'speed_kt': 0},
                 'meta': {'parent': s.parent, 'modes': sorted(s.modes), 'max_hail_in': round(s.max_hail, 2),
                          'max_sup': round(s.max_sup, 2), 'tornadoes': s.tors,
                          'merged_into': getattr(s, 'merged_into', None)}}
            if s.qlcs and self.morphology == 'procedural':
                d['evolve'] = [{'at': next(f['at'] for f in s.frames if f['params']['qlcs']), 'archetype': 'qlcs_segment'}]
            out.append(d)
        fine = {'dryline': 0.5, 'wedge_front': 0.35, 'cold_front': 0.6, 'warm_front': 0.3}
        lines = [{'name': b['name'], 'kind': b.get('kind', 'boundary'), 'fine_line': b.get('fine_line', fine.get(b.get('kind'), 0.3)),
                  'track': b['track']} for b in self.cfg.get('boundaries', [])]
        return {'_doc': f'Generated by wxsim.env.generator (seed {self.seed}, morphology {self.morphology}). Tracks and keyframed morphology per storm; '
                        f'boundary positions and cold-pool outlines through the run. Per storm, "hazards" rows are '
                        f'[time, lat, lon, hail_in, peak_gust_kt, tornado_EF or -1] each step (read by wxsim.reports).',
                'storms': out,
                'boundaries': {'lines': lines, 'pools': self.pools.export() if self.pools is not None else []}}


def summarize(storms_file, tz_h=-4):
    rows = []
    for s in storms_file['storms']:
        m = s['meta']
        t0, t1 = parse_time(s['life']['start']), parse_time(s['life']['end'])
        tors = ', '.join(f"EF{x['ef']} {x['start'][11:16]}-{x['end'][11:16]}Z" for x in m['tornadoes']) or '-'
        rows.append(f"{s['id']:<5} {s['split'][0].upper()}  {t0.strftime('%H%MZ')}-{t1.strftime('%H%MZ')}  "
                    f"{int((t1 - t0).total_seconds() / 60):>4} min  hail {m['max_hail_in']:.2f}\"  "
                    f"{'/'.join(m['modes']):<40} tornadoes: {tors}")
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--seed', type=int); ap.add_argument('--outdir')
    ap.add_argument('--no-maps', action='store_true')
    ap.add_argument('--ensemble', type=int, help='run seeds 1..N and print an ensemble summary (no maps)')
    ap.add_argument('--morphology', choices=('procedural', 'dynamic'), default='procedural',
                    help='storm reflectivity model; dynamic writes a separate storms_gen_dynamic_seedN.json by default')
    ap.add_argument('--storms-file', help='override the generated storms JSON path (relative to the scenario directory)')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    base = Path(a.config).parent
    outdir = Path(a.outdir) if a.outdir else base / 'output'
    outdir.mkdir(parents=True, exist_ok=True)
    if a.ensemble:
        rows = []
        for sd in range(1, a.ensemble + 1):
            g = Generator(cfg, seed=sd, morphology=a.morphology); g.run(); sf = g.to_storms_file()
            st = sf['storms']
            tors = [x for s in st for x in s['meta']['tornadoes']]
            rows.append(dict(seed=sd, storms=len(st), supercells=sum(s['meta']['max_sup'] > 0.6 for s in st),
                             tornadoes=len(tors), strong_tornadoes=sum(x['ef'] >= 2 for x in tors),
                             max_ef=max([x['ef'] for x in tors], default=None),
                             hail_2in_storms=sum(s['meta']['max_hail_in'] >= 2 for s in st),
                             max_hail_in=max([s['meta']['max_hail_in'] for s in st], default=0)))
        print(f"{'seed':>4} {'storms':>6} {'supercells':>10} {'tornadoes':>9} {'EF2+':>5} {'maxEF':>5} {'2in-hail storms':>15} {'max hail':>8}")
        for r in rows:
            print(f"{r['seed']:>4} {r['storms']:>6} {r['supercells']:>10} {r['tornadoes']:>9} {r['strong_tornadoes']:>5} "
                  f"{str(r['max_ef']) if r['max_ef'] is not None else '-':>5} {r['hail_2in_storms']:>15} {r['max_hail_in']:>8.2f}")
        n = len(rows)
        print(f"P(any tornado) = {sum(r['tornadoes'] > 0 for r in rows) / n:.0%}   P(EF2+) = {sum(r['strong_tornadoes'] > 0 for r in rows) / n:.0%}"
              f"   mean 2-in hail storms = {np.mean([r['hail_2in_storms'] for r in rows]):.1f}")
        json.dump(rows, open(outdir / 'ensemble_summary.json', 'w'), indent=1)
        return
    g = Generator(cfg, seed=a.seed, morphology=a.morphology)
    g.run()
    sf = g.to_storms_file()
    if a.storms_file:
        storms_path = Path(a.storms_file)
        if not storms_path.is_absolute():
            storms_path = base / storms_path
    else:
        output_key = 'storms' if a.morphology == 'procedural' else 'dynamic_storms'
        fallback = 'storms_gen_seed{seed}.json' if a.morphology == 'procedural' else 'storms_gen_dynamic_seed{seed}.json'
        name = cfg.get('outputs', {}).get(output_key, fallback).format(seed=g.seed)
        storms_path = base / name
    storms_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sf, open(storms_path, 'w'), indent=1)
    print(f'saved {storms_path}  ({len(sf["storms"])} storms; morphology={a.morphology})')
    print(summarize(sf))
    if not a.no_maps:
        from . import maps
        maps.render_all(cfg, sf, g.env, outdir, g.seed)
        maps.radar_frames(cfg, storms_path, outdir, g.seed, base)


if __name__ == '__main__':
    main()
