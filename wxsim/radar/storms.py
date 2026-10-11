"""Storm definitions: tracks, lifecycles, archetype changes. The single source of truth for where storms are.

storms file:
{
  "storms": [
    {"id": "S2", "archetype": "classic_supercell",
     "anchor": {"time": "2027-04-17T22:31Z", "lat": 33.80, "lon": -83.62},
     "motion": {"dir_from": 250, "speed_kt": 40},
     "split": "right",                       # or "left" (mirror image)
     "params": {"scale": 1.0, "dz": 0, "hail": 70, "ffd_rot": 12, "anvil": 1.0},
     "life": {"start": "...", "peak": "...", "decay": "...", "end": "..."},   # optional
     "evolve": [{"at": "2027-04-17T22:00Z", "archetype": "classic_supercell", "params": {...}}]  # optional
    }, ...]
}
"""
import json
import math
from collections.abc import Mapping

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from ..util import offset_point
from .archetypes import ARCHETYPES


def parse_time(s):
    """'2027-04-17T22:31Z' -> aware datetime (UTC)."""
    return datetime.strptime(s.replace('Z', ''), '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc)


def minutes_between(a, b):
    return (parse_time(b) - parse_time(a)).total_seconds() / 60.0


def load(path):
    return json.load(open(path))['storms']


def _track_interp(storm, when):
    tr = storm['track']                                   # [[iso, lat, lon], ...]
    ts = [parse_time(p[0]).timestamp() for p in tr]
    t = parse_time(when).timestamp()
    return float(np.interp(t, ts, [p[1] for p in tr])), float(np.interp(t, ts, [p[2] for p in tr]))


def position(storm, when):
    """(lat, lon) of the storm at ISO time `when`: from its track if it has one, else anchor + motion vector."""
    if storm.get('track'):
        return _track_interp(storm, when)
    a, m = storm['anchor'], storm['motion']
    dt_h = minutes_between(a['time'], when) / 60.0
    toward = (m['dir_from'] + 180) % 360
    miles = m['speed_kt'] * 1.15078 * dt_h
    if miles >= 0:
        return offset_point(a['lat'], a['lon'], miles, toward)
    return offset_point(a['lat'], a['lon'], -miles, (toward + 180) % 360)


def life_dz(storm, when):
    """dBZ offset from the lifecycle: -inf before start/after end, ramps up to the peak, decays after 'decay'."""
    L = storm.get('life')
    if not L:
        return 0.0
    t = parse_time(when)
    start, end = parse_time(L['start']), parse_time(L['end'])
    if t < start or t > end:
        return -math.inf
    peak = parse_time(L.get('peak', L['start']))
    decay = parse_time(L.get('decay', L['end']))
    if t < peak:
        f = (t - start) / (peak - start)
        return -18 * (1 - f) ** 1.5
    if t > decay:
        f = (t - decay) / (end - decay)
        return -22 * f ** 1.3
    return 0.0


def _keyframe_params(storm, when):
    kf = storm['keyframes']                               # [{"at": iso, "params": {...}}, ...]
    t = parse_time(when).timestamp()
    ts = [parse_time(k['at']).timestamp() for k in kf]
    if t <= ts[0]:
        return dict(kf[0]['params'])
    if t >= ts[-1]:
        return dict(kf[-1]['params'])
    i = int(np.searchsorted(ts, t))
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    a, b = kf[i - 1]['params'], kf[i]['params']
    out = {}
    for k, va in a.items():
        if not (isinstance(va, (int, float)) and k in b):
            out[k] = va
            continue
        vb = b[k]
        if k == 'cycle':
            # Phase is circular: choose the shortest direction across 0/1.
            # This keeps an occlusion cycle continuous for e.g. .9 -> .1.
            delta = (vb - va) % 1.0
            if delta > 0.5:
                delta -= 1.0
            out[k] = (va + delta * w) % 1.0
        else:
            out[k] = va + (vb - va) * w
    return out


def dynamic_params(storm, when):
    """Return merged parameters for a ``dynamic`` storm at ``when``.

    Dynamic reflectivity has its own engine defaults, while storm files use
    the same ``params`` and ``keyframes`` metadata as the existing archetypes.
    Keep the merge here so every caller sees the same snapshot.  A small set
    of accepted default names keeps this compatible with dynamic engine
    implementations that expose either ``DEFAULTS`` or ``DEFAULT_PARAMS``.
    """
    from . import dynamic

    out = {}
    for name in ('DEFAULTS', 'DEFAULT_PARAMS', 'PARAM_DEFAULTS'):
        defaults = getattr(dynamic, name, None)
        if isinstance(defaults, Mapping):
            out.update(defaults)

    # ``defaults`` is useful for hand-authored storms and ``dynamic`` is an
    # optional metadata block.  The latter may hold an explicit params mapping
    # or direct engine knobs; metadata keys are kept out of the engine call.
    defaults = storm.get('defaults')
    if isinstance(defaults, Mapping):
        out.update(defaults)
    dyn = storm.get('dynamic')
    if isinstance(dyn, Mapping):
        if isinstance(dyn.get('defaults'), Mapping):
            out.update(dyn['defaults'])
        if isinstance(dyn.get('params'), Mapping):
            out.update(dyn['params'])
        else:
            out.update({k: v for k, v in dyn.items() if k not in ('defaults', 'params', 'keyframes')})
    out.update(storm.get('params', {}))
    if storm.get('keyframes'):
        out.update(_keyframe_params(storm, when))
    for ev in sorted(storm.get('evolve', []), key=lambda e: e['at']):
        if parse_time(when) >= parse_time(ev['at']):
            out.update(ev.get('params', {}))
    return out


def age_minutes(storm, when):
    """Age of a storm in minutes, using lifecycle or first-track origin."""
    life = storm.get('life') or {}
    origin = life.get('start')
    if origin is None and storm.get('track'):
        origin = storm['track'][0][0]
    if origin is None:
        origin = storm.get('anchor', {}).get('time')
    return minutes_between(origin, when) if origin else 0.0


def active(storm, when):
    """Whether a storm is within its optional lifecycle window."""
    life = storm.get('life') or {}
    t = parse_time(when)
    start, end = life.get('start'), life.get('end')
    return not ((start and t < parse_time(start)) or (end and t > parse_time(end)))


def primitives(storm, when):
    """Primitive list for the storm at `when`, in its storm-relative frame (already mirrored for left movers)."""
    # Dynamic storms are evaluated directly by RadarSim's dynamic engine and
    # deliberately have no primitive approximation.  Returning [] keeps
    # overlays and legacy callers safe without routing them through an unknown
    # ARCHETYPES entry.
    if storm.get('archetype') == 'dynamic':
        return []
    arche, params = storm['archetype'], dict(storm.get('params', {}))
    if storm.get('keyframes'):
        params.update(_keyframe_params(storm, when))
    # A procedural storm is continuous only when its file carries the new
    # morphology controls.  Older hand-written procedural files used evolve to
    # switch to qlcs_segment and must retain that behavior.
    has_line_control = 'line' in storm.get('params', {}) or any(
        'line' in k.get('params', {}) for k in storm.get('keyframes', [])
    )
    continuous_procedural = storm.get('archetype') == 'procedural' and has_line_control
    for ev in sorted(storm.get('evolve', []), key=lambda e: e['at']):
        if parse_time(when) >= parse_time(ev['at']):
            ev_arche = ev.get('archetype', arche)
            # Generated procedural storms still carry the historical qlcs_segment
            # evolve marker for older readers, but their continuous line/bow
            # controls must remain active through the promotion.
            if not (continuous_procedural and ev_arche == 'qlcs_segment'):
                arche = ev_arche
            params.update(ev.get('params', {}))
    dz = life_dz(storm, when)
    if dz == -math.inf:
        return []
    params['dz'] = params.get('dz', 0) + dz
    prims = ARCHETYPES[arche](**params)
    if storm.get('split', 'right') == 'left':
        for P in prims:
            for k in ('v', 'mv'):
                if k in P:
                    P[k] = -P[k]
            if 'rot' in P:
                P['rot'] = -P['rot']
            if 'bend' in P:
                P['bend'] = -P['bend']
            if P['kind'] in ('hook', 'arc'):
                P['phi0'], P['phi1'] = -P['phi0'], -P['phi1']
    return prims


def motion_at(storm, when=None):
    """(dir_from deg, speed kt) at `when` (from the track if present)."""
    if storm.get('track') and when is not None:
        tr = storm['track']
        t = parse_time(when)
        i = max(1, min(len(tr) - 1, int(np.searchsorted([parse_time(p[0]) for p in tr], t))))
        (t0, la0, lo0), (t1, la1, lo1) = tr[i - 1], tr[i]
        dt_h = (parse_time(t1) - parse_time(t0)).total_seconds() / 3600
        dx = (lo1 - lo0) * 111.32 * math.cos(math.radians((la0 + la1) / 2)); dy = (la1 - la0) * 110.57
        spd = math.hypot(dx, dy) / 1.852 / max(dt_h, 1e-6)
        toward = math.degrees(math.atan2(dx, dy)) % 360
        return (toward + 180) % 360, spd
    return storm['motion']['dir_from'], storm['motion']['speed_kt']


def heading(storm, when=None):
    """Direction of motion (deg, toward)."""
    return (motion_at(storm, when)[0] + 180) % 360


def meso_position(storm, when):
    """(lat, lon) of the mesocyclone (the hook's center) at `when`: where a TVS / tornado shows on radar.
    Falls back to the storm reference point for storms without a hook."""
    lat, lon = position(storm, when)
    hk = [P for P in primitives(storm, when) if P['kind'] == 'hook']
    if not hk:
        return lat, lon
    th = math.radians(heading(storm, when))
    mu, mv = hk[0]['mu'], hk[0]['mv']                       # km forward / km left of motion
    dx = mu * math.sin(th) - mv * math.cos(th)
    dy = mu * math.cos(th) + mv * math.sin(th)
    return lat + dy / 110.57, lon + dx / (111.32 * math.cos(math.radians(lat)))
