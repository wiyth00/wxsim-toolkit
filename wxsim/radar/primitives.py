"""Shape primitives, evaluated in a storm-relative frame.

u = km along the storm's motion (forward), v = km to the left of motion.
Additive primitives return dBZ (−inf outside); 'notch' primitives return a dB reduction.
Each primitive has a vertical extent z = [bottom, top] in km AGL, applied separately by the sampler.
"""
import numpy as np

NEG = -np.inf


def _rot(u, v, deg):
    if not deg:
        return u, v
    a = np.deg2rad(deg)
    return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)


def _profile(d, peak, edge, p, cutoff=1.5):
    """dBZ falls from `peak` at d=0 to `edge` at d=1 (shape exponent p), and is cut off at `cutoff`."""
    out = peak - (peak - edge) * np.power(np.clip(d, 0, None), p)
    return np.where(d <= cutoff, out, NEG)


def ellipse(u, v, P):
    a, b = _rot(u - P['u'], v - P['v'], P.get('rot', 0))
    d = np.sqrt((a / P['su']) ** 2 + (b / P['sv']) ** 2)
    return _profile(d, P['peak'], P.get('edge', 15), P.get('p', 1.6), P.get('cutoff', 1.5))


def plume(u, v, P):
    """Tapered, optionally bent band starting at (u,v), running `length` km along `rot` degrees (left-positive)."""
    a, c = _rot(u - P['u'], v - P['v'], P.get('rot', 0))
    L = P['length']
    t = np.clip(a / L, 0, 1)
    cc = c - P.get('bend', 0.0) * np.clip(a, 0, L) ** 2
    half = (P['w0'] + (P['w1'] - P['w0']) * t) / 2
    over = np.where(a < 0, -a / (P['w0'] / 2), np.where(a > L, (a - L) / (P['w1'] / 2), 0))
    d = np.sqrt(over ** 2 + (cc / half) ** 2)
    peak = P['p0'] + (P['p1'] - P['p0']) * t
    return _profile(d, peak, P.get('edge', 12), P.get('p', 1.4), P.get('cutoff', 1.4))


def hook(u, v, P):
    """Spiral tube wrapped cyclonically around a mesocyclone at (mu, mv).
    phi0 -> phi1 in degrees (counter-clockwise from +u); radius r0 -> r1; half-width w0 -> w1; dBZ p0 -> p1."""
    n = 48
    phi = np.deg2rad(np.linspace(P['phi0'], P['phi1'], n))
    rr = np.linspace(P['r0'], P['r1'], n)
    cu, cv = P['mu'] + rr * np.cos(phi), P['mv'] + rr * np.sin(phi)
    su, sv = cu[:-1], cv[:-1]
    du, dv = np.diff(cu), np.diff(cv)
    seg2 = du ** 2 + dv ** 2
    U, V = u[..., None], v[..., None]
    t = np.clip(((U - su) * du + (V - sv) * dv) / seg2, 0, 1)
    dist = np.hypot(U - (su + t * du), V - (sv + t * dv))
    k = np.argmin(dist, axis=-1)
    dmin = np.take_along_axis(dist, k[..., None], -1)[..., 0]
    tt = np.take_along_axis(t, k[..., None], -1)[..., 0]
    frac = (k + tt) / (n - 1)
    half = P['w0'] + (P['w1'] - P['w0']) * frac
    peak = P['p0'] + (P['p1'] - P['p0']) * frac
    return _profile(dmin / half, peak, P.get('edge', 20), P.get('p', 1.5), P.get('cutoff', 1.4))


def arc(u, v, P):
    """A curved comma/bookend segment with a smoothly varying radius and width.

    This is the additive counterpart to ``hook`` for broad QLCS bookends.  It
    intentionally uses the same vectorized nearest-segment calculation as a
    hook, while accepting one peak value (or p0/p1 for a tapered arc).
    """
    n = 32
    phi = np.deg2rad(np.linspace(P['phi0'], P['phi1'], n))
    rr = np.linspace(P['r0'], P['r1'], n)
    cu, cv = P['mu'] + rr * np.cos(phi), P['mv'] + rr * np.sin(phi)
    su, sv = cu[:-1], cv[:-1]
    du, dv = np.diff(cu), np.diff(cv)
    seg2 = du ** 2 + dv ** 2
    U, V = u[..., None], v[..., None]
    t = np.clip(((U - su) * du + (V - sv) * dv) / np.maximum(seg2, 1e-9), 0, 1)
    dist = np.hypot(U - (su + t * du), V - (sv + t * dv))
    k = np.argmin(dist, axis=-1)
    dmin = np.take_along_axis(dist, k[..., None], -1)[..., 0]
    tt = np.take_along_axis(t, k[..., None], -1)[..., 0]
    frac = (k + tt) / (n - 1)
    half = P['w0'] + (P['w1'] - P['w0']) * frac
    p0, p1 = P.get('p0', P.get('peak', 30)), P.get('p1', P.get('peak', 30))
    peak = p0 + (p1 - p0) * frac
    return _profile(dmin / np.maximum(half, 1e-6), peak, P.get('edge', 18), P.get('p', 1.5), P.get('cutoff', 1.5))


def line(u, v, P):
    """Convective line segment perpendicular to motion (QLCS); `bow` pushes the middle forward."""
    a, c = _rot(u - P['u'], v - P['v'], P.get('rot', 0))
    L = P['length']
    s = np.clip(c / (L / 2), -1, 1)
    a2 = a - P.get('bow', 0) * (1 - s ** 2)
    over = np.clip(np.abs(c) - L / 2, 0, None) / P['width']
    d = np.sqrt((a2 / P['width']) ** 2 + over ** 2)
    return _profile(d, P['peak'], P.get('edge', 25), P.get('p', 1.5), P.get('cutoff', 2.0))


def notch(u, v, P):
    a, b = _rot(u - P['u'], v - P['v'], P.get('rot', 0))
    d2 = (a / P['su']) ** 2 + (b / P['sv']) ** 2
    return P['depth'] * np.exp(-d2)


ADDITIVE = {'ellipse': ellipse, 'plume': plume, 'hook': hook, 'arc': arc, 'line': line}
SUBTRACTIVE = {'notch': notch}


def reach(P):
    """Rough radius (km) beyond which the primitive contributes nothing, used to cull gates."""
    k = P['kind']
    if k in ('ellipse', 'notch'):
        return np.hypot(P['u'], P['v']) + 1.6 * max(P['su'], P['sv'])
    if k == 'plume':
        return np.hypot(P['u'], P['v']) + P['length'] + P['w1'] + abs(P.get('bend', 0)) * P['length'] ** 2
    if k == 'hook':
        return np.hypot(P['mu'], P['mv']) + max(P['r0'], P['r1']) + 2 * max(P['w0'], P['w1'])
    if k == 'arc':
        return np.hypot(P['mu'], P['mv']) + max(P['r0'], P['r1']) + 2 * max(P['w0'], P['w1'])
    if k == 'line':
        return np.hypot(P['u'], P['v']) + P['length'] / 2 + 3 * P['width'] + abs(P.get('bow', 0))
    return 80.0
