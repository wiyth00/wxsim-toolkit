"""Storm archetypes: presets that assemble primitives into recognizable radar morphologies.

All presets are written for a RIGHT-moving storm in the storm-relative frame (u forward, v left);
left movers are mirrored by the caller. Every preset takes the same knobs:
    scale    size multiplier (1.0 = typical)
    dz       dBZ offset applied to every additive primitive (intensity; lifecycle adds to this)
    hail     peak dBZ of the hail core (None = archetype default)
    ffd_rot  extra rotation (deg, + = toward the left) of the forward flank / anvil
    anvil    anvil length multiplier (0 disables)
Primitive vertical extents z are km AGL; the sampler shows only what the beam intersects.
"""

import numpy as np


def _sc(prims, s):
    """Scale all horizontal dimensions."""
    keys = ('u', 'v', 'su', 'sv', 'length', 'w0', 'w1', 'mu', 'mv', 'r0', 'r1', 'width', 'bow')
    for P in prims:
        for k in keys:
            if k in P:
                P[k] = P[k] * s
        if 'bend' in P:
            P['bend'] = P['bend'] / s
    return prims


def classic_supercell(scale=1.0, dz=0.0, hail=None, ffd_rot=12, anvil=1.0, **_):
    hail = 69 if hail is None else hail
    p = [
        dict(kind='ellipse', u=0, v=0, su=4.2, sv=3.4, peak=61 + dz, edge=38, p=1.3, z=[0, 12]),
        dict(kind='ellipse', u=-0.6, v=-0.4, su=1.8, sv=1.5, peak=hail + dz, edge=58, p=1.6, z=[0, 11]),
        dict(kind='plume', u=2.5, v=0.8, rot=ffd_rot, length=24, w0=8, w1=14, p0=50 + dz, p1=24 + dz, bend=0.006, z=[0, 11]),
        dict(kind='hook', mu=-4.4, mv=-4.8, phi0=50, phi1=338, r0=6.2, r1=1.5, w0=2.6, w1=1.0, p0=58 + dz, p1=42 + dz, z=[0, 6]),
        dict(kind='ellipse', u=-6.5, v=-0.6, su=3.2, sv=2.4, peak=36 + dz, edge=18, z=[0, 7]),
        dict(kind='notch', u=-0.8, v=-7.4, su=3.4, sv=2.6, depth=36, z=[0, 5]),
        dict(kind='notch', u=-4.4, v=-4.6, su=1.6, sv=1.6, depth=18, z=[2.5, 8.5]),            # BWER (mid-levels)
        dict(kind='plume', u=7, v=2, rot=ffd_rot + 4, length=55 * anvil, w0=12, w1=32, p0=26 + dz, p1=12 + dz,
             bend=0.002, z=[7.5, 13]),
    ]
    return _sc(p, scale) if anvil else _sc(p[:-1], scale)


def lp_supercell(scale=1.0, dz=0.0, hail=None, ffd_rot=15, anvil=1.0, **_):
    """Small, intense hail core; weak, mostly-evaporating forward flank (virga) and a thin hook."""
    hail = 71 if hail is None else hail
    p = [
        dict(kind='ellipse', u=0, v=0, su=2.6, sv=2.1, peak=58 + dz, edge=36, p=1.2, z=[0, 12.5]),
        dict(kind='ellipse', u=-0.2, v=-0.2, su=1.3, sv=1.1, peak=hail + dz, edge=60, p=1.7, z=[0, 12]),
        dict(kind='plume', u=1.8, v=0.6, rot=ffd_rot, length=20, w0=5, w1=9, p0=40 + dz, p1=18 + dz, bend=0.004, z=[2.2, 11]),
        dict(kind='plume', u=1.5, v=0.3, rot=ffd_rot, length=9, w0=3.5, w1=4.5, p0=44 + dz, p1=30 + dz, z=[0, 11]),
        dict(kind='hook', mu=-2.8, mv=-2.6, phi0=60, phi1=290, r0=3.3, r1=1.3, w0=0.9, w1=0.5, p0=42 + dz, p1=30 + dz, z=[0, 5]),
        dict(kind='notch', u=-2.8, v=-2.6, su=1.2, sv=1.2, depth=16, z=[2.5, 9]),
        dict(kind='plume', u=5, v=1.5, rot=ffd_rot + 5, length=70 * anvil, w0=9, w1=30, p0=28 + dz, p1=14 + dz,
             bend=0.0015, z=[7, 13.5]),
    ]
    return _sc(p, scale) if anvil else _sc(p[:-1], scale)


def hp_supercell(scale=1.0, dz=0.0, hail=None, ffd_rot=10, anvil=1.0, **_):
    """Large kidney-bean echo; heavy precipitation wrapping the mesocyclone (hook buried in rain)."""
    hail = 67 if hail is None else hail
    p = [
        dict(kind='ellipse', u=0, v=0.5, su=7.5, sv=6.0, peak=60 + dz, edge=45, p=1.2, z=[0, 12]),
        dict(kind='ellipse', u=-1.2, v=-1.0, su=2.4, sv=2.0, peak=hail + dz, edge=60, z=[0, 11]),
        dict(kind='plume', u=4, v=1.5, rot=ffd_rot, length=26, w0=13, w1=18, p0=54 + dz, p1=30 + dz, bend=0.005, z=[0, 11]),
        dict(kind='hook', mu=-5.5, mv=-4.8, phi0=40, phi1=300, r0=6.5, r1=3.0, w0=3.6, w1=2.0, p0=58 + dz, p1=48 + dz, z=[0, 7]),
        dict(kind='notch', u=-2.5, v=-6.2, su=2.6, sv=2.0, depth=22, z=[0, 5]),
        dict(kind='notch', u=-5.5, v=-4.8, su=2.0, sv=2.0, depth=12, z=[2.5, 8]),
        dict(kind='plume', u=9, v=2, rot=ffd_rot + 3, length=50 * anvil, w0=16, w1=34, p0=28 + dz, p1=14 + dz, z=[7.5, 13]),
    ]
    return _sc(p, scale) if anvil else _sc(p[:-1], scale)


def elevated_hail(scale=1.0, dz=0.0, hail=None, ffd_rot=10, anvil=1.0, **_):
    """Storm rooted above a stable layer: compact core, hail reaching the ground, no hook/notch."""
    hail = 66 if hail is None else hail
    p = [
        dict(kind='ellipse', u=0, v=0, su=3.6, sv=2.8, peak=57 + dz, edge=38, p=1.3, z=[0, 11.5]),
        dict(kind='ellipse', u=0.3, v=0.2, su=1.5, sv=1.2, peak=hail + dz, edge=57, z=[0, 10.5]),
        dict(kind='plume', u=2.5, v=0.5, rot=ffd_rot, length=22, w0=8, w1=13, p0=45 + dz, p1=22 + dz, bend=0.004, z=[0, 10]),
        dict(kind='ellipse', u=-4.5, v=0, su=4.0, sv=3.0, peak=32 + dz, edge=15, z=[1.5, 8]),
        dict(kind='plume', u=6, v=1, rot=ffd_rot + 3, length=45 * anvil, w0=10, w1=26, p0=25 + dz, p1=12 + dz, z=[7, 12]),
    ]
    return _sc(p, scale) if anvil else _sc(p[:-1], scale)


def high_based_cell(scale=1.0, dz=0.0, hail=None, ffd_rot=8, anvil=1.0, **_):
    """Dryline / inverted-V convection: modest cores aloft, precipitation largely evaporating (virga)."""
    hail = 55 if hail is None else hail
    p = [
        dict(kind='ellipse', u=0, v=0, su=3.0, sv=2.4, peak=48 + dz, edge=28, p=1.2, z=[3.0, 11]),
        dict(kind='ellipse', u=0.4, v=0.3, su=1.4, sv=1.2, peak=hail + dz, edge=45, z=[1.0, 10]),
        dict(kind='ellipse', u=-2.5, v=-1.2, su=2.2, sv=1.8, peak=38 + dz, edge=20, z=[0, 6]),       # shaft reaching sfc
        dict(kind='plume', u=2, v=0.5, rot=ffd_rot, length=16, w0=6, w1=10, p0=34 + dz, p1=16 + dz, z=[3.2, 10]),
        dict(kind='plume', u=4, v=1, rot=ffd_rot, length=35 * anvil, w0=8, w1=20, p0=22 + dz, p1=10 + dz, z=[7.5, 12]),
    ]
    return _sc(p, scale) if anvil else _sc(p[:-1], scale)


def multicell(scale=1.0, dz=0.0, hail=None, ffd_rot=10, anvil=1.0, **_):
    """Cluster of cells at different stages: new cells on the right/forward flank, decaying cells to the rear-left."""
    hail = 60 if hail is None else hail
    cells = [(3, -3, 2.2, 55), (0.5, -0.5, 3.0, 57), (-2.5, 2.5, 3.2, 50), (-5, 5, 3.6, 42), (5.5, -5.5, 1.5, 44)]
    p = [dict(kind='ellipse', u=u, v=v, su=s, sv=s * 0.85, peak=pk + dz, edge=25, p=1.3, z=[0, 11]) for u, v, s, pk in cells]
    p.append(dict(kind='ellipse', u=0.5, v=-0.5, su=1.2, sv=1.0, peak=hail + dz, edge=52, z=[0, 10]))
    p.append(dict(kind='ellipse', u=-3, v=3, su=8, sv=6, peak=30 + dz, edge=14, z=[0, 9]))
    return _sc(p, scale)


def pulse(scale=1.0, dz=0.0, hail=None, ffd_rot=0, anvil=1.0, **_):
    hail = 58 if hail is None else hail
    p = [dict(kind='ellipse', u=0, v=0, su=3.0, sv=2.8, peak=52 + dz, edge=25, p=1.3, z=[0, 11]),
         dict(kind='ellipse', u=0, v=0, su=1.1, sv=1.0, peak=hail + dz, edge=50, z=[0, 10])]
    return _sc(p, scale)


def qlcs_segment(scale=1.0, dz=0.0, hail=None, ffd_rot=0, anvil=1.0, **_):
    """Convective line (length ~60 km) with trailing stratiform; bow pushes the middle forward."""
    p = [dict(kind='line', u=0, v=0, length=60, width=3.0, bow=6, peak=56 + dz, edge=40, z=[0, 11]),
         dict(kind='plume', u=-2, v=0, rot=180, length=30, w0=55, w1=45, p0=38 + dz, p1=22 + dz, z=[0, 8]),
         dict(kind='notch', u=-4, v=0, su=3, sv=18, depth=10, z=[0, 8])]
    return _sc(p, scale)


def stratiform(scale=1.0, dz=0.0, hail=None, ffd_rot=0, anvil=1.0, **_):
    p = [dict(kind='ellipse', u=0, v=0, su=45, sv=30, peak=32 + dz, edge=18, p=2.0, cutoff=1.35, z=[0, 8])]
    return _sc(p, scale)


ARCHETYPES = {
    'classic_supercell': classic_supercell, 'lp_supercell': lp_supercell, 'hp_supercell': hp_supercell,
    'elevated_hail': elevated_hail, 'high_based_cell': high_based_cell, 'multicell': multicell,
    'pulse': pulse, 'qlcs_segment': qlcs_segment, 'stratiform': stratiform,
}


# ----------------------------------------------------------------------------------------------------------
# Continuous morphology used by the environment-driven generator.
# Three anchor storms (LP, classic, HP) share the same roles; `prec` in [-1, 1] blends between them and the
# other knobs modulate supercell organization, hook/notch strength, elevated character and cloud-base height.
_ANCHORS = {
    'LP': dict(
        core=dict(kind='ellipse', u=0, v=0, su=2.6, sv=2.1, peak=58, edge=36, p=1.2, z=[0, 12.5]),
        hail=dict(kind='ellipse', u=-0.2, v=-0.2, su=1.3, sv=1.1, peak=0, edge=60, p=1.7, z=[0, 12]),
        ffd_sfc=dict(kind='plume', u=1.5, v=0.3, rot=0, length=9, w0=3.5, w1=4.5, p0=44, p1=30, bend=0.0, z=[0, 11]),
        ffd_aloft=dict(kind='plume', u=1.8, v=0.6, rot=0, length=20, w0=5, w1=9, p0=40, p1=18, bend=0.004, z=[2.2, 11]),
        hook=dict(kind='hook', mu=-2.8, mv=-2.6, phi0=60, phi1=290, r0=3.3, r1=1.3, w0=0.9, w1=0.5, p0=42, p1=30, z=[0, 5]),
        rfd=dict(kind='ellipse', u=-4.0, v=-1.0, su=2.0, sv=1.5, peak=22, edge=12, p=1.6, z=[0, 7]),
        notch=dict(kind='notch', u=-0.5, v=-3.5, su=1.8, sv=1.4, depth=10, z=[0, 5]),
        bwer=dict(kind='notch', u=-2.8, v=-2.6, su=1.2, sv=1.2, depth=16, z=[2.5, 9]),
        anvil=dict(kind='plume', u=5, v=1.5, rot=5, length=70, w0=9, w1=30, p0=28, p1=14, bend=0.0015, z=[7, 13.5])),
    'CL': dict(
        core=dict(kind='ellipse', u=0, v=0, su=4.2, sv=3.4, peak=61, edge=38, p=1.3, z=[0, 12]),
        hail=dict(kind='ellipse', u=-0.6, v=-0.4, su=1.8, sv=1.5, peak=0, edge=58, p=1.6, z=[0, 11]),
        ffd_sfc=dict(kind='plume', u=2.5, v=0.8, rot=0, length=24, w0=8, w1=14, p0=50, p1=24, bend=0.006, z=[0, 11]),
        ffd_aloft=dict(kind='plume', u=2.5, v=0.8, rot=0, length=24, w0=8, w1=14, p0=30, p1=18, bend=0.006, z=[3, 11]),
        hook=dict(kind='hook', mu=-4.4, mv=-4.8, phi0=50, phi1=338, r0=6.2, r1=1.5, w0=2.6, w1=1.0, p0=58, p1=42, z=[0, 6]),
        rfd=dict(kind='ellipse', u=-6.5, v=-0.6, su=3.2, sv=2.4, peak=36, edge=18, p=1.6, z=[0, 7]),
        notch=dict(kind='notch', u=-0.8, v=-7.4, su=3.4, sv=2.6, depth=36, z=[0, 5]),
        bwer=dict(kind='notch', u=-4.4, v=-4.6, su=1.6, sv=1.6, depth=18, z=[2.5, 8.5]),
        anvil=dict(kind='plume', u=7, v=2, rot=4, length=55, w0=12, w1=32, p0=26, p1=12, bend=0.002, z=[7.5, 13])),
    'HP': dict(
        core=dict(kind='ellipse', u=0, v=0.5, su=7.5, sv=6.0, peak=60, edge=45, p=1.2, z=[0, 12]),
        hail=dict(kind='ellipse', u=-1.2, v=-1.0, su=2.4, sv=2.0, peak=0, edge=60, p=1.6, z=[0, 11]),
        ffd_sfc=dict(kind='plume', u=4, v=1.5, rot=0, length=26, w0=13, w1=18, p0=54, p1=30, bend=0.005, z=[0, 11]),
        ffd_aloft=dict(kind='plume', u=4, v=1.5, rot=0, length=26, w0=13, w1=18, p0=32, p1=18, bend=0.005, z=[3, 11]),
        hook=dict(kind='hook', mu=-5.5, mv=-4.8, phi0=40, phi1=300, r0=6.5, r1=3.0, w0=3.6, w1=2.0, p0=58, p1=48, z=[0, 7]),
        rfd=dict(kind='ellipse', u=-7.0, v=-2.0, su=4.0, sv=3.0, peak=48, edge=25, p=1.6, z=[0, 7]),
        notch=dict(kind='notch', u=-2.5, v=-6.2, su=2.6, sv=2.0, depth=22, z=[0, 5]),
        bwer=dict(kind='notch', u=-5.5, v=-4.8, su=2.0, sv=2.0, depth=12, z=[2.5, 8]),
        anvil=dict(kind='plume', u=9, v=2, rot=3, length=50, w0=16, w1=34, p0=28, p1=14, bend=0.0, z=[7.5, 13])),
}


def _lerp_prim(a, b, w):
    out = {}
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, (int, float)) and not isinstance(va, bool):
            out[k] = va + (vb - va) * w
        elif isinstance(va, list):
            out[k] = [x + (y - x) * w for x, y in zip(va, vb)]
        else:
            out[k] = va
    return out


def procedural(prec=0.0, sup=1.0, hook=0.7, elev=0.0, hb=0.0, scale=1.0, dz=0.0, hail=65.0, ffd_rot=12, anvil=1.0,
               line=0.0, bow=0.0, tor=0.0, cycle=0.0, lowtop=0.0, split_core=0.0, merge=0.0, merge_fade=0.0, **_):
    """Continuous storm: prec -1 (LP) .. 0 (classic) .. +1 (HP); sup 0 (ordinary cell) .. 1 (mature supercell);
    hook 0..1 (RFD/hook + inflow-notch prominence); elev 0..1 (elevated: no hook/notch); hb 0..1 (high-based:
    precipitation mostly evaporating before reaching the low levels).  ``line``, ``bow``, ``tor``, ``cycle`` and
    ``lowtop`` are optional 0..1 morphology controls.  They are appended after the legacy roles, so old storms with
    their default values retain the same primitive list and reflectivity path."""
    prec = max(-1.0, min(1.0, prec)); sup = max(0.0, min(1.0, sup))
    hook = max(0.0, min(1.0, hook)); elev = max(0.0, min(1.0, elev)); hb = max(0.0, min(1.0, hb))
    line = max(0.0, min(1.0, line)); bow = max(0.0, min(1.0, bow)); tor = max(0.0, min(1.0, tor))
    cycle = max(0.0, min(1.0, cycle)); lowtop = max(0.0, min(1.0, lowtop))
    split_core = max(0.0, min(1.0, split_core)); merge = max(0.0, min(1.0, merge)); merge_fade = max(0.0, min(1.0, merge_fade))
    a, b, w = (('LP', 'CL', prec + 1) if prec < 0 else ('CL', 'HP', prec))
    R = {k: _lerp_prim(_ANCHORS[a][k], _ANCHORS[b][k], w) for k in _ANCHORS['CL']}
    org = sup * hook * (1 - elev)
    # organization: weaker/rounder cells when sup is low
    for k in ('core',):
        R[k]['su'] *= 0.75 + 0.25 * sup
        R[k]['sv'] *= 0.85 + 0.15 * sup
    for k in ('ffd_sfc', 'ffd_aloft'):
        R[k]['length'] *= 0.35 + 0.65 * sup
        R[k]['rot'] = ffd_rot
    R['anvil']['length'] *= (0.45 + 0.55 * sup) * anvil
    R['anvil']['rot'] += ffd_rot
    R['hook']['p0'] -= (1 - org) * 42; R['hook']['p1'] -= (1 - org) * 38
    R['hook']['w0'] *= 0.6 + 0.4 * org
    R['notch']['depth'] *= org
    R['bwer']['depth'] *= sup * (1 - 0.5 * elev)
    R['rfd']['peak'] -= elev * 14 + (1 - sup) * 22
    # high-based: raise the bottom of everything except the hail shaft
    for k in ('core', 'ffd_sfc', 'hook', 'rfd'):
        R[k]['z'] = [R[k]['z'][0] + 3.0 * hb, R[k]['z'][1]]
    R['hail']['z'] = [R['hail']['z'][0] + 1.0 * hb, R['hail']['z'][1]]
    R['hail']['peak'] = hail - 8 * hb

    # Low-topped storms are physically smaller as well as lower.  Apply this
    # before any of the optional additions so all their dimensions still pass
    # through _sc() once at the end.
    if lowtop > 0:
        small = 1.0 - 0.22 * lowtop
        for P in R.values():
            for f in ('u', 'v', 'su', 'sv', 'length', 'w0', 'w1', 'mu', 'mv', 'r0', 'r1'):
                if f in P:
                    P[f] *= small
            if 'z' in P:
                P['z'] = [P['z'][0], P['z'][0] + (P['z'][1] - P['z'][0]) * (1.0 - 0.35 * lowtop)]
        R['anvil']['length'] *= 1.0 - 0.55 * lowtop
        R['anvil']['p0'] -= 5.0 * lowtop
        R['anvil']['p1'] -= 3.0 * lowtop

    # Optional continuum roles.  The base list below deliberately remains
    # unchanged when all five controls are zero for byte-compatible legacy
    # storms.  Weak additions start near the noise floor and strengthen
    # smoothly as their control rises.
    extras = []
    if line > 0:
        R['core']['su'] *= 1.0 + 0.10 * line
        R['core']['sv'] *= 1.0 + 0.65 * line
        R['hook']['w0'] *= 1.0 - 0.20 * line
        R['hook']['w1'] *= 1.0 - 0.15 * line
        line_len = 15.0 + 57.0 * line
        extras.append(dict(kind='line', u=0.7, v=0, length=line_len, width=1.6 + 2.0 * line,
                           bow=0.4 * line, peak=31.0 + 24.0 * line + dz, edge=12.0 + 24.0 * line,
                           p=1.45, cutoff=2.0, z=[0, 10.5]))
        # A rear stratiform/transition plume merges the forward-flank role into
        # the line instead of swapping to qlcs_segment at a threshold.
        extras.append(dict(kind='plume', u=-2.5, v=0, rot=180, length=13.0 + 30.0 * line,
                           w0=11.0 + 22.0 * line, w1=9.0 + 24.0 * line,
                           p0=22.0 + 17.0 * line + dz, p1=10.0 + 12.0 * line + dz,
                           edge=8.0 + 10.0 * line, bend=0.0, z=[0, 8.5]))

    if bow > 0:
        # Bow can be used on its own in a developing line, but is naturally
        # strongest once the line element has grown.
        bow_len = 15.0 + 57.0 * max(line, 0.18 * bow)
        if line <= 0:
            extras.append(dict(kind='line', u=0.7, v=0, length=bow_len, width=1.7 + 1.1 * bow,
                               bow=0.7 * bow, peak=24.0 + 22.0 * bow + dz, edge=10.0 + 18.0 * bow,
                               p=1.5, cutoff=2.0, z=[0, 9.5]))
        else:
            # The existing line is the first optional primitive.
            extras[0]['bow'] = 0.7 * bow
        # Rear-inflow notch and a weak-echo channel behind the bow apex.
        extras.append(dict(kind='notch', u=-2.5 - 1.2 * bow, v=0, su=2.0 + 1.2 * bow,
                           sv=3.0 + 3.5 * bow, depth=(7.0 + 18.0 * bow) * max(line, 0.3), z=[0, 7]))
        # Bookend vortex comma at the left/north end of the line.
        extras.append(dict(kind='arc', mu=-1.0, mv=0.42 * bow_len, phi0=155, phi1=325,
                           r0=5.0 + 2.0 * bow, r1=2.2 + 0.8 * bow, w0=1.3 + 1.1 * bow,
                           w1=0.8 + 0.5 * bow, p0=20.0 + 21.0 * bow + dz,
                           p1=14.0 + 14.0 * bow + dz, edge=8.0, p=1.5, z=[0, 7]))

    if tor > 0:
        # Tornadic circulation tightens the existing hook and sharpens its
        # inflow notch, while the debris ball remains a low-level-only return.
        R['hook']['w0'] *= 1.0 - 0.35 * tor
        R['hook']['w1'] *= 1.0 - 0.30 * tor
        R['hook']['phi1'] += 20.0 * tor
        R['notch']['depth'] *= 1.0 + 0.85 * tor
        phi = np.deg2rad(R['hook']['phi1'])
        tip_u = R['hook']['mu'] + R['hook']['r1'] * np.cos(phi)
        tip_v = R['hook']['mv'] + R['hook']['r1'] * np.sin(phi)
        extras.append(dict(kind='ellipse', u=tip_u, v=tip_v, su=0.55 + 0.30 * tor,
                           sv=0.45 + 0.22 * tor, peak=-20.0 + 88.0 * tor + dz,
                           edge=-24.0, p=1.25, z=[0, 1.5]))

    if cycle > 0:
        # A full cycle returns to the original geometry.  The cosine term is
        # zero at both endpoints and peaks at occlusion (cycle=0.5).
        phase = 2.0 * np.pi * cycle
        cycle_strength = float(np.clip((sup - 0.35) / 0.65, 0.0, 1.0))
        occlusion = 0.5 * (1.0 - np.cos(phase)) * cycle_strength
        shift = np.sin(np.pi * cycle) * cycle_strength
        hook_p0, hook_p1 = R['hook']['p0'], R['hook']['p1']
        R['hook']['phi0'] += 12.0 * shift
        R['hook']['phi1'] -= 58.0 * occlusion
        R['hook']['p0'] -= 18.0 * occlusion
        R['hook']['p1'] -= 26.0 * occlusion
        R['hook']['w1'] *= 1.0 + 0.25 * occlusion
        R['notch']['depth'] *= 1.0 + 0.65 * occlusion
        R['rfd']['u'] -= 5.0 * occlusion
        R['rfd']['peak'] += 18.0 * occlusion

        # As the original hook occludes, a tighter new hook grows farther down
        # the rear flank.  Its linear-Z amplitude follows a smooth periodic
        # envelope that vanishes at phases 0 and 1, so the cycle closes without
        # a geometry or reflectivity jump.
        new_weight = shift ** 2
        if new_weight > 1e-5:
            new_hook = dict(R['hook'])
            new_hook['mu'] -= 12.0 * shift
            new_hook['mv'] -= 5.0 * shift
            new_hook['phi0'] += 24.0 * shift
            new_hook['phi1'] += 36.0 * shift
            new_hook['w0'] *= 0.82
            new_hook['w1'] *= 0.72
            attenuation = 20.0 * np.log10(new_weight)
            new_hook['p0'] = hook_p0 + attenuation + dz
            new_hook['p1'] = hook_p1 + attenuation + dz
            extras.append(new_hook)

    if split_core > 0:
        # Both members of a split carry this transient shared core.  The caller
        # mirrors the left member, turning the small offset into a paired echo
        # while the parent retains a common central bridge.
        R['core']['su'] *= 1.0 + 0.18 * split_core
        R['core']['sv'] *= 1.0 + 0.12 * split_core
        extras.append(dict(kind='ellipse', u=-0.8, v=-1.7, su=1.55 + 0.65 * split_core,
                           sv=1.25 + 0.45 * split_core, peak=28.0 + 24.0 * split_core + dz,
                           edge=12.0, p=1.35, z=[0, 10.5]))

    if merge > 0:
        # The surviving storm briefly broadens and absorbs a low-level remnant.
        R['core']['su'] *= 1.0 + 0.28 * merge
        R['core']['sv'] *= 1.0 + 0.20 * merge
        R['hail']['su'] *= 1.0 + 0.16 * merge
        extras.append(dict(kind='ellipse', u=-1.8, v=0.4, su=1.6 + 1.7 * merge,
                           sv=1.3 + 1.2 * merge, peak=16.0 + 31.0 * merge + dz,
                           edge=10.0, p=1.45, z=[0, 9.5]))

    if merge_fade > 0:
        fade = 1.0 - 0.72 * merge_fade
        for P in R.values():
            for f in ('peak', 'p0', 'p1'):
                if f in P and P['kind'] != 'notch':
                    P[f] *= fade

    for k, P in R.items():
        for f in ('peak', 'p0', 'p1'):
            if f in P and P['kind'] != 'notch':
                P[f] = P[f] + dz
    prims = [R[k] for k in ('core', 'hail', 'ffd_sfc', 'ffd_aloft', 'hook', 'rfd', 'notch', 'bwer')]
    if anvil:
        prims.append(R['anvil'])
    prims.extend(extras)
    return _sc(prims, scale)


ARCHETYPES['procedural'] = procedural
