"""Render a labeled gallery of simulated, continuously varying radar morphology.

Run ``python -m wxsim.radar.gallery``. The default PNG is written to the Seaboard
scenario output directory; ``--out`` selects a different path.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .colormaps import REF_LEVELS, reflectivity
from .field import RadarSim


BG = '#0b1220'
PANEL_BG = '#07101b'
SITE = {'lat': 35.5, 'lon': -80.5, 'max_range_km': 180.0}
WHEN = '2027-04-29T20:30Z'
BASE = {'prec': 0.0, 'sup': 0.9, 'hook': 0.78, 'elev': 0.0, 'hb': 0.0,
        'scale': 1.0, 'dz': 0.0, 'hail': 62.0, 'ffd_rot': 12.0, 'anvil': 1.0}


def _storm(distance_km, params, split='right'):
    """One fixed synthetic storm, placed east of the radar for a clean view."""
    import math

    lat = SITE['lat']
    lon = SITE['lon'] + distance_km / (111.32 * math.cos(math.radians(lat)))
    return {
        'id': 'MORPHOLOGY-GALLERY',
        'archetype': 'procedural',
        'split': split,
        'anchor': {'time': WHEN, 'lat': lat, 'lon': lon},
        'motion': {'dir_from': 225.0, 'speed_kt': 32.0},
        'params': BASE | params,
    }


def render_gallery(out_path):
    """Render continuum sweeps with the same site, scan, texture, and palette."""
    cmap, norm = reflectivity()
    sim = RadarSim(SITE, [], seed=41, texture_db=1.8, warp_km=0.6)
    step = 2
    edge_x, edge_y = sim.grid.edges_xy()
    panels = []

    def add_row(name, specs):
        panels.append((name, specs))

    precs = np.linspace(-1.0, 1.0, 7)
    for sup in np.linspace(0.0, 1.0, 5):
        add_row(f'PREC continuum  |  SUP {sup:.2f}', [
            (f'prec {prec:+.2f}', BASE | {'prec': float(prec), 'sup': float(sup)}, 35.0, 42.0)
            for prec in precs
        ])

    controls = np.linspace(0.0, 1.0, 7)
    for knob in ('line', 'bow'):
        fixed = {'line': 0.88} if knob == 'bow' else {}
        add_row(f'{knob.upper()} continuum', [
            (f'{knob} {v:.2f}', BASE | {'prec': 0.15, 'sup': 0.92} | fixed | {knob: float(v)}, 35.0, 42.0)
            for v in controls
        ])

    add_row('TORNADO signature', [
        (f'tor {v}', BASE | {'prec': 0.1, 'sup': 1.0, 'tor': float(v)}, 35.0, 42.0)
        for v in (0.0, 1.0)
    ])

    cycle_values = np.linspace(0.0, 1.0, 6)
    add_row('CYCLIC mesocyclone phase', [
        (f'cycle {v:.2f}', BASE | {'prec': 0.1, 'sup': 1.0, 'hook': 0.95, 'cycle': float(v)}, 35.0, 29.0)
        for v in cycle_values
    ])

    add_row('LOWTOP continuum  |  HP environment', [
        (f'lowtop {v:.2f}', BASE | {'prec': 0.72, 'sup': 0.9, 'lowtop': float(v)}, 35.0, 42.0)
        for v in controls
    ])

    beam_specs = []
    for distance, label in ((30.0, 'near'), (130.0, 'far')):
        for prec, name in ((-1.0, 'LP'), (0.0, 'classic'), (1.0, 'HP')):
            beam_specs.append((f'{name} / {label}', BASE | {'prec': prec, 'sup': 1.0, 'hail': 65.0}, distance, 42.0))
    add_row('BEAM height effect  |  LP / classic / HP', beam_specs)

    add_row('THREE-BODY hail spike  |  radar-site aware', [
        ('hail 74 dBZ', BASE | {'prec': 0.15, 'sup': 1.0, 'hook': 0.82, 'hail': 74.0}, 42.0, 46.0)
    ])

    nrows, ncols = len(panels), 7
    fig = plt.figure(figsize=(15.4, 2.10 * nrows + 1.25), facecolor=BG)
    gs = fig.add_gridspec(nrows, ncols, left=0.025, right=0.985, top=0.965, bottom=0.055,
                          wspace=0.045, hspace=0.24)
    fig.suptitle('SIMULATED WSR-88D BASE REFLECTIVITY  |  CONTINUOUS STORM MORPHOLOGY',
                 color='#e6edf7', fontsize=15, fontweight='bold', y=0.988)

    for ri, (row_name, specs) in enumerate(panels):
        offset = (ncols - len(specs)) // 2
        for ci, (caption, params, distance, radius) in enumerate(specs):
            ax = fig.add_subplot(gs[ri, offset + ci])
            ax.set_facecolor(PANEL_BG)
            storm = _storm(distance, params)
            sim.storms = [storm]
            dbz = sim.base(WHEN, elev=0.5, clutter=False)
            mask = np.ma.masked_invalid(dbz[::step, ::step])
            ax.pcolormesh(edge_x[::step, ::step], edge_y[::step, ::step], mask,
                          shading='flat', cmap=cmap, norm=norm, rasterized=True)
            sx, sy = sim.to_xy.transform(storm['anchor']['lon'], storm['anchor']['lat'])
            ax.set_xlim(sx - radius, sx + radius)
            ax.set_ylim(sy - radius, sy + radius)
            ax.set_aspect('equal', adjustable='box')
            ax.set_axis_off()
            title = f'{row_name}\n{caption}' if ci == 0 else caption
            ax.set_title(title, color='#d5deeb', fontsize=8, pad=2)

    cax = fig.add_axes([0.30, 0.025, 0.40, 0.010])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=REF_LEVELS[::2])
    cb.ax.tick_params(labelsize=7, colors='#cad5e2', length=2)
    cb.outline.set_edgecolor('#455468')
    cb.set_label('Reflectivity (dBZ)  |  0.5° base scan  |  synthetic returns only',
                 color='#cbd5e1', fontsize=8, labelpad=4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    default = Path('scenarios/seaboard_outbreak_2027/output/radar_morphology_gallery.png')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(default), help='output PNG path')
    args = parser.parse_args()
    print(render_gallery(args.out))


if __name__ == '__main__':
    main()
