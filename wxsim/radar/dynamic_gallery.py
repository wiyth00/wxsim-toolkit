"""Render a labeled gallery of the evolving, site-independent dynamic radar storm.

Run ``python -m wxsim.radar.dynamic_gallery``.  The default PNG is written to
the Seaboard scenario output directory; ``--out`` selects a different path.
Every panel is explicitly simulated reflectivity, not an observation.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import dynamic
from .colormaps import REF_LEVELS, reflectivity


BG = '#0b1220'
PANEL_BG = '#07101b'
SEED = 41
STORM_ID = 'DYNAMIC-GALLERY'
AGES_MIN = (0.0, 20.0, 60.0, 110.0, 170.0, 230.0)
HEIGHTS_KM = (0.5, 1.5, 3.5, 5.5)
SAMPLE_AGE_MIN = 110.0
BASE_PARAMS = dict(getattr(dynamic, 'DEFAULT_PARAMS', getattr(dynamic, 'DEFAULTS', {})))

# One smoothly interpolated environment and morphology trajectory.  The dynamic
# engine remains the only field model; these values are its changing state.
_FORCING_AGES = np.asarray((0.0, 20.0, 60.0, 110.0, 170.0, 230.0), dtype=float)
_FORCING = {
    'mlcape': np.asarray((550.0, 1050.0, 1850.0, 2250.0, 1450.0, 450.0)),
    'shear_kt': np.asarray((20.0, 30.0, 43.0, 55.0, 43.0, 25.0)),
    'rh_low': np.asarray((44.0, 55.0, 68.0, 78.0, 66.0, 46.0)),
    'srh01': np.asarray((45.0, 85.0, 150.0, 235.0, 150.0, 40.0)),
    'prec': np.asarray((0.16, 0.30, 0.56, 0.78, 0.68, 0.28)),
    'sup': np.asarray((0.05, 0.22, 0.62, 0.94, 0.67, 0.16)),
    'hook': np.asarray((0.03, 0.12, 0.52, 0.86, 0.58, 0.10)),
    'line': np.asarray((0.00, 0.04, 0.24, 0.78, 1.00, 0.52)),
    'bow': np.asarray((0.00, 0.00, 0.05, 0.42, 0.82, 0.30)),
    'hail': np.asarray((48.0, 52.0, 60.0, 68.0, 61.0, 45.0)),
}
_STAGES = ('INITIATION', 'ORGANIZING', 'MATURE', 'UPSCALE GROWTH', 'DECAY', 'DISSIPATING')


def _forcing(age_min):
    """Interpolate the state trajectory with smoothstep transitions."""
    age = float(age_min)
    params = dict(BASE_PARAMS)
    for key, values in _FORCING.items():
        i = int(np.clip(np.searchsorted(_FORCING_AGES, age, side='right') - 1, 0, len(_FORCING_AGES) - 2))
        t = np.clip((age - _FORCING_AGES[i]) / (_FORCING_AGES[i + 1] - _FORCING_AGES[i]), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        params[key] = float(values[i] + (values[i + 1] - values[i]) * t)
    return params


def _forcing_label(params):
    return (f"CAPE {params['mlcape']:.0f} | shear {params['shear_kt']:.0f} kt\n"
            f"RH {params['rh_low']:.0f}% | SRH {params['srh01']:.0f} | "
            f"line/bow {params['line']:.2f}/{params['bow']:.2f}")


def _grid():
    """Build one storm-relative grid large enough for every age in the gallery."""
    reach = max(float(dynamic.reach_km(age, _forcing(age))) for age in AGES_MIN)
    half = max(40.0, reach + 3.0)
    resolution = 1.25
    axis = np.arange(-half, half + resolution, resolution)
    return np.meshgrid(axis, axis), half


def _field(U, V, z_km, age_min, params):
    return np.asarray(dynamic.reflectivity(
        U, V, float(z_km), float(age_min), params,
        seed=SEED, storm_id=STORM_ID,
    ), dtype=float)


def _draw(ax, field, half, cmap, norm, title):
    masked = np.ma.masked_less(field, 5)
    ax.set_facecolor(PANEL_BG)
    ax.imshow(masked, origin='lower', extent=(-half, half, -half, half),
              cmap=cmap, norm=norm, interpolation='nearest', rasterized=True)
    ax.axhline(0, color='#9fb0c1', lw=0.35, alpha=0.22)
    ax.axvline(0, color='#9fb0c1', lw=0.35, alpha=0.22)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(title, color='#d5deeb', fontsize=8, pad=3)


def render_gallery(out_path):
    """Render the dynamic age sequence and a selected-age height/composite view."""
    cmap, norm = reflectivity()
    (U, V), half = _grid()
    ages = list(AGES_MIN)
    sequence = [(age, _forcing(age), _field(U, V, 0.5, age, _forcing(age))) for age in ages]
    sample_params = _forcing(SAMPLE_AGE_MIN)
    height_fields = [_field(U, V, z, SAMPLE_AGE_MIN, sample_params) for z in HEIGHTS_KM]
    composite = np.max(np.stack(height_fields), axis=0)

    fig = plt.figure(figsize=(16.2, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 6, left=0.025, right=0.985, top=0.79, bottom=0.105,
                          wspace=0.045, hspace=0.28)
    fig.suptitle('SIMULATED SITE-INDEPENDENT DYNAMIC REFLECTIVITY',
                 color='#e6edf7', fontsize=15, fontweight='bold', y=0.985)
    fig.text(0.5, 0.952, 'Same seeded dynamic storm  |  smooth environmental forcing  |  storm-relative km  |  synthetic returns only',
             color='#9fb3c8', fontsize=9, ha='center')

    for i, (age, params, field) in enumerate(sequence):
        title = (f'{_STAGES[i]}\nAGE {age:02.0f} MIN\n'
                 f'CAPE {params["mlcape"]:.0f} | SH {params["shear_kt"]:.0f} kt | RH {params["rh_low"]:.0f}% | '
                 f'SRH {params["srh01"]:.0f}\nL/B {params["line"]:.2f}/{params["bow"]:.2f}')
        _draw(fig.add_subplot(gs[0, i]), field, half, cmap, norm, title)

    sample_label = _forcing_label(sample_params)
    _draw(fig.add_subplot(gs[1, 0]), sequence[ages.index(SAMPLE_AGE_MIN)][2], half, cmap, norm,
          f'BASE REFLECTIVITY SAMPLE\nAGE {SAMPLE_AGE_MIN:.0f} MIN | 0.5 km\n{sample_label}')
    for i, (z, field) in enumerate(zip(HEIGHTS_KM, height_fields), start=1):
        _draw(fig.add_subplot(gs[1, i]), field, half, cmap, norm,
              f'HEIGHT {z:.1f} km\nAGE {SAMPLE_AGE_MIN:.0f} MIN')
    _draw(fig.add_subplot(gs[1, 5]), composite, half, cmap, norm,
          f'HEIGHT COMPOSITE\nAGE {SAMPLE_AGE_MIN:.0f} MIN')

    cax = fig.add_axes([0.30, 0.045, 0.40, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=REF_LEVELS[::2])
    cb.ax.tick_params(labelsize=7, colors='#cad5e2', length=2)
    cb.outline.set_edgecolor('#455468')
    cb.set_label('Reflectivity (dBZ)  |  dynamic engine  |  explicitly simulated',
                 color='#cbd5e1', fontsize=8, labelpad=4)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=125, facecolor=BG)
    plt.close(fig)
    return out_path


def main():
    default = Path('scenarios/seaboard_outbreak_2027/output/dynamic_storm_gallery.png')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(default), help='output PNG path')
    args = parser.parse_args()
    print(render_gallery(args.out))


if __name__ == '__main__':
    main()
