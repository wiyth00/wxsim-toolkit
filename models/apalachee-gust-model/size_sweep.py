"""Verify and compare original, 2x, 2.5x, and 3x fictional storm controls.

Run after computing each control: .venv/bin/python size_sweep.py
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['APALACHEE_SKIP_CACHE'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import stormmodel as original
from plotkit import ACCENT, COLORS, FILL_LEVELS, INK, MUTED, draw_cbar
from station_catalog import ST

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'size_comparison'
CASES = [(1., ROOT), (2., ROOT / 'double_size'),
         (2.5, ROOT / 'size_2p5x'), (3., ROOT / 'size_3x')]


def load_model(scale, directory):
    if scale == 1:
        return original
    name = f'storm_size_{str(scale).replace(".", "p")}'
    spec = importlib.util.spec_from_file_location(name, directory / 'stormmodel.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect(scale, directory):
    model = load_model(scale, directory)
    np.testing.assert_array_equal(model.WP, original.WP)
    np.testing.assert_array_equal(model.LLJ, original.LLJ)
    np.testing.assert_allclose(model.RMAX[:, 1], scale * original.RMAX[:, 1])
    np.testing.assert_allclose(model.RADII, scale * original.RADII)
    np.testing.assert_array_equal(model.X, original.X)
    np.testing.assert_array_equal(model.Y, original.Y)
    assert model.STING == original.STING and model.CONVECTION == original.CONVECTION
    for path in (model.PEAK, model.FRAMES, directory / 'station_series.npz'):
        assert model.cache_valid(path), f'Stale cache: {path}'
    with np.load(model.PEAK) as data:
        peak, times = data['G'], data['T']
    with np.load(model.FRAMES) as data:
        frames, hours = data['F'].astype(np.float32), data['ft']
    with np.load(directory / 'station_series.npz') as data:
        station_peaks = data['gust_mph'].max(axis=0)
        np.testing.assert_allclose(data['hours'], np.arange(12, 114.05, .1), atol=1e-5)
    assert np.isfinite(peak).all() and (peak >= 5).all()
    assert ((times >= model.START) & (times <= model.END)).all()
    np.testing.assert_allclose(hours, np.arange(30, 112.1, .5))
    assert float((frames.max(axis=0) - peak).max()) < .07
    repeated = model.inst_gust(54)
    model.inst_gust(78)
    np.testing.assert_array_equal(repeated, model.inst_gust(54))
    cell_km2 = (model.DX / 1000) ** 2
    snapshot_58 = (frames >= 58).sum(axis=(1, 2)) * cell_km2
    metrics = dict(radius_scale=scale, geometric_area_scale=scale ** 2,
        grid_max_mph=round(float(peak.max()), 2),
        land_max_mph=round(float(peak[model.land].max()), 2),
        event_footprint_km2={str(k): int((peak >= k).sum() * cell_km2)
                             for k in (40, 58, 74, 100)},
        land_event_footprint_km2={str(k): int(((peak >= k) & model.land).sum() * cell_km2)
                                  for k in (58, 74)},
        largest_snapshot_58_km2=int(snapshot_58.max()),
        largest_snapshot_58_hour=float(hours[snapshot_58.argmax()]),
        landfall_34kt_low_level_radii_km_NE_SE_SW_NW=model.quadrant_radii(54)[0].round(2).tolist(),
        model_signature=model.signature(), simulated=True)
    return metrics, peak, station_peaks


def run():
    OUT.mkdir(exist_ok=True)
    results = [collect(scale, directory) for scale, directory in CASES]
    summaries = [r[0] for r in results]
    for summary in summaries:
        summary['snapshot_58_area_ratio_to_original'] = round(
            summary['largest_snapshot_58_km2'] / summaries[0]['largest_snapshot_58_km2'], 3)
    summary = dict(simulated=True, scenarios=summaries,
        fixed='Track, timing, pressure, prescribed jet speeds, local sting/downburst scales, geography, surface mixing and random seeds.',
        area_definition='Projected 3 km grid areas within the original Southeast map. Event footprint is the union across six-minute samples; largest snapshot uses stored half-hour samples.',
        limitations='Larger storms extend outside the map. Single deterministic parametric controls; no balanced atmospheric dynamics, new ensemble, surge or precipitation calculations.')
    (OUT / 'size_sweep.json').write_text(json.dumps(summary, indent=2) + '\n')
    station_rows = []
    for j, (sid, name, lat, lon) in enumerate(ST):
        row = dict(id=sid, name=name, simulated=True)
        row.update({f'grid_peak_{scale:g}x_mph': round(float(result[2][j]), 2)
                    for (scale, _), result in zip(CASES, results)})
        station_rows.append(row)
    with (OUT / 'station_size_sweep.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(station_rows[0]))
        writer.writeheader(); writer.writerows(station_rows)

    fig = plt.figure(figsize=(13, 15), dpi=160, facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.04, right=.98, bottom=.16, top=.88,
                            hspace=.22, wspace=.06)
    track = original.track_xy(np.arange(30, 113))
    cities = [('Atlanta', -84.39, 33.75), ('Mobile', -88.04, 30.69),
              ('Tampa', -82.46, 27.95), ('Charleston', -79.93, 32.78)]
    for index, ((scale, _), (stats, peak, _)) in enumerate(zip(CASES, results)):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        original.m.drawmapboundary(fill_color='#e6ecf1', linewidth=.5, ax=ax)
        original.m.fillcontinents(color='#f5f3ee', lake_color='#e6ecf1', ax=ax)
        ax.contourf(original.X, original.Y, peak, levels=FILL_LEVELS, colors=COLORS, alpha=.9)
        original.m.drawstates(linewidth=.5, color='#4d4d55', ax=ax)
        original.m.drawcoastlines(linewidth=.6, color='#3a3a42', ax=ax)
        ax.contour(original.X, original.Y, peak, levels=[58], colors=[INK], linewidths=.8)
        ax.contour(original.X, original.Y, peak, levels=[74], colors=[INK], linewidths=.8, linestyles='dotted')
        ax.plot(track[:, 0], track[:, 1], color=INK, lw=1.)
        ax.plot(*original.center_xy(54), marker='*', ms=10, mfc='#ffd23f', mec=INK)
        for name, lon, lat in cities:
            x, y = original.m(lon, lat)
            ax.plot(x, y, 'o', ms=2.5, color=INK)
            ax.annotate(name, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=7,
                        bbox=dict(fc='white', ec='none', alpha=.75, pad=.7))
        ax.set_title('Original · 1× radius' if scale == 1 else f'{scale:g}× original radius',
                     fontsize=15, weight='bold', color=INK, pad=12)
        ax.text(.5, -.045,
            f"Peak {stats['grid_max_mph']:.1f} mph · land peak {stats['land_max_mph']:.1f} mph\n"
            f"58+ mph event footprint: {stats['event_footprint_km2']['58']:,} km²",
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color=INK, linespacing=1.5)
    fig.text(.04, .956, 'How big can this storm get?', fontsize=25, weight='bold', color=INK)
    fig.text(.04, .925, 'Apalachee Bay sou’easter · Dec 15–19, 2026 · same track, pressure and prescribed jet speeds',
             fontsize=10, color=MUTED)
    fig.text(.98, .979, 'SIMULATED SCENARIO', ha='right', fontsize=9, weight='bold', color=ACCENT)
    fig.text(.04, .108, 'Maximum gust at each point (mph)', fontsize=9, color=INK)
    draw_cbar(fig, [.04, .081, .43, .014])
    fig.text(.52, .09, 'Solid contour: 58 mph gust · Dotted: 74 mph gust\nYellow star: landfall near St. Marks',
             fontsize=9, color=INK, linespacing=1.5)
    fig.text(.04, .026, 'All panels use the original map domain; the larger storms extend beyond it. Areas are limited to this map.\n'
             'Cyclone length scales change; local sting jets, downbursts, geography and mixing retain their original settings. Fictional, uncalibrated controls.',
             fontsize=8, color=MUTED, linespacing=1.6)
    fig.savefig(OUT / 'size_sweep.png', facecolor='white')
    print(json.dumps(summary, indent=2))
    for row in station_rows:
        if row['id'] in ('KMOB', 'KTPA', 'KCHS', 'KATL', 'KTLH', 'KVLD', 'KBHM'):
            print(row)
    print('All four controls passed radius, input, provenance, time-step, frame/peak and repeatability checks.')


if __name__ == '__main__':
    run()
