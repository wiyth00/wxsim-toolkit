"""Compare the preserved v3 control with the doubled cyclone-size experiment."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import stormmodel as sm
from plotkit import ACCENT, COLORS, FILL_LEVELS, INK, MUTED, draw_cbar
from station_catalog import ST

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['APALACHEE_SKIP_CACHE'] = '1'
spec = importlib.util.spec_from_file_location('original_stormmodel', ROOT / 'stormmodel.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)

np.testing.assert_array_equal(sm.WP, original.WP)
np.testing.assert_array_equal(sm.LLJ, original.LLJ)
np.testing.assert_array_equal(sm.RMAX[:, 1], 2 * original.RMAX[:, 1])
np.testing.assert_array_equal(sm.RADII, 2 * original.RADII)
assert sm.STING == original.STING and sm.CONVECTION == original.CONVECTION
assert sm.cache_valid(sm.PEAK) and sm.cache_valid(sm.FRAMES)
assert original.cache_valid(original.PEAK) and original.cache_valid(original.FRAMES)

with np.load(ROOT / 'peak.npz') as d:
    old_peak = d['G']
with np.load(ROOT / 'station_series.npz') as d:
    old_stations = d['gust_mph'].max(axis=0)
with np.load(HERE / 'station_series.npz') as d:
    new_stations = d['gust_mph'].max(axis=0)
with np.load(ROOT / 'frames.npz') as d:
    old_frames, hours = d['F'].astype(np.float32), d['ft']
with np.load(sm.FRAMES) as d:
    new_frames = d['F'].astype(np.float32)
    np.testing.assert_array_equal(d['ft'], hours)
assert np.isfinite(sm.G).all() and (sm.G >= 5).all()
assert float((new_frames.max(axis=0) - sm.G).max()) < .07
repeat = sm.inst_gust(54)
sm.inst_gust(78)
np.testing.assert_array_equal(repeat, sm.inst_gust(54))

cell_km2 = (sm.DX / 1000) ** 2
def metrics(peak, frames):
    return dict(grid_max_mph=round(float(peak.max()), 2),
                land_max_mph=round(float(peak[sm.land].max()), 2),
                event_footprint_km2={str(k): int((peak >= k).sum() * cell_km2)
                                     for k in (40, 58, 74, 100)},
                land_event_footprint_km2={str(k): int(((peak >= k) & sm.land).sum() * cell_km2)
                                          for k in (58, 74)},
                largest_snapshot_58_km2=int((frames >= 58).sum(axis=(1, 2)).max() * cell_km2))

summary = dict(simulated=True, experiment='2x cyclone length scales; same intensity inputs',
               original=metrics(old_peak, old_frames), doubled=metrics(sm.G, new_frames),
               limitations='Areas are projected grid areas within the original Southeast domain; the larger storm extends beyond it. Local sting/downburst scales, terrain, wedge, and coastal gradient masks are retained. Single deterministic control; no new ensemble.',
               baseline_signature=original.signature(), experiment_signature=sm.signature())
summary['landfall_34kt_low_level_radii_km_NE_SE_SW_NW'] = dict(
    original=original.quadrant_radii(54)[0].round(2).tolist(),
    doubled=sm.quadrant_radii(54)[0].round(2).tolist())
(HERE / 'size_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
rows = [dict(id=sid, name=name, original_grid_peak_mph=round(float(old_stations[j]), 2),
             doubled_grid_peak_mph=round(float(new_stations[j]), 2),
             change_mph=round(float(new_stations[j] - old_stations[j]), 2), simulated=True)
        for j, (sid, name, lat, lon) in enumerate(ST)]
with (HERE / 'station_size_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 8.8), dpi=150)
fig.subplots_adjust(left=.035, right=.98, bottom=.21, top=.85, wspace=.04)
track = sm.track_xy(np.arange(30, 113))
cities = [('Atlanta', -84.39, 33.75), ('Tallahassee', -84.28, 30.44),
          ('Jacksonville', -81.66, 30.33), ('Charleston', -79.93, 32.78),
          ('Tampa', -82.46, 27.95), ('Mobile', -88.04, 30.69)]
for ax, peak, title, stats in zip(axes, (old_peak, sm.G), ('Original storm', 'Double the radius'),
                                 (summary['original'], summary['doubled'])):
    sm.m.drawmapboundary(fill_color='#e6ecf1', linewidth=.5, ax=ax)
    sm.m.fillcontinents(color='#f5f3ee', lake_color='#e6ecf1', ax=ax)
    ax.contourf(sm.X, sm.Y, peak, levels=FILL_LEVELS, colors=COLORS, alpha=.9)
    sm.m.drawstates(linewidth=.5, color='#4d4d55', ax=ax)
    sm.m.drawcoastlines(linewidth=.6, color='#3a3a42', ax=ax)
    ax.contour(sm.X, sm.Y, peak, levels=[58], colors=[INK], linewidths=.8)
    ax.contour(sm.X, sm.Y, peak, levels=[74], colors=[INK], linewidths=.8, linestyles='dotted')
    ax.plot(track[:, 0], track[:, 1], color=INK, lw=1.0)
    ax.plot(*sm.center_xy(54), marker='*', ms=11, mfc='#ffd23f', mec=INK)
    for name, lon, lat in cities:
        x, y = sm.m(lon, lat)
        ax.plot(x, y, 'o', ms=2.5, color=INK)
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=7,
                    bbox=dict(fc='white', ec='none', alpha=.75, pad=.7))
    ax.set_title(title, fontsize=15, color=INK, weight='bold', pad=13)
    ax.text(.5, -.055, f"Grid peak {stats['grid_max_mph']:.1f} mph  |  Land peak {stats['land_max_mph']:.1f} mph\n"
            f"58+ mph event footprint: {stats['event_footprint_km2']['58']:,} km² within map",
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color=INK, linespacing=1.6)
fig.text(.04, .95, 'Same storm. Twice the size.', fontsize=24, weight='bold', color=INK)
fig.text(.04, .91, 'Apalachee Bay sou’easter · Dec 15–19, 2026 · same track, pressure and prescribed jet speeds',
         fontsize=10, color=MUTED)
fig.text(.98, .95, 'SIMULATED SCENARIO', ha='right', fontsize=9, color=ACCENT, weight='bold')
draw_cbar(fig, [.04, .085, .42, .017])
fig.text(.04, .12, 'Maximum gust at each point (mph)', fontsize=9, color=INK)
fig.text(.51, .102, 'Solid contour: 58 mph gust  ·  Dotted: 74 mph gust\nYellow star: landfall near St. Marks',
         fontsize=9, color=INK, linespacing=1.5)
fig.text(.04, .025, 'Cyclone core, quadrant radii, conveyor arc and radial decay lengths doubled. Local sting jets and downbursts retain their scales.\n'
         'Areas cover only this map; the doubled storm extends beyond it. Gusts respond to the existing terrain and mixing. Fictional, uncalibrated model.',
         fontsize=8, color=MUTED, linespacing=1.6)
fig.savefig(HERE / 'size_comparison.png', facecolor='white')
print(json.dumps(summary, indent=2))
for row in rows:
    if row['id'] in ('KATL', 'KTLH', 'KJAX', 'KSAV', 'KCHS', 'KTPA', 'KMCO', 'KMOB', 'KVLD'):
        print(row)
print('Verified doubled radii, preserved scenario inputs, source fingerprints, finite fields, frame maxima and repeatability.')
