"""Compare size-only controls with explicit fictional pressure/wind scaling."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from size_sweep import load_model, original, COLORS, FILL_LEVELS, INK, MUTED, ACCENT, draw_cbar, ST

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'pressure_comparison'


def collect(scale, pressure_scaled):
    dirname = ('size_2p5x' if scale == 2.5 else 'size_3x') + ('_pressure' if pressure_scaled else '')
    path = ROOT / dirname
    model = load_model(scale, path)
    np.testing.assert_array_equal(model.WP[:, :3], original.WP[:, :3])
    expected = 1012 - scale * (1012 - original.WP[:, 3]) if pressure_scaled else original.WP[:, 3]
    np.testing.assert_allclose(model.WP[:, 3], expected)
    np.testing.assert_array_equal(model.LLJ, original.LLJ)
    np.testing.assert_allclose(model.RMAX[:, 1], scale * original.RMAX[:, 1])
    np.testing.assert_allclose(model.RADII, scale * original.RADII)
    assert model.STING == original.STING and model.CONVECTION == original.CONVECTION
    np.testing.assert_array_equal(model.X, original.X)
    for cache in (model.PEAK, model.FRAMES, path / 'station_series.npz'):
        assert model.cache_valid(cache), f'Stale cache: {cache}'
    with np.load(model.PEAK) as d:
        peak, times = d['G'], d['T']
    with np.load(model.FRAMES) as d:
        frames, hours = d['F'].astype(np.float32), d['ft']
    with np.load(path / 'station_series.npz') as d:
        station_peaks = d['gust_mph'].max(axis=0)
    np.testing.assert_allclose(hours, np.arange(30, 112.1, .5))
    assert np.isfinite(peak).all() and (peak >= 5).all()
    assert ((times >= 12) & (times <= 114)).all()
    # Float16 rounding tolerance grows with the extreme synthetic gust values.
    assert float((frames.max(axis=0) - peak).max()) <= .13
    repeated = model.inst_gust(54)
    model.inst_gust(78)
    np.testing.assert_array_equal(repeated, model.inst_gust(54))
    if pressure_scaled:
        assert model.PRESSURE_DEFICIT_SCALE == scale
        np.testing.assert_allclose(model.CYCLONE_WIND_SCALE, np.sqrt(scale))
    summary = dict(radius_scale=scale, pressure_deficit_scale=scale if pressure_scaled else 1.,
        cyclone_wind_scale=float(np.sqrt(scale)) if pressure_scaled else 1.,
        min_pressure_mb=model.pressure(51), landfall_pressure_mb=model.pressure(54),
        grid_max_mph=round(float(peak.max()), 2), land_max_mph=round(float(peak[model.land].max()), 2),
        event_footprint_km2={str(k): int((peak >= k).sum() * 9) for k in (58, 74, 100, 150, 200)},
        model_signature=model.signature(), simulated=True)
    return summary, peak, station_peaks


def run():
    OUT.mkdir(exist_ok=True)
    cases = [(2.5, False), (2.5, True), (3., False), (3., True)]
    results = [collect(*case) for case in cases]
    summary = dict(simulated=True, scenarios=[r[0] for r in results],
        rule='p(t)=1012-s*(1012-p_original(t)); multiply the matching size-only cyclone tangential profile by sqrt(s).',
        limitations='Assumed sensitivity, not a balanced physical solution or a necessary consequence of storm size. All areas limited to the original map; geography and local convective/sting scales fixed.')
    (OUT / 'pressure_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    rows = []
    for j, (sid, name, lat, lon) in enumerate(ST):
        row = dict(id=sid, name=name, simulated=True)
        for (scale, pressure_scaled), result in zip(cases, results):
            row[f'{scale:g}x_' + ('scaled_pressure_mph' if pressure_scaled else 'size_only_mph')] = round(float(result[2][j]), 2)
        rows.append(row)
    with (OUT / 'station_pressure_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig = plt.figure(figsize=(13, 15), dpi=160, facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.04, right=.98, bottom=.16, top=.88, hspace=.22, wspace=.06)
    track = original.track_xy(np.arange(30, 113))
    for i, ((scale, scaled), (stats, peak, _)) in enumerate(zip(cases, results)):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        original.m.drawmapboundary(fill_color='#e6ecf1', linewidth=.5, ax=ax)
        original.m.fillcontinents(color='#f5f3ee', lake_color='#e6ecf1', ax=ax)
        ax.contourf(original.X, original.Y, peak, levels=FILL_LEVELS, colors=COLORS, alpha=.9)
        original.m.drawstates(linewidth=.5, color='#4d4d55', ax=ax)
        original.m.drawcoastlines(linewidth=.6, color='#3a3a42', ax=ax)
        # The inherited palette saturates at 120; extra contours expose extremes.
        levels = [k for k in (74, 150, 200, 250) if peak.min() < k < peak.max()]
        contour_colors = [INK if k < 150 else '#fff4fa' for k in levels]
        contours = ax.contour(original.X, original.Y, peak, levels=levels, colors=contour_colors, linewidths=.9)
        ax.clabel(contours, fmt='%d', fontsize=7, inline=True)
        ax.plot(track[:, 0], track[:, 1], color=INK, lw=1.)
        ax.plot(*original.center_xy(54), marker='*', ms=10, mfc='#ffd23f', mec=INK)
        for name, lon, lat in [('Mobile', -88.04, 30.69), ('Atlanta', -84.39, 33.75), ('Tampa', -82.46, 27.95)]:
            x, y = original.m(lon, lat)
            ax.plot(x, y, 'o', ms=2.5, color=INK)
            ax.annotate(name, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=7,
                        bbox=dict(fc='white', ec='none', alpha=.75, pad=.7))
        ax.set_title(f'{scale:g}× radius · ' + ('scaled deficit + winds' if scaled else 'size only'),
                     fontsize=14, weight='bold', color=INK, pad=12)
        ax.text(.5, -.045, f"Minimum {stats['min_pressure_mb']:g} mb · grid peak {stats['grid_max_mph']:.1f} mph\n"
                f"Land peak {stats['land_max_mph']:.1f} mph", transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color=INK, linespacing=1.5)
    fig.text(.04, .956, 'Now scale the pressure deficit, too.', fontsize=24, weight='bold', color=INK)
    fig.text(.04, .925, 'Explicit fictional rule: pressure deficit × radius multiplier; cyclone winds × square root of that multiplier',
             fontsize=10, color=MUTED)
    fig.text(.98, .979, 'EXTREME SIMULATED SCENARIO', ha='right', fontsize=9, weight='bold', color=ACCENT)
    fig.text(.04, .108, 'Maximum gust at each point (mph)', fontsize=9, color=INK)
    draw_cbar(fig, [.04, .081, .43, .014])
    fig.text(.52, .09, 'Contours: 74 / 150 / 200 / 250 mph where present\nPalette saturates at 120+ mph; contours show higher values.',
             fontsize=9, color=INK, linespacing=1.5)
    fig.text(.04, .026, 'The pressure and wind scaling is assumed, not a physically balanced solution. Size alone does not imply these pressures.\n'
             'All panels use the original Southeast domain. Areas beyond the map are omitted; local sting jets, convection and geographic masks are retained.',
             fontsize=8, color=MUTED, linespacing=1.6)
    fig.savefig(OUT / 'pressure_comparison.png', facecolor='white')
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row['id'] in ('KBHM', 'KMOB', 'KJAX', 'KATL', 'KTLH', 'KVLD'):
            print(row)
    print('Verified prescribed pressure rule, size scaling, preserved inputs, source fingerprints, finite fields, frames and repeatability.')


if __name__ == '__main__':
    run()
