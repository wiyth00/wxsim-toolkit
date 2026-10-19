"""Zoom the saved blue-hook event and advect 2-D tracers through its winds."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import model as fm

CASE = HERE.parent / 'size_3x'
BEGIN, END, REFERENCE = 64., 80., 70.


def label(hour):
    dt = datetime(2026, 12, 15, tzinfo=timezone.utc) + timedelta(hours=float(hour))
    return dt.astimezone(ZoneInfo('America/New_York')).strftime('%a Dec %d · %-I:%M %p EST')


def run():
    meta = json.loads((CASE / 'summary.json').read_text())
    assert meta['source_fingerprint'] == fm.SOURCE_FINGERPRINT
    with np.load(CASE / 'fields.npz') as data:
        assert str(data['source_fingerprint']) == fm.SOURCE_FINGERPRINT
        hours = data['hours'].astype(float)
        theta = data['theta_K']
        front = data['F_K_100km_3h']
        u, v = data['u_ms'], data['v_ms']

    def velocity(t, xy):
        lo = int(np.clip(np.searchsorted(hours, t, side='right') - 1, 0, len(hours) - 2))
        weight = (t - hours[lo]) / (hours[lo + 1] - hours[lo])
        result = np.full_like(xy, np.nan, dtype=float)
        active = np.isfinite(xy).all(axis=1)
        coords = [xy[active, 1] / fm.DX, xy[active, 0] / fm.DX]
        for k, field in enumerate((u, v)):
            a = map_coordinates(field[lo], coords, order=1, mode='constant', cval=np.nan)
            b = map_coordinates(field[lo + 1], coords, order=1, mode='constant', cval=np.nan)
            result[active, k] = a * (1 - weight) + b * weight
        return result

    ref_index = int(np.argmin(np.abs(hours - REFERENCE)))
    targets = []
    for low, high in ((31.5, 32.3), (29.5, 30.5), (28.0, 29.0)):
        roi = ((fm.bg.LON >= -89) & (fm.bg.LON <= -83)
               & (fm.bg.LAT >= low) & (fm.bg.LAT <= high))
        j, i = np.unravel_index(np.where(roi, front[ref_index], np.inf).argmin(), fm.X.shape)
        assert front[ref_index, j, i] < 0
        targets.append((fm.X[j, i], fm.Y[j, i]))
    hook_origins = np.asarray(targets, dtype=float)
    for h in np.arange(REFERENCE, BEGIN + 1e-8, -1 / 60):
        a = velocity(h, hook_origins)
        hook_origins -= 60 * velocity(h - 1 / 120, hook_origins - 30 * a)
    assert np.isfinite(hook_origins).all(), 'Hook backtrace exited the saved wind domain'
    gulf_seeds = [(-87.0, 28.2), (-85.5, 28.3), (-84.4, 28.5)]
    initial = np.vstack((hook_origins, np.asarray([fm.m(*p) for p in gulf_seeds])))
    seeds = [fm.m(*p, inverse=True) for p in hook_origins] + gulf_seeds
    groups = ['Hook-side seeds'] * 3 + ['Gulf seeds'] * 3
    times = np.arange(BEGIN, END + 1 / 120, 1 / 60)
    tracks = np.empty((len(times), len(seeds), 2), float)
    tracks[0] = initial
    for k in range(len(times) - 1):
        h = times[k]
        a = velocity(h, tracks[k])
        mid = tracks[k] + 30 * a
        tracks[k + 1] = tracks[k] + 60 * velocity(h + 1 / 120, mid)
    np.savez_compressed(HERE / 'tracers.npz', hours=times, xy_m=tracks,
                        seed_lonlat=seeds, groups=groups,
                        source_fingerprint=fm.SOURCE_FINGERPRINT, simulated=True)
    # Check integration against half-sized steps, independent of rendering.
    fine = initial.copy()
    for h in np.arange(BEGIN, REFERENCE - 1e-8, 1 / 120):
        a = velocity(h, fine)
        fine += 30 * velocity(h + 1 / 240, fine + 15 * a)
    reference_track_index = int(round((REFERENCE - BEGIN) * 60))
    error = np.linalg.norm(tracks[reference_track_index] - fine, axis=1)
    active = np.isfinite(error)
    assert active.any() and error[active].max() < 3000, error
    target_error = np.linalg.norm(tracks[reference_track_index, :3] - targets, axis=1)
    assert target_error.max() < 3000, target_error

    corners = [fm.m(lon, lat) for lon in (-89.4, -81.5) for lat in (26.8, 35.4)]
    xmin, xmax = min(x for x, y in corners), max(x for x, y in corners)
    ymin, ymax = min(y for x, y in corners), max(y for x, y in corners)
    xi = np.flatnonzero((fm.X[0] >= xmin) & (fm.X[0] <= xmax))
    yi = np.flatnonzero((fm.Y[:, 0] >= ymin) & (fm.Y[:, 0] <= ymax))
    sl = (slice(yi[0], yi[-1] + 1), slice(xi[0], xi[-1] + 1))
    x, y = fm.X[sl], fm.Y[sl]
    extent = (x[0, 0] - fm.DX / 2, x[0, -1] + fm.DX / 2,
              y[0, 0] - fm.DX / 2, y[-1, 0] + fm.DX / 2)
    frame_indices = np.flatnonzero((hours >= BEGIN) & (hours <= END))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.2), dpi=100, facecolor='white')
    fig.subplots_adjust(left=.035, right=.98, top=.86, bottom=.18, wspace=.08)
    for ax in axes:
        fm.m.drawmapboundary(fill_color='#e7edf2', linewidth=0, ax=ax)
        fm.m.fillcontinents(color='#f5f3ee', lake_color='#e7edf2', zorder=.2, ax=ax)
        fm.m.drawcoastlines(linewidth=.7, color='#33343a', zorder=5, ax=ax)
        fm.m.drawstates(linewidth=.65, color='#55565d', zorder=5, ax=ax)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        for name, lon, lat in [('Atlanta', -84.39, 33.75), ('Birmingham', -86.80, 33.52),
                                ('Mobile', -88.04, 30.69), ('Tallahassee', -84.28, 30.44)]:
            cx, cy = fm.m(lon, lat)
            ax.plot(cx, cy, 'o', ms=2.5, color='#22232b', zorder=8)
            ax.annotate(name, (cx, cy), xytext=(4, 4), textcoords='offset points', fontsize=8,
                        zorder=8, bbox=dict(fc='white', ec='none', alpha=.8, pad=.8))
    im_theta = axes[0].imshow(theta[frame_indices[0]][sl], extent=extent, origin='lower',
        cmap='RdYlBu_r', vmin=274, vmax=297, zorder=2, interpolation='nearest')
    im_front = axes[1].imshow(front[frame_indices[0]][sl], extent=extent, origin='lower',
        cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-16, vcenter=0, vmax=16),
        zorder=2, interpolation='nearest')
    axes[0].set_title('Air masses + moving tracers', loc='left', weight='bold', fontsize=13)
    axes[1].set_title('Where the temperature contrast changes', loc='left', weight='bold', fontsize=13)
    qsl = (slice(None, None, 28), slice(None, None, 28))
    arrows = axes[0].quiver(x[qsl], y[qsl], u[frame_indices[0]][sl][qsl], v[frame_indices[0]][sl][qsl],
        angles='xy', scale_units='xy', scale=1 / 3500, width=.0022, color='#303139', alpha=.65, zorder=6)
    axes[0].quiverkey(arrows, .14, .045, 20, '20 m/s layer wind', coordinates='axes',
                      labelpos='E', fontproperties={'size': 7})
    mask = fm.cad_mask(fm.X, fm.Y)[sl]
    for ax in axes:
        ax.contour(x, y, mask, levels=[.5], colors=['#247f7b'], linestyles='dashed', linewidths=1., zorder=6)
    colors = ['#116d91'] * 3 + ['#bf491c'] * 3
    lines, dots = [], []
    for ax in axes:
        lines.append([ax.plot([], [], lw=2., color=color, alpha=.95, zorder=9)[0] for color in colors])
        dots.append([ax.plot([], [], 'o', ms=5, mfc=color, mec='white', mew=.8, zorder=10)[0] for color in colors])
    lows = [ax.plot([], [], marker='*', ms=14, mfc='#ffe17c', mec='#25252d', zorder=11)[0] for ax in axes]
    low_texts = [ax.text(0, 0, '', color='#25252d', fontsize=10, weight='bold', zorder=11,
                        bbox=dict(fc='white', ec='none', alpha=.85, pad=1.)) for ax in axes]
    contour_artists = []
    fig.text(.035, .965, 'Inside the blue hook', fontsize=23, weight='bold', color='#20212a')
    timestamp = fig.text(.035, .92, '', fontsize=10, color='#555660')
    fig.text(.98, .965, 'SIMULATED · 3× CASE', ha='right', fontsize=9, weight='bold', color='#9b1c63')
    c0 = fig.colorbar(im_theta, cax=fig.add_axes([.075, .105, .36, .017]), orientation='horizontal')
    c0.set_label('Potential temperature (K) · cooler → warmer', fontsize=9)
    c1 = fig.colorbar(im_front, cax=fig.add_axes([.57, .105, .36, .017]), orientation='horizontal', extend='both')
    c1.set_label('K / 100 km / 3 h · blue weakens, red strengthens', fontsize=9)
    for cb in (c0, c1):
        cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([0], [0], color='#116d91', marker='o', label='Hook tracers'),
                        Line2D([0], [0], color='#bf491c', marker='o', label='Gulf tracers'),
                        Line2D([0], [0], color='#247f7b', ls='--', label='Schematic CAD corridor')],
               loc='lower center', bbox_to_anchor=(.5, .019), ncol=3, frameon=False, fontsize=8)
    fig.text(.035, .008, 'Saved 2-D layer winds; 4-hour tracer trails. Wider frontogenesis color scale reveals detail within the original saturated blue.',
             fontsize=7, color='#666771')

    def draw(index):
        nonlocal contour_artists
        for artist in contour_artists:
            artist.remove()
        contour_artists = []
        t = hours[index]
        im_theta.set_data(theta[index][sl]); im_front.set_data(front[index][sl])
        arrows.set_UVC(u[index][sl][qsl], v[index][sl][qsl])
        for ax in axes:
            c = ax.contour(x, y, theta[index][sl], levels=np.arange(276, 298, 4),
                           colors='#32333a', linewidths=.55, alpha=.6, zorder=4)
            contour_artists.append(c)
        ti = int(round((t - BEGIN) * 60))
        first = max(0, ti - 240)
        for a in range(2):
            for j in range(len(seeds)):
                lines[a][j].set_data(tracks[first:ti + 1, j, 0], tracks[first:ti + 1, j, 1])
                dots[a][j].set_data([tracks[ti, j, 0]], [tracks[ti, j, 1]])
        cx, cy = fm.bg.center_xy(t)
        for point, text in zip(lows, low_texts):
            point.set_data([cx], [cy])
            text.set_position((cx + 15_000, cy - 30_000))
            text.set_text(f'L · {fm.bg.pressure(t):.0f} mb')
        timestamp.set_text(f'{label(t)} · arrows: lower-layer transport · contours: potential temperature')

    draw(ref_index)
    reference_xy = fm.m(-87.022, 31.905)
    marker = axes[1].plot(*reference_xy, marker='o', ms=8, mfc='white', mec='#20212a', mew=1.2, zorder=15)[0]
    note = axes[1].annotate('Spreading flow + stretching\nweaken the contrast here', reference_xy,
        xytext=(35, 20), textcoords='offset points', fontsize=9, color='#20212a',
        bbox=dict(fc='white', ec='#b7bbc2', alpha=.95, boxstyle='round,pad=.4'),
        arrowprops=dict(arrowstyle='->', color='#20212a'), zorder=15)
    fig.savefig(HERE / 'hook_zoom.png', dpi=160, facecolor='white')
    marker.remove(); note.remove()
    writer = FFMpegWriter(fps=5, codec='libx264', bitrate=2200,
                         extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                     '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    temporary = HERE / 'hook_zoom.encoding.mp4'
    with writer.saving(fig, str(temporary), dpi=100):
        for index in frame_indices:
            draw(index); writer.grab_frame(facecolor='white')
        for _ in range(10):
            writer.grab_frame(facecolor='white')
    import subprocess
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(temporary), '-f', 'null', '-'], check=True)
    temporary.replace(HERE / 'hook_zoom.mp4')
    plt.close(fig)
    trace_meta = dict(source_fingerprint=fm.SOURCE_FINGERPRINT, source_case=str(CASE),
                      simulated=True, start_hour=BEGIN, end_hour=END, reference_hour=REFERENCE,
                      dt_s=60, tracer_seed_lonlat=seeds, max_half_step_difference_m=float(error[active].max()),
                      hook_reference_targets_xy_m=np.asarray(targets).tolist(),
                      max_backward_forward_return_error_m=float(target_error.max()),
                      wind_time_interpolation='linear between stored half-hour winds',
                      note='2-D prescribed-flow tracers are not 3-D atmospheric parcel trajectories. Screenshot time/case is approximate.')
    (HERE / 'visual_provenance.json').write_text(json.dumps(trace_meta, indent=2) + '\n')
    print(json.dumps(trace_meta, indent=2))
    print('Saved hook_zoom.png and fully decoded hook_zoom.mp4')


if __name__ == '__main__':
    run()
