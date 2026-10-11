"""Diagnostics for the generator: mesoanalysis panels, storm-track summary, and radar frames of a generated run."""
import json, warnings as _pw
from pathlib import Path

_pw.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from .. import geo
from ..radar.storms import parse_time

BCOL = {'dryline': '#b5651d', 'wedge_front': '#1f5fd6', 'cold_front': '#1f5fd6', 'warm_front': '#d62728',
        'outflow': '#7f3fbf', 'boundary': '#444'}
DEF_CITIES = [('Atlanta', 33.749, -84.388), ('Athens', 33.961, -83.378), ('Macon', 32.841, -83.632),
              ('Columbus', 32.461, -84.988), ('Augusta', 33.471, -81.975), ('Birmingham', 33.519, -86.810),
              ('Chattanooga', 35.046, -85.309), ('Greenville', 34.853, -82.394), ('Columbia', 34.001, -81.035),
              ('Gainesville', 34.298, -83.824), ('LaGrange', 33.039, -85.031), ('Rome', 34.257, -85.165),
              ('Albany', 31.578, -84.156), ('Montgomery', 32.377, -86.300), ('Charlotte', 35.227, -80.843)]


def _base_ax(ax, cfg, tr, proj, cities=True, fs=7.5):
    lon0, lon1, lat0, lat1 = cfg['domain']
    ext = geo.extent_box(lon0, lon1, lat0, lat1, pad=0.8)
    geo.basemap(ax, proj, ext, land_color='#f7f6f2', ocean='#dce7ef', county_color='#b9c0c7', county_lw=0.2,
                county_alpha=0.6, state_color='#3d434a', state_lw=0.9, z_land=0, z_county=5, z_state=6)
    if cities:
        for n, la, lo in cfg.get('cities', DEF_CITIES):
            x, y = tr.transform(lo, la)
            ax.plot(x, y, 'o', ms=2.2, color='#111', zorder=9)
            ax.text(x + 5000, y + 4000, n, fontsize=fs, zorder=9, color='#111',
                    path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
    geo.set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=True)


def _boundaries(ax, env, when, tr, lw=2.2, alpha=1.0, label=True):
    for name, kind, L in env.boundaries_at(when):
        x, y = tr.transform(L[:, 0], L[:, 1])
        col = BCOL.get(kind, '#444')
        ax.plot(x, y, color=col, lw=lw, alpha=alpha, zorder=7, solid_capstyle='round')
        if kind == 'dryline':   # scallops
            ax.plot(x[::5], y[::5], 'o', ms=5, mfc='none', mec=col, mew=1.4, alpha=alpha, zorder=7)
        if label:
            k = len(x) // 3
            ax.text(x[k], y[k], ' ' + name, color=col, fontsize=8, weight='bold', zorder=8, alpha=alpha,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def _pools(ax, env, when, tr, fill=True, lw=1.6, label=False):
    """Cold pools: light purple fill for the stable interior, OFB edge as a dash-dot purple line (WPC style)."""
    P = getattr(env, 'pools', None)
    if P is None:
        return 0
    n = 0
    for pid, I, geom, parents in P.snapshot(when):
        gs = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for g in gs:
            lo, la = env.lonlat(*np.asarray(g.exterior.coords).T)
            x, y = tr.transform(lo, la)
            if fill:
                ax.fill(x, y, color=BCOL['outflow'], alpha=0.10 + 0.22 * I, lw=0, zorder=6.5)
            ax.plot(x, y, color=BCOL['outflow'], lw=lw * (0.6 + 0.6 * I), ls=(0, (6, 2, 1.5, 2)), zorder=7)
            if label:
                k = int(np.argmax(y))
                ax.text(x[k], y[k], f' {pid}', color=BCOL['outflow'], fontsize=7, zorder=8,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
            n += 1
    return n


def mesoanalysis(cfg, env, when, out):
    lon0, lon1, lat0, lat1 = cfg['domain']
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    lons, lats = np.meshgrid(np.linspace(lon0, lon1, 180), np.linspace(lat0, lat1, 130))
    F = env.sample(lons, lats, when)
    X, Y = tr.transform(lons, lats)
    panels = [
        ('MLCAPE (J/kg), MLCIN < −50 hatched', F['mlcape'], [250, 500, 1000, 1500, 2000, 2500, 3000, 3500], 'YlOrRd'),
        ('0–6 km shear (kt) + vectors', F['shear_kt'], [30, 40, 50, 60, 70, 80], 'PuBu'),
        ('0–1 km SRH (m²/s²), LCL (m) contours', F['srh01'], [50, 100, 150, 200, 250, 300, 400], 'RdPu'),
        ('STP (fill), SHIP (contours)', F['stp'], [0.5, 1, 2, 3, 4, 6, 8], 'Reds'),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(17, 12), facecolor='white')
    for ax, (title, fld, lv, cm) in zip(axs.flat, panels):
        cmap = plt.get_cmap(cm)
        norm = BoundaryNorm(lv, cmap.N, extend='max')
        cf = ax.contourf(X, Y, fld, levels=lv, cmap=cmap, norm=norm, extend='max', alpha=0.85, zorder=2)
        cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.01); cb.ax.tick_params(labelsize=8)
        if 'MLCAPE' in title:
            ax.contourf(X, Y, F['mlcin'], levels=[-1000, -50], colors='none', hatches=['///'], zorder=3)
            ax.contour(X, Y, F['mlcin'], levels=[-50], colors='#555', linewidths=0.8, zorder=3)
        if 'shear' in title:
            sl = (slice(4, None, 10), slice(4, None, 10))
            ax.barbs(X[sl], Y[sl], F['shr_u'][sl], F['shr_v'][sl], length=5.5, color='#222', zorder=8)
        if 'LCL' in title:
            cs = ax.contour(X, Y, F['lcl'], levels=[1000, 1500, 2000, 2500], colors='#1f4e8c', linewidths=1.1, zorder=4)
            ax.clabel(cs, fmt='%d', fontsize=7)
        if 'SHIP' in title:
            cs = ax.contour(X, Y, F['ship'], levels=[1, 2, 3], colors='#2b6d2b', linewidths=1.3, zorder=4)
            ax.clabel(cs, fmt='%.0f', fontsize=8)
        _base_ax(ax, cfg, tr, proj, fs=7)
        _boundaries(ax, env, when, tr, label=False)
        _pools(ax, env, when, tr, fill=False, lw=1.3)
        ax.set_title(title, fontsize=11, loc='left')
    t = parse_time(when)
    fig.suptitle(f"Simulated mesoanalysis  •  {t.strftime('%H%MZ %a %b %d %Y')}  •  environment driving the storm generator",
                 fontsize=15, weight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110, facecolor='white'); plt.close(fig)
    print('saved', out)


def _mode_color(p):
    if p.get('qlcs'):
        return '#6a3d9a'
    if p['elev'] > 0.6:
        return '#3a78c2'
    if p['hb'] > 0.6:
        return '#c49a5a'
    if p['sup'] > 0.6:
        return '#d62728'
    return '#8c8c8c'


def track_summary(cfg, env, storms_file, out, seed):
    lon0, lon1, lat0, lat1 = cfg['domain']
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    ax = fig.add_axes([0.01, 0.2, 0.98, 0.74])
    _base_ax(ax, cfg, tr, proj)
    times = cfg.get('maps', {}).get('times', [cfg['start']])
    for i, w in enumerate(times):
        _boundaries(ax, env, w, tr, lw=1.6, alpha=0.25 + 0.6 * i / max(1, len(times) - 1), label=(i == len(times) - 1))
    rows = []
    for s in storms_file['storms']:
        tk, kf = s['track'], s['keyframes']
        xs, ys = tr.transform([p[2] for p in tk], [p[1] for p in tk])
        ls = '--' if s['split'] == 'left' else '-'
        for k in range(len(tk) - 1):
            p = kf[k]['params']
            live = p['dz'] > -12
            ax.plot(xs[k:k + 2], ys[k:k + 2], color=_mode_color(p), lw=2.6 if live else 1.0, ls=ls,
                    alpha=1 if live else 0.5, zorder=10, solid_capstyle='round')
        ax.plot(xs[0], ys[0], 'o', ms=4, color='#222', zorder=11)
        ax.text(xs[0], ys[0] - 9000, f"{s['id']}\n{tk[0][0][11:16].replace(':', '')}Z", fontsize=7, ha='center', va='top',
                zorder=11, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        for tor in s['meta']['tornadoes']:
            a = tr.transform(tor['lon'], tor['lat']); b = tr.transform(tor.get('end_lon', tor['lon']), tor.get('end_lat', tor['lat']))
            ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=5, zorder=12, solid_capstyle='round')
            ax.text(b[0], b[1] + 6000, f"EF{tor['ef']}", fontsize=9, weight='bold', zorder=13,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='yellow')])
        mh = s['meta']['max_hail_in']
        if mh >= 2:
            k = int(np.argmax([f['params']['hail'] for f in kf]))
            ax.plot(xs[k], ys[k], marker='^', ms=6 + 3 * mh, mfc='#2ca02c', mec='black', zorder=12)
        tors = ', '.join(f"EF{x['ef']}" for x in s['meta']['tornadoes']) or '–'
        rows.append((s['id'], s['split'][0].upper(), tk[0][0][11:16], tk[-1][0][11:16], f"{mh:.2f}", tors,
                     max(s['meta']['modes'], key=lambda m: ('supercell' in m, len(m)))))
    ax.set_title(f"Generated storm tracks  •  seed {seed}  •  {len(storms_file['storms'])} storms", fontsize=15,
                 weight='bold', loc='left')
    handles = [Line2D([], [], color='#d62728', lw=3, label='Supercell'), Line2D([], [], color='#8c8c8c', lw=3, label='Ordinary / multicell'),
               Line2D([], [], color='#3a78c2', lw=3, label='Elevated'), Line2D([], [], color='#c49a5a', lw=3, label='High-based'),
               Line2D([], [], color='#6a3d9a', lw=3, label='QLCS'), Line2D([], [], color='#555', lw=2, ls='--', label='Left-moving split'),
               Line2D([], [], color='black', lw=5, label='Tornado'),
               Line2D([], [], marker='^', color='none', mfc='#2ca02c', mec='black', ms=9, label='Peak hail ≥ 2"')]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    tax = fig.add_axes([0.01, 0.005, 0.98, 0.18]); tax.axis('off')
    cols = ['Storm', 'Mover', 'Start', 'End', 'Max hail (in)', 'Tornadoes', 'Peak mode']
    tbl = tax.table(cellText=rows[:28], colLabels=cols, loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 0.95)
    if len(rows) > 14:
        tbl.remove()
        half = (len(rows) + 1) // 2
        for j, part in enumerate((rows[:half], rows[half:])):
            if not part:
                continue
            ax_ = fig.add_axes([0.01 + 0.495 * j, 0.005, 0.485, 0.18]); ax_.axis('off')
            t_ = ax_.table(cellText=part, colLabels=cols, loc='upper center', cellLoc='center')
            t_.auto_set_font_size(False); t_.set_fontsize(7.5); t_.scale(1, 0.95)
    fig.savefig(out, dpi=120, facecolor='white'); plt.close(fig)
    print('saved', out)


def _storm_state(s, when):
    """(lat, lon, keyframe params) of a generated storm at `when`, or None if it isn't alive then."""
    tk = s['track']
    t = parse_time(when)
    if t < parse_time(tk[0][0]) or t > parse_time(tk[-1][0]):
        return None
    ts = [parse_time(p[0]).timestamp() for p in tk]
    i = int(np.clip(np.searchsorted(ts, t.timestamp()), 0, len(tk) - 1))
    la = float(np.interp(t.timestamp(), ts, [p[1] for p in tk])); lo = float(np.interp(t.timestamp(), ts, [p[2] for p in tk]))
    return la, lo, s['keyframes'][i]['params']


def boundary_evolution(cfg, env, storms_file, out, seed, every_h=1):
    """Small multiples: dynamic boundaries, cold pools/OFBs and live storms every `every_h` hours."""
    from datetime import timedelta
    lon0, lon1, lat0, lat1 = cfg['domain']
    tr = geo.transformer(lat_0=(lat0 + lat1) / 2, lon_0=(lon0 + lon1) / 2)
    proj = geo.projector(tr)
    t0, t1 = parse_time(cfg['start']), parse_time(cfg['end'])
    times = []
    t = t0
    while t <= t1:
        times.append(t.strftime('%Y-%m-%dT%H:%MZ')); t += timedelta(hours=every_h)
    ncol = 4; nrow = int(np.ceil(len(times) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow + 0.8), facecolor='white')
    for ax in axs.flat[len(times):]:
        ax.axis('off')
    for ax, w in zip(axs.flat, times):
        _base_ax(ax, cfg, tr, proj, cities=False)
        for n, la, lo in cfg.get('cities', DEF_CITIES)[:6]:
            x, y = tr.transform(lo, la); ax.plot(x, y, 'o', ms=1.6, color='#333', zorder=9)
        # previous hour's boundaries ghosted for motion
        prev = (parse_time(w) - timedelta(hours=every_h)).strftime('%Y-%m-%dT%H:%MZ')
        if parse_time(prev) >= t0:
            _boundaries(ax, env, prev, tr, lw=1.2, alpha=0.25, label=False)
        _boundaries(ax, env, w, tr, lw=1.8, label=False)
        npool = _pools(ax, env, w, tr, lw=1.2)
        nst = 0
        for s in storms_file['storms']:
            st = _storm_state(s, w)
            if st is None or st[2]['dz'] < -14:
                continue
            x, y = tr.transform(st[1], st[0])
            ax.plot(x, y, marker='o', ms=3 + 0.25 * max(0, st[2]['dz'] + 14), color=_mode_color(st[2]), mec='black',
                    mew=0.5, zorder=11)
            nst += 1
        ax.set_title(f"{parse_time(w).strftime('%H%MZ')}   storms {nst}   pools {npool}", fontsize=10, loc='left')
    kinds = {b.get('kind', 'boundary') for b in cfg.get('boundaries', [])}
    names = {'dryline': 'Dryline', 'wedge_front': 'Wedge front', 'cold_front': 'Cold front', 'warm_front': 'Warm front',
             'boundary': 'Boundary'}
    handles = []
    for k in [k for k in names if k in kinds]:
        h = Line2D([], [], color=BCOL.get(k, '#444'), lw=2, label=names[k])
        if k == 'dryline':
            h = Line2D([], [], color=BCOL[k], lw=2, marker='o', mfc='none', label=names[k])
        handles.append(h)
    handles += [Line2D([], [], color=BCOL['outflow'], lw=1.6, ls=(0, (6, 2, 1.5, 2)), label='Outflow boundary (cold pool shaded)'),
                Line2D([], [], color='#999', lw=1.2, alpha=0.5, label='Boundary 1 h earlier'),
                Line2D([], [], marker='o', color='none', mfc='#d62728', mec='k', ms=7, label='Supercell'),
                Line2D([], [], marker='o', color='none', mfc='#8c8c8c', mec='k', ms=7, label='Ordinary'),
                Line2D([], [], marker='o', color='none', mfc='#3a78c2', mec='k', ms=7, label='Elevated'),
                Line2D([], [], marker='o', color='none', mfc='#c49a5a', mec='k', ms=7, label='High-based'),
                Line2D([], [], marker='o', color='none', mfc='#6a3d9a', mec='k', ms=7, label='QLCS')]
    fig.legend(handles=handles, loc='lower center', ncol=min(10, len(handles)), fontsize=9, frameon=False)
    fig.suptitle(f'Boundary evolution  •  seed {seed}  •  simulated (dynamic boundaries; OFBs come from the storms)',
                 fontsize=14, weight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(out, dpi=100, facecolor='white'); plt.close(fig)
    print('saved', out)


def radar_frames(cfg, storms_path, outdir, seed, base):
    m = cfg.get('maps', {})
    if not m.get('radar_template'):
        return
    from ..radar import render as rr
    rc = json.load(open(base / m['radar_template']))
    rc['storms_file'] = storms_path.name
    rc['products'] = ['base']
    rc['overlays'] = {'storm_ids': True, 'range_rings': False}
    rc['extra_views'] = []
    rc['_base_dir'] = str(base)
    rc['scenario_label'] = f'SIMULATED RADAR — generated storms (seed {seed})'
    rc['overlays']['tornado_markers'] = True
    sim = rr.make_sim(rc)
    for w in m.get('radar_times', []):
        rc['outputs'] = {'base': f"GEN_seed{seed}_{rc['site']['id']}_BREF_{parse_time(w).strftime('%Y%m%d%H%M')}.png"}
        rr.render(rc, outdir, w, sim=sim)
    L = m.get('radar_loop')
    if L:
        from ..radar.loop import make_loop
        make_loop(rc, L['start'], L['end'], Path(outdir) / f"GEN_seed{seed}_{rc['site']['id']}_BREF_loop.gif",
                  step_min=L.get('step_min', 10), width=L.get('width', 1200), frame_ms=L.get('frame_ms', 180))


def render_all(cfg, storms_file, env, outdir, seed, base=None):
    outdir = Path(outdir)
    for w in cfg.get('maps', {}).get('times', []):
        mesoanalysis(cfg, env, w, outdir / f"MESO_{parse_time(w).strftime('%Y%m%d%H%M')}.png")
    track_summary(cfg, env, storms_file, outdir / f'GEN_tracks_seed{seed}.png', seed)
    if getattr(env, 'pools', None) is not None or any('dynamic' in b for b in cfg.get('boundaries', [])):
        boundary_evolution(cfg, env, storms_file, outdir / f'GEN_boundaries_seed{seed}.png', seed,
                           every_h=cfg.get('maps', {}).get('evolution_every_h', 1))
