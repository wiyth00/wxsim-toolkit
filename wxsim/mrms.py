"""Site-independent, MRMS-inspired composite reflectivity for generated storms.

Run: python -m wxsim.mrms scenarios/seaboard_outbreak_2027/mrms_seaboard.json
This is a synthetic storm-primitive mosaic, not an actual MRMS product or radar observation.
"""
import argparse
import json
import math
import zlib
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

from . import geo
from .radar import primitives as prim, storms as st
from .radar.colormaps import reflectivity
from .radar.field import _sample, _texture, _vweight


class Mosaic:
    """Gridded reflectivity, maximum over constant-height slices, without radar beam geometry."""

    def __init__(self, storms, extent, resolution_km=2.0, heights_km=(0.5, 1.5, 3.5, 5.5), seed=24):
        self.storms = storms
        self.extent = extent
        self.heights = np.asarray(heights_km, dtype=float)
        lon0, lon1, lat0, lat1 = extent
        self.tr = geo.transformer((lat0 + lat1) / 2, (lon0 + lon1) / 2)
        corners = [self.tr.transform(lo, la) for lo in (lon0, lon1) for la in (lat0, lat1)]
        xmin, xmax = min(x for x, _ in corners), max(x for x, _ in corners)
        ymin, ymax = min(y for _, y in corners), max(y for _, y in corners)
        self.x = np.arange(xmin - resolution_km * 1000, xmax + resolution_km * 1000, resolution_km * 1000)
        self.y = np.arange(ymin - resolution_km * 1000, ymax + resolution_km * 1000, resolution_km * 1000)
        self.seed = seed
        self.detail, self.wu, self.wv = _texture(seed)

    def frame(self, when):
        # Sum storm contributions in linear Z separately at each height, then composite.
        # Keep this site-independent: RadarSim's radar-site-aware three-body
        # hail spike is intentionally not part of the MRMS primitive mosaic.
        layers = np.zeros((len(self.heights), len(self.y), len(self.x)), dtype=np.float32)
        active = 0
        for s in self.storms:
            if s.get('archetype') == 'dynamic':
                from .radar import dynamic as dynamic_engine

                if not st.active(s, when):
                    continue
                age_min = st.age_minutes(s, when)
                params = st.dynamic_params(s, when)
                la, lo = st.position(s, when)
                sx, sy = self.tr.transform(lo, la)
                reach_km = float(dynamic_engine.reach_km(age_min, params))
                if not np.isfinite(reach_km) or reach_km <= 0:
                    continue
                reach_m = reach_km * 1000
                ix0, ix1 = np.searchsorted(self.x, [sx - reach_m, sx + reach_m])
                iy0, iy1 = np.searchsorted(self.y, [sy - reach_m, sy + reach_m])
                if ix0 == ix1 or iy0 == iy1:
                    continue
                dx = (self.x[ix0:ix1][None, :] - sx) / 1000
                dy = (self.y[iy0:iy1][:, None] - sy) / 1000
                th = math.radians(st.heading(s, when))
                u = dx * math.sin(th) + dy * math.cos(th)
                v = -dx * math.cos(th) + dy * math.sin(th)
                if s.get('split', 'right') == 'left':
                    v = -v
                for hi, z_km in enumerate(self.heights):
                    db = np.asarray(dynamic_engine.reflectivity(
                        u, v, float(z_km), age_min, params,
                        seed=self.seed, storm_id=s.get('id', 'dynamic'),
                    ), dtype=np.float32)
                    with np.errstate(over='ignore', invalid='ignore'):
                        power = np.where(np.isfinite(db), 10 ** (db / 10), 0).astype(np.float32)
                    layers[hi, iy0:iy1, ix0:ix1] += power
                active += 1
                continue
            if when < s['life']['start'] or when > s['life']['end']:
                continue
            ps = st.primitives(s, when)
            if not ps:
                continue
            la, lo = st.position(s, when)
            sx, sy = self.tr.transform(lo, la)
            reach_m = (max(prim.reach(P) for P in ps) + 3) * 1000
            ix0, ix1 = np.searchsorted(self.x, [sx - reach_m, sx + reach_m])
            iy0, iy1 = np.searchsorted(self.y, [sy - reach_m, sy + reach_m])
            if ix0 == ix1 or iy0 == iy1:
                continue
            dx = (self.x[ix0:ix1][None, :] - sx) / 1000
            dy = (self.y[iy0:iy1][:, None] - sy) / 1000
            th = math.radians(st.heading(s, when))
            u = dx * math.sin(th) + dy * math.cos(th)
            v = -dx * math.cos(th) + dy * math.sin(th)
            phase = zlib.crc32(s['id'].encode())
            ox = (phase % 110) - 55
            oy = ((phase // 110) % 110) - 55
            uu = u + 0.9 * _sample(self.wu, u + ox, v + oy)
            vv = v + 0.9 * _sample(self.wv, u + ox, v + oy)
            additive = np.zeros((len(self.heights), *u.shape), dtype=np.float32)
            subtractive = np.zeros_like(additive)
            for P in ps:
                weights = np.asarray([_vweight(h, 0.3, P['z']) for h in self.heights], dtype=np.float32)
                if max(weights) < 0.002:
                    continue
                if P['kind'] in prim.SUBTRACTIVE:
                    subtractive += weights[:, None, None] * prim.SUBTRACTIVE[P['kind']](uu, vv, P)
                else:
                    db = prim.ADDITIVE[P['kind']](uu, vv, P)
                    with np.errstate(over='ignore'):
                        power = np.where(np.isfinite(db), 10 ** (db / 10), 0).astype(np.float32)
                    additive += weights[:, None, None] * power
            texture = np.clip(4.0 * _sample(self.detail, u + ox, v + oy), -12, 12)
            with np.errstate(divide='ignore', invalid='ignore'):
                storm_db = 10 * np.log10(additive) - subtractive + texture
                power = np.where(np.isfinite(storm_db), 10 ** (storm_db / 10), 0)
            layers[:, iy0:iy1, ix0:ix1] += power.astype(np.float32)
            active += 1
        with np.errstate(divide='ignore'):
            dbz = 10 * np.log10(np.max(layers, axis=0))
        dbz = np.where(dbz > 66, 66 + 10 * np.tanh((dbz - 66) / 10), dbz)
        return np.ma.masked_less(dbz, 5), active


def render(config_path, times_override=None, storms_file=None, outdir_override=None):
    path = Path(config_path)
    cfg = json.loads(path.read_text())
    if storms_file:
        storms_path = Path(storms_file)
        if not storms_path.is_absolute() and not storms_path.exists():
            storms_path = path.parent / storms_path
    else:
        storms_path = path.parent / cfg['storms']
    storms = st.load(storms_path)
    mosaic = Mosaic(storms, cfg['extent'], cfg.get('resolution_km', 2),
                    cfg.get('heights_km', [0.5, 1.5, 3.5, 5.5]), cfg.get('seed', 24))
    start = st.parse_time(cfg['start']); end = st.parse_time(cfg['end'])
    step = timedelta(minutes=cfg.get('step_min', 20))
    times = [st.parse_time(t) for t in times_override] if times_override else []
    if not times:
        t = start
        while t <= end:
            times.append(t); t += step
    outdir = Path(outdir_override) if outdir_override else path.parent / cfg.get('output_dir', 'output/mrms')
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.2, 11.4), dpi=115)
    fig.set_facecolor('#0b1623')
    fig.subplots_adjust(left=.04, right=.96, bottom=.095, top=.905)
    lon0, lon1, lat0, lat1 = cfg['extent']
    geo.basemap(ax, geo.projector(mosaic.tr), geo.extent_box(*cfg['extent'], pad=.7),
                land_color='#26343a', ocean='#152b3c', county_color='#96a2a5',
                county_lw=.26, county_alpha=.32, state_color='#d0d7d7', state_lw=.9,
                z_land=0, z_county=5, z_state=6)
    geo.set_extent(ax, mosaic.tr, lon0, lon1, lat0, lat1, inner=True)
    cmap, norm = reflectivity()
    mesh = ax.pcolormesh(mosaic.x, mosaic.y, np.ma.masked_all((len(mosaic.y), len(mosaic.x))),
                         shading='nearest', cmap=cmap, norm=norm, zorder=3, rasterized=True)
    for name, lat, lon in cfg.get('cities', []):
        x, y = mosaic.tr.transform(lon, lat)
        ax.plot(x, y, '.', color='#f5f2df', ms=3, zorder=12)
        ax.annotate(name, (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=7, color='#f9f8ef', zorder=12,
                    path_effects=[pe.withStroke(linewidth=2.3, foreground='#101a25')])
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_color('#667785')
    title = fig.text(.04, .957, 'ATLANTIC SEABOARD  |  SIMULATED COMPOSITE REFLECTIVITY',
                     color='white', weight='bold', size=14, va='center')
    stamp = fig.text(.96, .957, '', color='#b9f2fa', weight='bold', size=13, ha='right', va='center')
    fig.text(.04, .034, 'MRMS-style grid mosaic  •  max across 0.5 / 1.5 / 3.5 / 5.5 km AGL  •  synthetic',
             size=9, color='#bfd0d9')
    fig.text(.04, .012, 'Fictional April 29, 2027 scenario  |  EDT = UTC − 4h',
             size=8, color='#8f9fae')
    cax = fig.add_axes([.26, .067, .56, .017])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[5,15,25,35,45,55,65,75])
    cb.ax.tick_params(colors='#d9e5eb', labelsize=7)
    cb.outline.set_edgecolor('#9db0bd')
    fig.text(.19, .069, 'dBZ', color='#d9e5eb', fontsize=8)
    stills = set(cfg.get('stills', []))
    images = []
    for t in times:
        iso = t.strftime('%Y-%m-%dT%H:%MZ')
        field, count = mosaic.frame(iso)
        mesh.set_array(field.ravel())
        stamp.set_text(t.strftime('%H%MZ  /  ') + (t - timedelta(hours=4)).strftime('%I:%M %p EDT').lstrip('0'))
        img = outdir / ('MRMS_Composite_' + t.strftime('%Y%m%d_%H%M') + '.png')
        fig.savefig(img, facecolor=fig.get_facecolor(), dpi=115)
        print(f'{iso}: {count} active storms; saved {img}', flush=True)
        if not times_override:
            images.append(Image.open(img).convert('RGB'))
            if iso not in stills and not cfg.get('keep_frames', False):
                img.unlink()
    if images:
        anim = outdir / 'Seaboard_MRMS_loop.gif'
        images[0].save(anim, save_all=True, append_images=images[1:], duration=350, loop=0,
                       optimize=True)
        print(f'saved {anim}', flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', help='JSON rendering config')
    ap.add_argument('--time', action='append', help='render only this UTC ISO time; repeatable')
    ap.add_argument('--storms-file', help='override the storms JSON selected by the config')
    ap.add_argument('--outdir', help='override the output directory selected by the config')
    args = ap.parse_args()
    render(args.config, args.time, args.storms_file, args.outdir)


if __name__ == '__main__':
    main()
