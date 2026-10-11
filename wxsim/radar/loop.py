"""Animated radar loop (GIF) from a radar config: one frame every `step` minutes between `start` and `end`.

usage: python -m wxsim.radar.loop scenarios/<name>/radar_XXXX.json --start 2027-04-17T20:00Z --end 2027-04-18T01:00Z \
           [--step 10] [--product base] [--view NAME] [--width 1200] [--out loop.gif]

The same RadarSim is reused for every frame, so each storm keeps its texture and the loop doesn't flicker.
Storms appear and die by their lifecycles, and boundary fine lines move with the boundaries in a generated storms file.
"""
import argparse, json, tempfile
from datetime import timedelta
from pathlib import Path

from PIL import Image

from . import render as rr
from .storms import parse_time


def frame_times(start, end, step_min):
    t, t1, out = parse_time(start), parse_time(end), []
    while t <= t1:
        out.append(t.strftime('%Y-%m-%dT%H:%MZ')); t += timedelta(minutes=step_min)
    return out


def make_loop(cfg, start, end, out, step_min=10, product='base', view=None, width=1200,
              frame_ms=180, end_hold_ms=1600, keep_frames=None):
    cfg = dict(cfg)
    cfg['products'] = [product]
    if view:                                   # use a named extra view as the main view
        v = next(v for v in cfg.get('extra_views', []) if v['name'] == view)
        cfg['view'] = v['view']
    cfg['extra_views'] = []
    cfg['_quiet'] = True
    cfg.setdefault('dpi', 110)
    sim = rr.make_sim(cfg)
    times = frame_times(start, end, step_min)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        fdir = Path(keep_frames) if keep_frames else Path(tmp)
        fdir.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(times):
            cfg['outputs'] = {product: f'frame_{i:03d}.png'}
            path = rr.render(cfg, fdir, w, sim=sim)[0]
            im = Image.open(path).convert('RGB')
            if width and im.width > width:
                im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            frames.append(im)
            print(f'  frame {i + 1}/{len(times)}  {w}', flush=True)
    # one shared palette keeps colors stable across the loop and the file small. The exact reflectivity colors (and the
    # overlay colors) are pinned into it; median-cut alone washes out small high-dBZ cores.
    from .colormaps import reflectivity
    from matplotlib.colors import to_rgb
    cmap, _ = reflectivity()
    pinned = [tuple(int(round(255 * c)) for c in cmap(i)[:3]) for i in range(cmap.N)]
    pinned += [tuple(int(round(255 * c)) for c in to_rgb(h)) for h in
               ('#ffffff', '#000000', '#ff1f1f', '#ff2a2a', '#ffb000', '#ffe600', '#ff7b72', '#0e1116', '#1b1f25')]
    pinned = list(dict.fromkeys(pinned))
    sample = Image.new('RGB', (frames[0].width, frames[0].height * 3))
    for k, f in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]):
        sample.paste(f, (0, k * frames[0].height))
    rest = sample.quantize(colors=256 - len(pinned), method=Image.MEDIANCUT).getpalette()[:3 * (256 - len(pinned))]
    flat = [v for c in pinned for v in c] + rest
    pal = Image.new('P', (1, 1)); pal.putpalette(flat + [0] * (768 - len(flat)))
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    durations = [frame_ms] * (len(q) - 1) + [end_hold_ms]
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print('saved', out, f'({len(q)} frames, {Path(out).stat().st_size / 1e6:.1f} MB)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config'); ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    ap.add_argument('--step', type=int, default=10); ap.add_argument('--product', default='base')
    ap.add_argument('--view'); ap.add_argument('--width', type=int, default=1200); ap.add_argument('--out')
    ap.add_argument('--frame-ms', type=int, default=180); ap.add_argument('--keep-frames')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg['_base_dir'] = str(Path(a.config).parent)
    outdir = Path(a.config).parent / 'output'; outdir.mkdir(exist_ok=True)
    out = a.out or outdir / f"{cfg['site']['id']}_{a.product}_loop_{parse_time(a.start).strftime('%H%M')}-" \
                            f"{parse_time(a.end).strftime('%H%M')}Z.gif"
    make_loop(cfg, a.start, a.end, out, a.step, a.product, a.view, a.width, a.frame_ms, keep_frames=a.keep_frames)


if __name__ == '__main__':
    main()
