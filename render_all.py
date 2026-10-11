"""Render every graphic in a scenario folder.

usage: python render_all.py scenarios/<name>

Picks up files by prefix: sounding_*.json, outlook_*.json, mcd_*.json, watch_*.json, hazards_*.json, radar_*.json, tc_*.json
(tropical cone + TCM/TCP), tcwarn_*.json (zoomed local inland-warning map),
tcwind_*.json (peak wind swath), qpf_*.json and ero_*.json (WPC rainfall map and ERO).
Outputs go to scenarios/<name>/output/.
"""
import json, sys
from pathlib import Path

from wxsim import sounding, outlook, mcd, watch, hazards
from wxsim.radar import render as radar
from wxsim import tropical
from wxsim.tropical import local as tclocal, windmap as tcwind
from wxsim.wpc import qpf as wpcqpf, ero as wpcero

RUNNERS = [('sounding_', sounding.render), ('outlook_', outlook.render), ('mcd_', mcd.render), ('watch_', watch.render),
           ('hazards_', hazards.render), ('radar_', radar.render),
           ('tc_', tropical.render),
           ('tcwarn_', tclocal.render), ('tcwind_', tcwind.render),
           ('qpf_', wpcqpf.render), ('ero_', wpcero.render)]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    scen = Path(sys.argv[1])
    out = scen / 'output'
    out.mkdir(exist_ok=True)
    for prefix, fn in RUNNERS:
        for cfg_path in sorted(scen.glob(f'{prefix}*.json')):
            print(f'== {cfg_path.name}')
            cfg = json.load(open(cfg_path))
            cfg['_base_dir'] = str(scen)
            fn(cfg, out)


if __name__ == '__main__':
    main()
