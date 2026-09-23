"""Render every graphic in a scenario folder.

usage: python render_all.py scenarios/<name>

Picks up files by prefix: sounding_*.json, outlook_*.json, mcd_*.json, watch_*.json.
Outputs go to scenarios/<name>/output/.
"""
import json, sys
from pathlib import Path

from wxsim import sounding, outlook, mcd, watch

RUNNERS = [('sounding_', sounding.render), ('outlook_', outlook.render), ('mcd_', mcd.render), ('watch_', watch.render)]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    scen = Path(sys.argv[1])
    out = scen / 'output'
    out.mkdir(exist_ok=True)
    for prefix, fn in RUNNERS:
        for cfg_path in sorted(scen.glob(f'{prefix}*.json')):
            print(f'== {cfg_path.name}')
            fn(json.load(open(cfg_path)), out)


if __name__ == '__main__':
    main()
