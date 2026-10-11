"""Coastal breakpoints and breakpoint-to-breakpoint watch/warning segments.

data/tc_breakpoints.json lists NHC-style breakpoints from the TX/Mexico border to the NC/VA border, in coastal
order. Each has the hand-entered location (`lat`, `lon`) and the point snapped onto the mainland coastline of
data/nation.geojson (`snap_lat`, `snap_lon`). A segment is the stretch of that coastline between its two
breakpoints, so warnings follow the actual shore.

Re-snap after editing the list:
    python -m wxsim.tropical.coast --snap
"""
import argparse, json
from functools import lru_cache

from shapely.geometry import Point, LineString
from shapely.ops import substring

from .. import geo

BP_FILE = geo.DATA / 'tc_breakpoints.json'


@lru_cache(maxsize=None)
def mainland_coast():
    """Exterior ring of the contiguous-US polygon, as a LineString in lon/lat."""
    land = geo.land()
    polys = list(getattr(land, 'geoms', [land]))
    main = next(p for p in polys if p.contains(Point(-84.0, 33.0)))
    return LineString(main.exterior.coords)


@lru_cache(maxsize=None)
def breakpoints():
    d = json.load(open(BP_FILE))
    return {b['name'].lower(): b for b in d['breakpoints']}


def lookup(name):
    try:
        return breakpoints()[name.lower()]
    except KeyError:
        raise KeyError(f'breakpoint "{name}" not in {BP_FILE.name}') from None


def _loc(b):
    coast = mainland_coast()
    lat, lon = b.get('snap_lat', b['lat']), b.get('snap_lon', b['lon'])
    return coast.project(Point(lon, lat))


def segment_line(frm, to):
    """Coastline LineString (lon/lat) between two breakpoint names. Takes the shorter way round the ring."""
    coast = mainland_coast()
    a, b = sorted((_loc(lookup(frm)), _loc(lookup(to))))
    inner = substring(coast, a, b)
    if (b - a) <= coast.length / 2:
        return inner
    # wrap-around (never needed for TX-NC, but keeps the function honest)
    return LineString(list(substring(coast, b, coast.length).coords) + list(substring(coast, 0, a).coords))


def segment_label(seg):
    return f"{seg['from']} to {seg['to']}"


def snap_all():
    d = json.load(open(BP_FILE))
    coast = mainland_coast()
    worst = 0
    for b in d['breakpoints']:
        p = coast.interpolate(coast.project(Point(b['lon'], b['lat'])))
        b['snap_lat'], b['snap_lon'] = round(p.y, 4), round(p.x, 4)
        moved = ((p.x - b['lon']) ** 2 + (p.y - b['lat']) ** 2) ** 0.5 * 69
        worst = max(worst, moved)
        if moved > 8:
            print(f"  note: {b['name']} moved ~{moved:.0f} mi to reach the mainland shore")
    # coastal order check: position along the ring should be monotonic
    locs = [coast.project(Point(b['snap_lon'], b['snap_lat'])) for b in d['breakpoints']]
    if not (all(x <= y for x, y in zip(locs, locs[1:])) or all(x >= y for x, y in zip(locs, locs[1:]))):
        print('  warning: breakpoints are not in coastal order along the shoreline')
    json.dump(d, open(BP_FILE, 'w'), indent=1)
    print(f'snapped {len(d["breakpoints"])} breakpoints (max move {worst:.1f} mi) -> {BP_FILE}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snap', action='store_true', help='snap every breakpoint to the coastline and save')
    a = ap.parse_args()
    if a.snap:
        snap_all()
    else:
        for b in breakpoints().values():
            print(f"{b['name']:<36} {b.get('snap_lat', b['lat']):8.3f} {b.get('snap_lon', b['lon']):9.3f}")


if __name__ == '__main__':
    main()
