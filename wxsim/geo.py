"""Shared map helpers: boundary data, projection, polygon drawing, smoothing."""
import json
from functools import lru_cache
from pathlib import Path as FSPath

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pyproj import Transformer
from shapely.geometry import shape, Polygon, box
from shapely.ops import unary_union, transform as shp_transform

DATA = FSPath(__file__).resolve().parent.parent / 'data'

# Default city labels (name, lat, lon). Scenario files can override with "cities".
CITIES_SE = [
    ('Atlanta', 33.749, -84.388), ('Athens', 33.961, -83.378), ('Macon', 32.841, -83.632),
    ('Columbus', 32.461, -84.988), ('Augusta', 33.471, -81.975), ('Savannah', 32.081, -81.091),
    ('Birmingham', 33.519, -86.810), ('Montgomery', 32.377, -86.300), ('Huntsville', 34.730, -86.586),
    ('Chattanooga', 35.046, -85.309), ('Knoxville', 35.961, -83.921), ('Nashville', 36.163, -86.781),
    ('Greenville', 34.853, -82.394), ('Columbia', 34.001, -81.035), ('Charlotte', 35.227, -80.843),
    ('Asheville', 35.595, -82.551), ('Tallahassee', 30.438, -84.281), ('Raleigh', 35.780, -78.639),
    ('Albany', 31.578, -84.156), ('Gainesville', 34.298, -83.824), ('LaGrange', 33.039, -85.031),
    ('Rome', 34.257, -85.165), ('Jacksonville', 30.332, -81.656), ('Louisville', 38.253, -85.759),
    ('Lexington', 38.040, -84.503), ('Charleston', 32.776, -79.931), ('Mobile', 30.695, -88.040)]


def transformer(lat_0=34.0, lon_0=-84.0):
    return Transformer.from_crs(
        'EPSG:4326', f'+proj=lcc +lat_1=33 +lat_2=45 +lat_0={lat_0} +lon_0={lon_0} +units=m', always_xy=True)


def projector(tr):
    return lambda g: shp_transform(lambda x, y, z=None: tr.transform(x, y), g)


@lru_cache(maxsize=None)
def _load(name):
    return json.load(open(DATA / f'{name}.geojson'))['features']


def states():
    return _load('states')


def counties():
    return _load('counties')


@lru_cache(maxsize=None)
def land():
    return shape(_load('nation')[0]['geometry']).buffer(0)


def chaikin(pts, n=4):
    """Corner-cutting smoother for hand-drawn outlook/risk polygons (closed rings)."""
    pts = np.asarray(pts, float)
    for _ in range(n):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, axis=0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, axis=0)
        pts = np.empty((2 * len(pts), 2)); pts[0::2] = q; pts[1::2] = r
    return pts


def nest(raw_list, clip):
    """Smooth each ring (outer -> inner) and force every inner area inside its outer one."""
    gs = [Polygon(chaikin(r)) for r in raw_list]
    for i in range(len(gs) - 2, -1, -1):
        gs[i] = unary_union([gs[i], gs[i + 1].buffer(0.12)])
    return [g.intersection(clip) for g in gs]


def draw(ax, g, proj, **kw):
    """Draw a shapely (Multi)Polygon in projected coordinates."""
    g = proj(g)
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, 'geoms', []))
    for p in polys:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            c = np.asarray(ring.coords)
            verts += c.tolist(); codes += [Path.MOVETO] + [Path.LINETO] * (len(c) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), **kw))


def basemap(ax, proj, ext, *, land_color='#fbfbf8', ocean='#dce7ef', county_box=None,
            county_color='#8d969e', county_lw=0.25, county_alpha=0.5, state_color='#4a4f55', state_lw=1.0,
            z_land=1, z_county=9, z_state=10, draw_land=True):
    if ocean is not None:
        ax.set_facecolor(ocean)
    if draw_land:
        draw(ax, land().intersection(ext), proj, facecolor=land_color, edgecolor='none', zorder=z_land)
    cb = county_box if county_box is not None else ext
    for f in counties():
        g = shape(f['geometry']).buffer(0)
        if g.intersects(cb):
            draw(ax, g.intersection(ext), proj, facecolor='none', edgecolor=county_color, lw=county_lw,
                 alpha=county_alpha, zorder=z_county)
    for f in states():
        g = shape(f['geometry']).buffer(0).intersection(ext)
        if not g.is_empty:
            draw(ax, g, proj, facecolor='none', edgecolor=state_color, lw=state_lw, zorder=z_state)


def set_extent(ax, tr, lon0, lon1, lat0, lat1, inner=False):
    """inner=True crops to the largest box fully inside the lat/lon corners (no blank corners)."""
    x0, y0 = tr.transform(lon0, lat0); x1, y1 = tr.transform(lon1, lat1)
    xa, ya = tr.transform(lon0, lat1); xb, yb = tr.transform(lon1, lat0)
    if inner:
        ax.set_xlim(max(x0, xa), min(x1, xb)); ax.set_ylim(max(y0, yb), min(y1, ya))
    else:
        ax.set_xlim(min(x0, xa), max(x1, xb)); ax.set_ylim(min(y0, yb), max(y1, ya))
    ax.set_xticks([]); ax.set_yticks([])


def extent_box(lon0, lon1, lat0, lat1, pad=1.0):
    return box(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
