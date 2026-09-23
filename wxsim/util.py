"""Small helpers shared by the product generators."""
import math


def latlon_code(lat, lon):
    """SPC 8-digit point: lat*100 then |lon|*100 with the leading 1 dropped west of 100W."""
    lo = int(round(abs(lon) * 100))
    if lo >= 10000:
        lo -= 10000
    return f'{int(round(lat * 100)):04d}{lo:04d}'


def latlon_string(points, per_line=6):
    """points: [[lat, lon], ...]; closes the polygon if needed. Returns the formatted LAT...LON block."""
    pts = [tuple(p) for p in points]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    codes = [latlon_code(*p) for p in pts]
    lines = [' '.join(codes[i:i + per_line]) for i in range(0, len(codes), per_line)]
    return 'LAT...LON   ' + ('\n' + ' ' * 12).join(lines)


def offset_point(lat, lon, miles, bearing_deg):
    """Point `miles` statute miles from (lat, lon) along a compass bearing (flat-earth approximation)."""
    d = miles * 1.609344
    dlat = d * math.cos(math.radians(bearing_deg)) / 111.0
    dlon = d * math.sin(math.radians(bearing_deg)) / (111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


COMPASS = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
           'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}


DIR_WORDS = {'N': 'north', 'NNE': 'north northeast', 'NE': 'northeast', 'ENE': 'east northeast', 'E': 'east',
             'ESE': 'east southeast', 'SE': 'southeast', 'SSE': 'south southeast', 'S': 'south',
             'SSW': 'south southwest', 'SW': 'southwest', 'WSW': 'west southwest', 'W': 'west',
             'WNW': 'west northwest', 'NW': 'northwest', 'NNW': 'north northwest'}
