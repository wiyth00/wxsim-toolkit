"""Radar color tables."""
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Classic NWS / NEXRAD reflectivity table, 5 dBZ steps starting at 5 dBZ.
REF_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
REF_COLORS = ['#04e9e7', '#019ff4', '#0300f4', '#02fd02', '#01c501', '#008e00', '#fdf802', '#e5bc00',
              '#fd9500', '#fd0000', '#d40000', '#bc0000', '#f800fd', '#9854c6', '#fdfdfd']


def reflectivity():
    cmap = ListedColormap(REF_COLORS, name='nws_ref')
    cmap.set_under((0, 0, 0, 0)); cmap.set_bad((0, 0, 0, 0))
    return cmap, BoundaryNorm(REF_LEVELS, cmap.N)
