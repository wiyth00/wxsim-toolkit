import numpy as np
from matplotlib.patches import Rectangle

INK, MUTED, ACCENT = "#1f1f24", "#5d5d66", "#9b1c63"
LEVELS = [40, 50, 60, 70, 80, 90, 100, 110, 120]
# single-hue pink -> magenta -> plum ramp, 9 steps, light to dark
COLORS = ["#fbe3ec", "#f6c3d6", "#ee9bbd", "#e0689d", "#c73a80", "#9b1c63", "#74124d", "#520a3b", "#33052a"]
TICKS = ["40", "50", "60", "70", "80", "90", "100", "110", "120+"]
FILL_LEVELS = LEVELS + [400]


def oklab_L(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = np.cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2])
    mm = np.cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2])
    s = np.cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2])
    return 0.2104542553 * l + 0.7936177850 * mm - 0.0040720468 * s


_L = [oklab_L(c) for c in COLORS]
assert all(a > b for a, b in zip(_L, _L[1:])), _L   # sequential ramp: lightness strictly falls


def draw_cbar(fig, rect):
    cax = fig.add_axes(rect)
    n = len(COLORS)
    for i, c in enumerate(COLORS):
        cax.add_patch(Rectangle((i, 0), 1, 1, fc=c, ec="none"))
    cax.set_xlim(0, n); cax.set_ylim(0, 1)
    cax.set_xticks(np.arange(n)); cax.set_xticklabels(TICKS, fontsize=7.5, color=INK, ha="left")
    cax.set_yticks([])
    cax.tick_params(length=2, width=0.4, pad=2)
    for s in cax.spines.values():
        s.set_linewidth(0.4)
    return cax
