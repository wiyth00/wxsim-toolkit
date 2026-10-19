"""Render frames [a, b) of the gust animation:  python3 anim.py a b"""
import sys, os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import stormmodel as sm
from plotkit import *

m, X, Y = sm.m, sm.X, sm.Y
d = np.load(sm.FRAMES); ft = d["ft"]; F = d["F"]
a, b = int(sys.argv[1]), int(sys.argv[2])
OUTDIR = os.path.join(sm.HERE, "frames"); os.makedirs(OUTDIR, exist_ok=True)
DPI = 100
halo = [pe.withStroke(linewidth=2.0, foreground="white")]

fig = plt.figure(figsize=(10.4, 12.2), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.035, 0.135, 0.93, 0.765])
m.drawmapboundary(fill_color="#e6ecf1", linewidth=0, ax=ax)
m.fillcontinents(color="#f5f3ee", lake_color="#e6ecf1", zorder=0.5, ax=ax)
try:
    m.drawcounties(linewidth=0.10, color="#9a9aa2", zorder=2.5, ax=ax)
except Exception as e:
    print("no counties", e)
m.drawstates(linewidth=0.55, color="#4d4d55", zorder=3, ax=ax)
m.drawcountries(linewidth=0.6, color="#4d4d55", zorder=3, ax=ax)
m.drawcoastlines(linewidth=0.6, color="#3a3a42", zorder=3, ax=ax)
m.drawparallels(np.arange(24, 38, 2), labels=[1, 0, 0, 0], linewidth=0.3, color="#9a9aa2", dashes=[1, 4], fontsize=7, zorder=1, ax=ax)
m.drawmeridians(np.arange(-92, -75, 2), labels=[0, 0, 0, 1], linewidth=0.3, color="#9a9aa2", dashes=[1, 4], fontsize=7, zorder=1, ax=ax)

# reference cities
CITIES = [("Atlanta", 33.75, -84.39, "left", 4, 2), ("Birmingham", 33.52, -86.80, "left", 4, 2), ("Montgomery", 32.37, -86.30, "left", 4, 2),
          ("Mobile", 30.69, -88.04, "right", -4, 2), ("New Orleans", 29.95, -90.07, "left", 4, 2), ("Tallahassee", 30.44, -84.28, "right", -4, 3),
          ("Jacksonville", 30.33, -81.66, "left", 4, 2), ("Savannah", 32.08, -81.09, "left", 4, 2), ("Charleston", 32.78, -79.93, "left", 4, 2),
          ("Tampa", 27.95, -82.46, "right", -4, 2), ("Orlando", 28.54, -81.38, "left", 4, 2), ("Miami", 25.76, -80.19, "left", 4, 2),
          ("Charlotte", 35.23, -80.84, "left", 4, 2)]
for nm, la, lo, ha, ox, oy in CITIES:
    x, y = m(lo, la)
    ax.plot(x, y, "o", ms=2.6, mfc=INK, mec="white", mew=0.5, zorder=6)
    ax.annotate(nm, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=6.8, color="#33333a", ha=ha, va="bottom", zorder=7, path_effects=halo)

# track
hs = ft.astype(float)
trk = sm.track_xy(hs)
ax.plot(trk[:, 0], trk[:, 1], color="#6b6b73", lw=0.9, ls=(0, (2, 3)), zorder=4)
trail, = ax.plot([], [], color=INK, lw=1.8, zorder=5, solid_capstyle="round")
star, = ax.plot([], [], marker="*", ms=19, mfc="#ffd23f", mec=INK, mew=1.0, ls="none", zorder=9)
plab = ax.annotate("", (0, 0), xytext=(14, 8), textcoords="offset points", fontsize=9.5, fontweight="bold", color=INK, ha="left", zorder=10, path_effects=halo)

# HUD
hud = ax.text(0.014, 0.988, "", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, family="DejaVu Sans Mono", color=INK, zorder=12,
              linespacing=1.55, bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="#c9c9d0", lw=0.6, alpha=0.95))

# pressure inset
ins = ax.inset_axes([0.815, 0.032, 0.17, 0.15]); ins.set_zorder(12)
ins.patch.set_facecolor("white"); ins.patch.set_alpha(0.93)
tp = np.arange(24, 114, 0.5); pp = [sm.pressure(t) for t in tp]
ins.plot(tp, pp, color=INK, lw=1.1)
ins.set_xlim(24, 112); ins.set_ylim(930, 1010)
ins.set_xticks([24, 48, 72, 96]); ins.set_xticklabels(["Wed", "Thu", "Fri", "Sat"], fontsize=6.5, color=INK)
ins.set_yticks([940, 960, 980, 1000]); ins.set_yticklabels(["940", "960", "980", "1000"], fontsize=6.5, color=INK)
ins.tick_params(length=2, width=0.4, pad=1.5)
for s in ins.spines.values():
    s.set_linewidth(0.5); s.set_color("#9a9aa2")
ins.set_title("Central pressure (mb)", fontsize=7, color=INK, pad=2)
ins.text(58, 934, "min 939", fontsize=6.3, color=ACCENT, ha="left", va="bottom")
vl = ins.axvline(30, color="#c9c9d0", lw=0.8, zorder=0)
dot, = ins.plot([], [], "o", ms=5, mfc="#c73a80", mec=INK, mew=0.8, zorder=5)

# titles, legend, colorbar
fig.text(0.04, 0.970, "Apalachee Bay Sou’easter: 3× the Original Radius", fontsize=16.5, fontweight="bold", color=INK, ha="left")
fig.text(0.04, 0.948, "3× cyclone radii · gust proxy (mph) · Southeast U.S. · Dec 16–19, 2026", fontsize=9.5, color=MUTED, ha="left")
fig.text(0.96, 0.970, "SIMULATED SCENARIO", fontsize=8.5, fontweight="bold", color=ACCENT, ha="right")
tag = fig.text(0.04, 0.917, "", fontsize=11, fontweight="bold", color=ACCENT, ha="left")
draw_cbar(fig, [0.04, 0.074, 0.46, 0.012])
fig.text(0.04, 0.099, "Gust now (mph) · unshaded below 40", fontsize=8, color=INK, ha="left")
leg_ax = fig.add_axes([0.54, 0.052, 0.43, 0.05]); leg_ax.axis("off")
handles = [Line2D([0], [0], color=INK, lw=0.9, label="58 mph (NWS severe)"),
           Line2D([0], [0], color=INK, lw=0.9, ls=(0, (1.5, 1.5)), label="74 mph (hurricane-force)"),
           Line2D([0], [0], marker="*", color="none", mfc="#ffd23f", mec=INK, ms=10, label="Low center"),
           Line2D([0], [0], color=INK, lw=1.8, label="Track so far")]
leg_ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.5, frameon=False, handlelength=2.2, columnspacing=1.6)
fig.text(0.04, 0.020, "Fictional event. Parametric gust model (asymmetric winds + mixing + sting jet + convective swaths); not observations.\n"
         "v3 surface mixing · Bergeron = latitude-adjusted deepening rate; 1 B = 24 mb/24 h at 60°N.", fontsize=6.8, color=MUTED, ha="left", va="center", linespacing=1.5)

dyn = []


def event(t):
    if t < 36: return "Cyclogenesis over the Gulf"
    if t < 52: return "Bombogenesis: explosive deepening"
    if t < 58: return "Landfall near St. Marks, Florida"
    if t < 66: return "Sting jet over south Georgia"
    if t < 96: return "Cyclonic loop over southwest Georgia"
    return "Exiting off Brunswick toward the Atlantic"


def render(i):
    for art in dyn:
        try: art.remove()
        except Exception: pass
    dyn.clear()
    t = float(ft[i]); g = F[i].astype(np.float32)
    dyn.append(ax.contourf(X, Y, g, levels=FILL_LEVELS, colors=COLORS, alpha=0.9, zorder=2))
    if g.max() > 58:
        dyn.append(ax.contour(X, Y, g, levels=[58], colors=[INK], linewidths=0.9, zorder=4))
    if g.max() > 74:
        dyn.append(ax.contour(X, Y, g, levels=[74], colors=[INK], linewidths=0.9, linestyles=[(0, (1.5, 1.5))], zorder=4))
    trail.set_data(trk[:i + 1, 0], trk[:i + 1, 1])
    star.set_data([trk[i, 0]], [trk[i, 1]])
    p = sm.pressure(t)
    plab.xy = (trk[i, 0], trk[i, 1]); plab.set_text(f"{p:.0f} mb")
    lat = sm.center_ll(t)[1]
    if t >= 36:
        dp = sm.pressure(t - 24) - p
        B = dp / 24.0 * (np.sin(np.radians(60)) / np.sin(np.radians(lat)))
        chg = f"{-dp:+.0f} mb ({B:.1f} B)" if dp > 0 else f"{-dp:+.0f} mb"
    else:
        chg = "n/a"
    area = float((g >= 58).sum()) * 9.0
    day = sm.DAY[int(t // 24)]; dom = 15 + int(t // 24); hh = int(t % 24); mm_ = int(round((t % 1) * 60))
    hud.set_text(f"{day.upper()} DEC {dom}  {hh:02d}:{mm_:02d}Z\n"
                 f"Central pressure   {p:5.0f} mb\n"
                 f"24-h change     {chg:>14}\n"
                 f"Max gust now       {g.max():5.0f} mph\n"
                 f"Gusts ≥58 mph  {area/1000:6,.0f}k km²")
    tag.set_text(event(t))
    vl.set_xdata([t, t]); dot.set_data([t], [p])
    target = os.path.join(OUTDIR, f"f_{i:04d}.png")
    temporary = target + ".tmp"
    fig.savefig(temporary, format="png", dpi=DPI, facecolor="white")
    with Image.open(temporary) as check: check.verify()
    os.replace(temporary, target)


for i in range(a, min(b, len(ft))):
    render(i)
print("done", a, b, len(ft))
