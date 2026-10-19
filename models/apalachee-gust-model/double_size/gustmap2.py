import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import stormmodel as gm
from station_catalog import ST
from stations import station_reports
from plotkit import *

m, X, Y, G, T = gm.m, gm.X, gm.Y, gm.G, gm.T
OUT = "apalachee_gust_map_size2x.png"

fig = plt.figure(figsize=(10.4, 12.2), dpi=200, facecolor="white")
ax = fig.add_axes([0.035, 0.135, 0.93, 0.765])
m.drawmapboundary(fill_color="#e6ecf1", linewidth=0, ax=ax)
m.fillcontinents(color="#f5f3ee", lake_color="#e6ecf1", zorder=0.5, ax=ax)
ax.contourf(X, Y, G, levels=FILL_LEVELS, colors=COLORS, alpha=0.9, zorder=2)
try:
    m.drawcounties(linewidth=0.10, color="#9a9aa2", zorder=2.5, ax=ax)
except Exception as e:
    print("no counties:", e)
m.drawstates(linewidth=0.55, color="#4d4d55", zorder=3, ax=ax)
m.drawcountries(linewidth=0.6, color="#4d4d55", zorder=3, ax=ax)
m.drawcoastlines(linewidth=0.6, color="#3a3a42", zorder=3, ax=ax)
m.drawparallels(np.arange(24, 38, 2), labels=[1, 0, 0, 0], linewidth=0.3, color="#9a9aa2", dashes=[1, 4], fontsize=7, zorder=1, ax=ax)
m.drawmeridians(np.arange(-92, -75, 2), labels=[0, 0, 0, 1], linewidth=0.3, color="#9a9aa2", dashes=[1, 4], fontsize=7, zorder=1, ax=ax)
ax.contour(X, Y, G, levels=[58], colors=[INK], linewidths=0.9, zorder=4)
ax.contour(X, Y, G, levels=[74], colors=[INK], linewidths=0.9, linestyles=[(0, (1.5, 1.5))], zorder=4)

# track
hrs = np.arange(12, 115, 1.0)
tr = gm.track_xy(hrs)
ax.plot(tr[:, 0], tr[:, 1], color=INK, lw=1.4, zorder=5, solid_capstyle="round")
for h in range(18, 115, 6):
    x, y = gm.center_xy(h)
    ax.plot(x, y, "o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=6)
lx, ly = gm.center_xy(54)
ax.plot(lx, ly, marker="*", ms=14, mfc="#ffd23f", mec=INK, mew=0.9, zorder=7)

MARQUEE = {"KTLH","KATL","KJAX","KSAV","KCHS","KMCO","KTPA","KMIA","KBHM","KMOB","KCLT","KVLD","KMGM","KAGS","KMGR","KSSI","KBQK"}
reports = {r["id"]: r for r in station_reports()}
rows = []
for sid, name, lat, lon in ST:
    x, y = m(lon, lat)
    if not (0 < x < m.xmax and 0 < y < m.ymax):
        continue
    r = reports[sid]
    rows.append(dict(id=sid, name=name, x=x, y=y, g=int(round(r["reported_gust_mph"])), t=r["reported_peak_h"], outage=r["outage"], rank_g=r["reported_gust_mph"]))
rows.sort(key=lambda r: -r["rank_g"])

fig.canvas.draw()
FS = 6.8
px_per_pt = fig.dpi / 72.0
halo = [pe.withStroke(linewidth=2.2, foreground="white")]
axbb = ax.get_window_extent()
placed = []
def overlaps(b):
    return any(not (b[2] < p[0] or b[0] > p[2] or b[3] < p[1] or b[1] > p[3]) for p in placed)

panel = (0.008, 0.765, 0.270, 0.992)
pb = (axbb.x0 + panel[0] * axbb.width, axbb.y0 + panel[1] * axbb.height,
      axbb.x0 + panel[2] * axbb.width, axbb.y0 + panel[3] * axbb.height)
placed.append(pb)
for r in rows:
    d = ax.transData.transform((r["x"], r["y"])); r["d"] = d
    placed.append((d[0] - 7, d[1] - 7, d[0] + 7, d[1] + 7))
for d in ax.transData.transform(tr)[::2]:
    placed.append((d[0] - 6, d[1] - 6, d[0] + 6, d[1] + 6))

TL = [(30, (14, -22)), (42, (14, -22)), (54, (-170, -70)), (78, (-70, 4)), (102, (8, -34))]
for h, off in TL:
    x, y = gm.center_xy(h)
    txt = f"{gm.hlabel(h)}\n{gm.pressure(h):.0f} mb"
    if h == 54:
        txt = f"Landfall near St. Marks\n{gm.hlabel(h)} · {gm.pressure(h):.0f} mb"
    ha = "left" if off[0] >= 0 else "right"
    t = ax.annotate(txt, (x, y), xytext=off, textcoords="offset pixels", fontsize=6.8, color=INK, ha=ha, va="center",
                    fontweight="bold" if h == 54 else "normal", zorder=9, path_effects=halo)
    fig.canvas.draw()
    bb = t.get_window_extent(); placed.append((bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2))

CANDS = [("left", "center", 9, 0), ("right", "center", -9, 0), ("center", "bottom", 0, 8), ("center", "top", 0, -8),
         ("left", "bottom", 7, 6), ("right", "bottom", -7, 6), ("left", "top", 7, -6), ("right", "top", -7, -6)]
skipped = []
for r in sorted(rows, key=lambda r: (r["id"] not in MARQUEE, -r["g"])):
    txt = f'{r["id"][1:]} {r["g"]}' if r["id"] in MARQUEE else f'{r["g"]}'
    w = len(txt) * 0.60 * FS * px_per_pt + 4; h = 1.25 * FS * px_per_pt + 2
    ok = False
    for scale in (1.0, 1.9, 3.2, 4.5):
        for ha, va, ox, oy in CANDS:
            ox2, oy2 = ox * scale, oy * scale
            cx, cy = r["d"][0] + ox2, r["d"][1] + oy2
            x0 = cx if ha == "left" else (cx - w if ha == "right" else cx - w / 2)
            y0 = cy if va == "bottom" else (cy - h if va == "top" else cy - h / 2)
            b = (x0, y0, x0 + w, y0 + h)
            own = (r["d"][0] - 7, r["d"][1] - 7, r["d"][0] + 7, r["d"][1] + 7)
            placed.remove(own); clash = overlaps(b)
            inside = b[0] > axbb.x0 + 4 and b[2] < axbb.x1 - 4 and b[1] > axbb.y0 + 4 and b[3] < axbb.y1 - 4
            placed.append(own)
            if not clash and inside:
                placed.append(b)
                ax.annotate(txt, (r["x"], r["y"]), xytext=(ox2, oy2), textcoords="offset pixels", ha=ha, va=va,
                            fontsize=FS, color=INK, fontweight="bold" if r["g"] >= 74 else "normal", zorder=10, path_effects=halo)
                ok = True; break
        if ok: break
    if not ok: skipped.append(r["id"])
print("unlabeled stations:", skipped)
ax.scatter([r["x"] for r in rows], [r["y"] for r in rows], s=9, c="white", edgecolors=INK, linewidths=0.6, zorder=8)

top = rows[:8]
lines = ["SYNTHETIC REPORTS (mph)"] + [f'{r["id"][1:]}  {r["name"][:13]:<13} {r["g"]:>3}  {gm.hlabel(r["t"])}' for r in top]
ax.text(panel[0] + 0.010, panel[3] - 0.010, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
        family="DejaVu Sans Mono", color=INK, zorder=12, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c9c9d0", lw=0.6, alpha=0.95))

fig.text(0.04, 0.970, "Apalachee Bay Sou’easter: Double the Size", fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.04, 0.948, "Southeast U.S. · Dec 15–19, 2026 · maximum gust at each point · 2× cyclone radii · minimum 939 mb", fontsize=9.5, color=MUTED, ha="left")
fig.text(0.96, 0.970, "SIMULATED SCENARIO", fontsize=8.5, fontweight="bold", color=ACCENT, ha="right")
draw_cbar(fig, [0.04, 0.074, 0.46, 0.012])
fig.text(0.04, 0.099, "Peak gust (mph) · unshaded below 40", fontsize=8, color=INK, ha="left")
leg_ax = fig.add_axes([0.54, 0.052, 0.43, 0.05]); leg_ax.axis("off")
handles = [Line2D([0], [0], color=INK, lw=0.9, label="58 mph (NWS severe)"),
           Line2D([0], [0], color=INK, lw=0.9, ls=(0, (1.5, 1.5)), label="74 mph (hurricane-force)"),
           Line2D([0], [0], color=INK, lw=1.4, marker="o", ms=3.2, mfc="white", label="Low track, 6-hourly"),
           Line2D([0], [0], marker="o", color="none", mfc="white", mec=INK, ms=4, label="Synthetic station gust")]
leg_ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.5, frameon=False, handlelength=2.2, columnspacing=1.6)
fig.text(0.04, 0.020, "Fictional event. The gust field is a parametric model (asymmetric winds + mixing + sting jet + convective swaths);\n"
         "synthetic station gusts include exposure / turbulence / possible outages. Bold = gusts 74+ mph.", fontsize=6.8, color=MUTED, ha="left", va="center", linespacing=1.5)
fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
print("top:", [(r["id"], r["g"], gm.hlabel(r["t"])) for r in top])
print("grid max", G.max().round(1))
