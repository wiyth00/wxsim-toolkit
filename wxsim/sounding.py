"""Skew-T / hodograph / parameter panel from a scenario sounding JSON.

usage: python -m wxsim.sounding scenarios/<name>/sounding_XXX.json [--outdir DIR]
"""
import argparse, json, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from metpy.plots import SkewT, Hodograph
import metpy.calc as mpcalc
from metpy.units import units

BG = '#0b0e14'; PANEL = '#121722'; FG = '#d8dee9'; MUTED = '#7a8499'; GRID = '#2a3242'
COL = dict(red='#ff4d4d', green='#35d07f', cyan='#4fd1ff', white='#f2f2f2', gold='#f2c14e',
           mag='#ff5fd2', muted=MUTED, fg=FG, fzl='#7fa6e0')


class Sounding:
    def __init__(self, cfg):
        a = np.array(cfg['profile'], dtype=float)
        self.elev = cfg['elevation_m']
        self.p = a[:, 0] * units.hPa
        self.T = a[:, 1] * units.degC
        self.Td = a[:, 2] * units.degC
        self.u, self.v = mpcalc.wind_components(a[:, 4] * units.knots, a[:, 3] * units.deg)
        h = [self.elev]
        for i in range(1, len(a)):
            sl = slice(i - 1, i + 1)
            w = mpcalc.mixing_ratio_from_relative_humidity(
                self.p[sl], self.T[sl], mpcalc.relative_humidity_from_dewpoint(self.T[sl], self.Td[sl]))
            h.append(h[-1] + mpcalc.thickness_hydrostatic(self.p[sl], self.T[sl], mixing_ratio=w).to('m').m)
        self.z_msl = np.array(h) * units.m
        self.z_agl = self.z_msl - self.elev * units.m

    # ---- helpers
    def interp_p(self, arr, pres):
        return np.interp(np.log(pres), np.log(self.p.m[::-1]), np.asarray(arr)[::-1])

    def p_at_z(self, z_agl):
        return float(np.exp(np.interp(z_agl, self.z_agl.m, np.log(self.p.m))))

    def z_at_p(self, pres):
        return float(self.interp_p(self.z_agl.m, pres))

    def lapse_p(self, p1, p2, Tprof=None):
        Tprof = self.T if Tprof is None else Tprof
        t1, t2 = self.interp_p(Tprof.m, p1), self.interp_p(Tprof.m, p2)
        z1, z2 = self.interp_p(self.z_msl.m, p1), self.interp_p(self.z_msl.m, p2)
        return (t1 - t2) / ((z2 - z1) / 1000.)

    def lapse_z(self, z1, z2, Tprof):
        return (np.interp(z1, self.z_agl.m, Tprof.m) - np.interp(z2, self.z_agl.m, Tprof.m)) / ((z2 - z1) / 1e3)

    def zero_level(self, prof):
        prof = np.asarray(prof)
        for i in range(len(prof) - 1):
            if prof[i] > 0 >= prof[i + 1]:
                f = prof[i] / (prof[i] - prof[i + 1])
                p = np.exp(np.log(self.p.m[i]) + f * (np.log(self.p.m[i + 1]) - np.log(self.p.m[i])))
                return p, self.z_at_p(p)
        return np.nan, np.nan

    def p_at_T(self, target):
        for i in range(len(self.T) - 1):
            if self.T.m[i] > target >= self.T.m[i + 1]:
                f = (self.T.m[i] - target) / (self.T.m[i] - self.T.m[i + 1])
                return np.exp(np.log(self.p.m[i]) + f * (np.log(self.p.m[i + 1]) - np.log(self.p.m[i])))
        return np.nan

    def tropopause(self):
        """Lowest level above 500 mb where the 2 km mean lapse rate drops below 2 C/km (WMO-ish)."""
        for i in range(len(self.p)):
            if self.p.m[i] > 500:
                continue
            z0 = self.z_agl.m[i]
            lr = self.lapse_z(z0, z0 + 2000, self.T) if z0 + 2000 < self.z_agl.m[-1] else 99
            if lr < 2.0:
                return self.p.m[i], self.T.m[i]
        return np.nan, np.nan

    def modified(self, sfcT_C, sfcTd_C):
        """Well-mixed boundary layer from a new surface T/Td (dry adiabat + constant mixing ratio)."""
        p, T, Td = self.p, self.T, self.Td
        th = mpcalc.potential_temperature(p[0], sfcT_C * units.degC)
        Tdry = mpcalc.temperature_from_potential_temperature(p, th).to('degC')
        mix = Tdry > T
        top = np.argmin(mix) if not mix.all() else len(mix)
        Tm = T.m.copy(); Tm[:top] = Tdry.m[:top]
        w = mpcalc.mixing_ratio_from_relative_humidity(
            p[0], sfcT_C * units.degC, mpcalc.relative_humidity_from_dewpoint(sfcT_C * units.degC, sfcTd_C * units.degC))
        Tdm = Td.m.copy()
        for i in range(top):
            Tdm[i] = mpcalc.dewpoint(mpcalc.vapor_pressure(p[i], w)).to('degC').m
        return Tm * units.degC, Tdm * units.degC, top

    def convective_temp(self, tol=0.2, ptop=500):
        """Lowest surface temperature (surface moisture unchanged) whose surface parcel is nowhere
        virtually colder than the environment between the surface and `ptop` mb -- i.e. the cap is
        removed by heating alone (virtual-temperature correction applied, as in CIN)."""
        p = self.p
        above = (p.m < p.m[0]) & (p.m >= ptop)
        Tv_env = mpcalc.virtual_temperature_from_dewpoint(p, self.T, self.Td).to('degC').m
        for t in np.arange(self.T.m[0], self.T.m[0] + 25, 0.1):
            prof = mpcalc.parcel_profile(p, t * units.degC, self.Td[0])
            lcl_p, _ = mpcalc.lcl(p[0], t * units.degC, self.Td[0])
            w_sfc = mpcalc.saturation_mixing_ratio(p[0], self.Td[0])
            w_par = np.where(p.m >= lcl_p.m, w_sfc.m, mpcalc.saturation_mixing_ratio(p, prof).m)
            Tv_par = mpcalc.virtual_temperature(prof, w_par * units('kg/kg')).to('degC').m
            if np.min(Tv_par[above] - Tv_env[above]) >= -tol:
                return t
        return np.nan

    def params(self, Tm=None, Tdm=None):
        Tm = self.T if Tm is None else Tm
        Tdm = self.Td if Tdm is None else Tdm
        p = self.p
        o = {'Tm': Tm, 'Tdm': Tdm}
        o['sb_prof'] = mpcalc.parcel_profile(p, Tm[0], Tdm[0]).to('degC')
        o['SBCAPE'], o['SBCIN'] = mpcalc.surface_based_cape_cin(p, Tm, Tdm)
        o['MLCAPE'], o['MLCIN'] = mpcalc.mixed_layer_cape_cin(p, Tm, Tdm, depth=100 * units.hPa)
        o['MUCAPE'], o['MUCIN'] = mpcalc.most_unstable_cape_cin(p, Tm, Tdm)
        mlp, mlT, mlTd = mpcalc.mixed_parcel(p, Tm, Tdm, depth=100 * units.hPa)
        o['ML_LCL_p'], _ = mpcalc.lcl(mlp, mlT, mlTd)
        o['ML_LCL_m'] = self.z_at_p(o['ML_LCL_p'].m)
        o['PW'] = mpcalc.precipitable_water(p, Tdm).to('inch')
        o['DCAPE'] = mpcalc.downdraft_cape(p, Tm, Tdm)[0]
        o['LR03'] = self.lapse_z(0, 3000, Tm)
        o['LR75'] = self.lapse_p(700, 500, Tm)
        mu = mpcalc.most_unstable_parcel(p, Tm, Tdm)
        w = mpcalc.mixing_ratio_from_relative_humidity(
            mu[0], mu[1], mpcalc.relative_humidity_from_dewpoint(mu[1], mu[2])).to('g/kg').m
        t500 = self.interp_p(Tm.m, 500)
        o['SHIP'] = (o['MUCAPE'].m * w * o['LR75'] * (-t500) * self.kin()['SHR06'].to('m/s').m / 42e6)
        return o

    def kin(self):
        if hasattr(self, '_kin'):
            return self._kin
        k = {}
        rm, lm, mean = mpcalc.bunkers_storm_motion(self.p, self.u, self.v, self.z_msl)
        k['RM'], k['LM'] = rm, lm
        for lbl, d in [('01', 1000), ('03', 3000)]:
            _, _, tot = mpcalc.storm_relative_helicity(self.z_agl, self.u, self.v, d * units.m,
                                                       storm_u=rm[0], storm_v=rm[1])
            k['SRH' + lbl] = tot
        for lbl, d in [('01', 1000), ('06', 6000), ('08', 8000)]:
            su, sv = mpcalc.bulk_shear(self.p, self.u, self.v, height=self.z_agl, depth=d * units.m)
            k['SHR' + lbl] = mpcalc.wind_speed(su, sv).to('knots')
        self._kin = k
        return k


def f2c(f):
    return (f - 32) * 5 / 9


def vec_str(vec):
    d = mpcalc.wind_direction(vec[0] * units.kt, vec[1] * units.kt).m
    return f'{float(d):03.0f}/{float(np.hypot(*vec)):02.0f}'


def write_sharppy(snd, cfg, path):
    lines = ['%TITLE%', cfg.get('sharppy_title', f" {cfg['station'][-3:]}   {cfg['time_label']}"), '',
             '   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD',
             '-------------------------------------------------------------------', '%RAW%']
    for row, z in zip(cfg['profile'], snd.z_msl.m):
        pp, t, td, wd, ws = row
        lines.append(f'{pp:8.2f}, {z:9.2f}, {t:8.2f}, {td:8.2f}, {wd:8.2f}, {ws:8.2f}')
    lines.append('%END%')
    Path(path).write_text('\n'.join(lines) + '\n')


def render(cfg, outdir):
    with plt.rc_context({'font.family': 'DejaVu Sans Mono', 'text.color': FG, 'axes.labelcolor': FG,
                         'xtick.color': MUTED, 'ytick.color': MUTED}):
        _render(cfg, outdir)


def _render(cfg, outdir):
    snd = Sounding(cfg)
    p, T, Td = snd.p, snd.T, snd.Td
    obs = snd.params()
    ms = cfg.get('modified_surface')
    if ms:
        Tm, Tdm, top = snd.modified(f2c(ms['T_F']), f2c(ms['Td_F']))
        mod = snd.params(Tm, Tdm)
    kin = snd.kin()
    Tw = mpcalc.wet_bulb_temperature(p, T, Td).to('degC')
    wbz_p, wbz_z = snd.zero_level(Tw.m)
    fzl_p, fzl_z = snd.zero_level(T.m)
    trop_p, trop_T = snd.tropopause()
    tc_c = snd.convective_temp()
    tokens = dict(wbz_kft=wbz_z * 3.28084 / 1000, fzl_kft=fzl_z * 3.28084 / 1000, lr75=obs['LR75'],
                  tc_f=tc_c * 9 / 5 + 32, trop_p=trop_p)

    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    gs = GridSpec(12, 20, figure=fig, left=0.045, right=0.985, top=0.93, bottom=0.05, wspace=0.6, hspace=0.9)
    skew = SkewT(fig, rotation=45, subplot=gs[:, :12])
    ax = skew.ax
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_ylim(1000, 100); ax.set_xlim(*cfg.get('xlim', [-40, 45]))
    skew.plot_dry_adiabats(colors='#5a4636', alpha=0.45, linewidths=0.7)
    skew.plot_moist_adiabats(colors='#2f5a45', alpha=0.45, linewidths=0.7)
    skew.plot_mixing_lines(colors='#3a4f6b', alpha=0.55, linewidths=0.7, linestyles=':')
    ax.grid(color=GRID, linewidth=0.6)
    ax.axvline(0, color='#4a6fa5', lw=1.1, ls='--', alpha=0.8)
    for iso in (-20, -30):
        ax.axvline(iso, color='#4a6fa5', lw=0.8, ls=':', alpha=0.6)
    p10, p30 = snd.p_at_T(-10), snd.p_at_T(-30)
    ax.axhspan(p10, p30, color=COL['cyan'], alpha=0.06, zorder=0)

    if ms:
        skew.plot(p[:top + 1], Tm[:top + 1], color=COL['red'], lw=1.6, ls=(0, (4, 3)), alpha=0.8)
        skew.plot(p[:top + 1], Tdm[:top + 1], color=COL['green'], lw=1.6, ls=(0, (4, 3)), alpha=0.8)
        skew.shade_cape(p, Tm, mod['sb_prof'], facecolor=COL['red'], alpha=0.13)
        skew.shade_cin(p, Tm, mod['sb_prof'], Tdm, facecolor='#4f7cff', alpha=0.18)
        skew.plot(p, mod['sb_prof'], color=COL['white'], lw=1.4, ls='--',
                  label=f"{ms.get('label', 'MOD')} modified parcel ({ms['T_F']}/{ms['Td_F']})")
    else:
        skew.shade_cape(p, T, obs['sb_prof'], facecolor=COL['red'], alpha=0.13)
        skew.shade_cin(p, T, obs['sb_prof'], Td, facecolor='#4f7cff', alpha=0.18)
    skew.plot(p, obs['sb_prof'], color=MUTED, lw=1.0, ls=':', label=f"{cfg.get('obs_label', 'OBS')} surface parcel")
    skew.plot(p, Tw, color=COL['cyan'], lw=0.9, alpha=0.85)
    skew.plot(p, T, color=COL['red'], lw=2.6)
    skew.plot(p, Td, color=COL['green'], lw=2.6)

    bl = cfg.get('barb_levels')
    idx = [i for i, pp in enumerate(p.m) if (bl is None or pp in bl)]
    skew.plot_barbs(p[idx], snd.u[idx], snd.v[idx], xloc=0.965, color=FG, length=6.3, linewidth=0.9,
                    sizes=dict(emptybarb=0.05))
    YT = ax.get_yaxis_transform()
    for km in [1, 2, 3, 4, 5, 6, 9, 12, 15]:
        if km * 1000 > snd.z_agl.m[-1]:
            continue
        pk = snd.p_at_z(km * 1000)
        ax.text(0.012, pk, f'{km} km', transform=YT, color=MUTED, fontsize=8, va='center')
        ax.plot([0, 0.008], [pk, pk], transform=YT, color=MUTED, lw=0.8)

    for a in cfg.get('annotations', []):
        pa = {'wbz': wbz_p, 'fzl': fzl_p, 'trop': trop_p}.get(a['p'], a['p'])
        if isinstance(pa, str) or not np.isfinite(pa):
            continue
        prof = {'wbz': Tw.m, 'fzl': T.m}.get(a['p'], T.m)
        xT = a.get('T', None)
        xT = float(snd.interp_p(prof, pa)) if xT is None else xT
        side = a.get('side', 'right')
        xt = a.get('xtext', 0.765 if side == 'right' else 0.40)
        color = COL.get(a.get('color', 'fg'), a.get('color', FG))
        ax.annotate(a['text'].format(**tokens), xy=(xT, pa), xycoords='data', xytext=(xt, pa), textcoords=YT,
                    color=color, fontsize=a.get('size', 9), ha='left' if side == 'right' else 'right',
                    va='center', arrowprops=dict(arrowstyle='-', color=color, lw=0.7, alpha=0.6, shrinkA=2, shrinkB=2))
    ax.text(0.075, (p10 * p30) ** 0.5 * 1.02, 'hail growth zone\n(−10 to −30°C)', transform=YT,
            color=COL['cyan'], fontsize=8, alpha=0.8, va='center')
    ax.set_xlabel('Temperature (°C)', fontsize=10); ax.set_ylabel('Pressure (hPa)', fontsize=10)
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.06, 0.995), fontsize=8.5, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)

    # hodograph
    axh = fig.add_subplot(gs[0:6, 12:20], anchor='S')
    axh.set_facecolor(PANEL)
    for s in axh.spines.values():
        s.set_color(GRID)
    h = Hodograph(axh, component_range=140)
    h.add_grid(increment=20, color=GRID, linewidth=0.8)
    zi = np.arange(0, 12001, 50)
    ui = np.interp(zi, snd.z_agl.m, snd.u.to('kt').m); vi = np.interp(zi, snd.z_agl.m, snd.v.to('kt').m)
    for (lo, hi), c in zip([(0, 1000), (1000, 3000), (3000, 6000), (6000, 9000), (9000, 12000)],
                           [COL['mag'], COL['red'], COL['green'], COL['gold'], COL['cyan']]):
        m = (zi >= lo) & (zi <= hi)
        axh.plot(ui[m], vi[m], color=c, lw=2.4, solid_capstyle='round')
    for km in [1, 3, 6, 9]:
        j = np.argmin(abs(zi - km * 1000))
        axh.plot(ui[j], vi[j], 'o', ms=3.5, color=FG); axh.text(ui[j] + 2, vi[j] - 4, str(km), color=FG, fontsize=8)
    rm, lm = kin['RM'].to('kt').m, kin['LM'].to('kt').m
    axh.plot(*rm, marker='o', ms=7, mfc='none', mec=COL['red'], mew=1.6)
    axh.text(rm[0] + 3, rm[1] - 3, 'RM', color=COL['red'], fontsize=8.5)
    axh.plot(*lm, marker='o', ms=7, mfc='none', mec=COL['cyan'], mew=1.6)
    axh.text(lm[0] + 3, lm[1] + 2, 'LM', color=COL['cyan'], fontsize=8.5)
    m01 = zi <= 1000
    axh.fill(np.r_[rm[0], ui[m01]], np.r_[rm[1], vi[m01]], color=COL['mag'], alpha=0.18)
    hx = cfg.get('hodo_xlim', [-25, 125]); hy = cfg.get('hodo_ylim', [-45, 85])
    axh.set_xlim(*hx); axh.set_ylim(*hy); axh.set_aspect('equal', adjustable='box')
    axh.tick_params(labelsize=7)
    for r in (20, 40, 60, 80, 100, 120):
        axh.text(0, -r + 1.5, f'{r}', color=MUTED, fontsize=7, ha='center')
    axh.set_title('Hodograph (kt)   0–1 / 1–3 / 3–6 / 6–9 / 9–12 km', color=FG, fontsize=9, loc='left')

    # parameter panel
    axt = fig.add_subplot(gs[6:12, 12:20])
    axt.set_facecolor(PANEL); axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values():
        s.set_color(GRID)
    f0 = lambda q, nd=0: f'{float(np.atleast_1d(q.m if hasattr(q, "m") else q)[0]):.{nd}f}'
    cols = [obs] + ([mod] if ms else [])
    head = [cfg.get('obs_label', 'OBS')] + ([ms.get('label', 'MOD')] if ms else [])
    rows = [('', *head),
            ('SBCAPE / CIN (J/kg)', *[f"{f0(o['SBCAPE'])} / {f0(o['SBCIN'])}" for o in cols]),
            ('MLCAPE / CIN (J/kg)', *[f"{f0(o['MLCAPE'])} / {f0(o['MLCIN'])}" for o in cols]),
            ('MUCAPE / CIN (J/kg)', *[f"{f0(o['MUCAPE'])} / {f0(o['MUCIN'])}" for o in cols]),
            ('ML LCL (m AGL)', *[f"{o['ML_LCL_m']:.0f}" for o in cols]),
            ('DCAPE (J/kg)', *[f0(o['DCAPE']) for o in cols]),
            ('0–3 km LR (°C/km)', *[f"{o['LR03']:.1f}" for o in cols]),
            ('700–500 LR (°C/km)', *[f"{o['LR75']:.1f}" for o in cols]),
            ('PW (in)', *[f0(o['PW'], 2) for o in cols]),
            ('SHIP', *[f"{o['SHIP']:.1f}" for o in cols])]
    y = 0.95
    axt.text(0.03, y + 0.005, 'THERMODYNAMICS', color=COL['gold'], fontsize=9.5, transform=axt.transAxes, weight='bold')
    y -= 0.07
    for i, r in enumerate(rows):
        axt.text(0.03, y, r[0], color=MUTED, fontsize=9.3, transform=axt.transAxes)
        xs = [0.97] if len(r) == 2 else [0.66, 0.97]
        for j, (x, val) in enumerate(zip(xs, r[1:])):
            last = j == len(xs) - 1
            axt.text(x, y, val, color=(MUTED if i == 0 else (COL['white'] if last and ms else FG)), fontsize=9.3,
                     transform=axt.transAxes, ha='right', weight='bold' if (i == 0 or (last and ms)) else 'normal')
        y -= 0.057
    y -= 0.01
    axt.text(0.03, y, f"KINEMATICS ({cfg.get('obs_label', 'OBS')})", color=COL['gold'], fontsize=9.5,
             transform=axt.transAxes, weight='bold')
    y -= 0.065
    for a, b in [('0–1 / 0–3 km SRH (m²/s²)', f"{kin['SRH01'].m:.0f} / {kin['SRH03'].m:.0f}"),
                 ('0–1 / 0–6 / 0–8 km shear (kt)', f"{kin['SHR01'].m:.0f} / {kin['SHR06'].m:.0f} / {kin['SHR08'].m:.0f}"),
                 ('Bunkers RM / LM', f"{vec_str(rm)} / {vec_str(lm)}"),
                 ('Conv. temp  |  WBZ  |  FZL', f"{tokens['tc_f']:.0f}°F  |  {tokens['wbz_kft']:.1f} kft  |  {tokens['fzl_kft']:.1f} kft")]:
        axt.text(0.03, y, a, color=MUTED, fontsize=9.3, transform=axt.transAxes)
        axt.text(0.97, y, b, color=FG, fontsize=9.3, transform=axt.transAxes, ha='right')
        y -= 0.057

    fig.text(0.045, 0.965, f"{cfg['station']}  {cfg['station_name']}   {cfg['time_label']}   ({cfg.get('status', 'Observed')})",
             fontsize=16, color=COL['white'], weight='bold')
    if cfg.get('subtitle'):
        fig.text(0.045, 0.94, cfg['subtitle'], fontsize=9.5, color=MUTED)
    out = Path(outdir) / cfg['output']
    fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    print('saved', out)
    if cfg.get('sharppy_output'):
        write_sharppy(snd, cfg, Path(outdir) / cfg['sharppy_output'])
        print('saved', Path(outdir) / cfg['sharppy_output'])
    # quick console summary
    print(f"  SBCAPE {f0(obs['SBCAPE'])}  SBCIN {f0(obs['SBCIN'])}  MUCAPE {f0(obs['MUCAPE'])}  PW {f0(obs['PW'], 2)}  "
          f"LR75 {obs['LR75']:.1f}  Tc {tokens['tc_f']:.0f}F  SRH01 {kin['SRH01'].m:.0f}  SHR06 {kin['SHR06'].m:.0f}")
    if ms:
        print(f"  MOD: MLCAPE {f0(mod['MLCAPE'])}  MLCIN {f0(mod['MLCIN'])}  ML LCL {mod['ML_LCL_m']:.0f} m  "
              f"DCAPE {f0(mod['DCAPE'])}  SHIP {mod['SHIP']:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config'); ap.add_argument('--outdir')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    outdir = a.outdir or Path(a.config).parent / 'output'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    render(cfg, outdir)


if __name__ == '__main__':
    main()
