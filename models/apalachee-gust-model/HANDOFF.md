# Handoff — v3 upgrade completed

**SIMULATED SCENARIO. All event winds and station reports are fictional.**

Read `README.md` for current commands, output semantics, interfaces, and limitations. The original v2 handoff follows below as historical context; its task list and regression values no longer describe the current model.

## Current state

- Priorities 1–4 implemented: stability-aware surface mixing; quadrant radii and a bent-back / cold-conveyor maximum; twenty independently integrated perturbed members; seeded, moving convective swaths and transient bow gusts.
- Priority 7 implemented as reporting proxies: correlated turbulence, fixed exposure biases, mean vs gust fields, hazard-dependent synthetic outages and separate reported/full peaks.
- Priority 5 remains partial: no actual DEM or land-cover input was supplied. A validated aligned input hook and schematic fallback exist; gap flows are not resolved.
- Priority 6 is an inspected primary-source analog review and calibration protocol, **not a fitted calibration**. See `CALIBRATION_NOTES.md`.
- Scenario dates, central pressure waypoints, track, pink-to-plum palette, and simulation labels are retained.
- The model retains the v2 public grid / sampling / track interfaces and `peak.npz` / `frames.npz` keys. Output image/video names now use `_v3`.

## Changed headline values

Control grid maximum: **123.1 → 149.7 mph**. Land maximum: **121.7 → 135.7 mph**. These are numerical grid peaks, distinct from station reports. The higher values reflect prescribed low-level jet speeds and the revised momentum-transfer closure; they are not an observed event or a pressure-derived hurricane category.

The table compares the original v2 synthetic reporting algorithm with v3 exposure/turbulence/outage reports. Raw grid changes are separately recorded for all 69 station locations in `v2_v3_comparison.csv`.

| Station | v2 report mph | v3 report mph | Change mph |
| --- | ---: | ---: | ---: |
| Moultrie | 115 | 137 | +22 |
| Valdosta | 112 | 121 | +9 |
| St. Simons | 103 | 89 | -14 |
| Brunswick | 101 | 91 | -10 |
| Apalachicola | 94 | 105 | +11 |
| Tifton | 87 | 98 | +11 |
| Panama City | 85 | 93 | +8 |
| Jacksonville | 85 | 82 | -3 |
| Savannah | 76 | 78 | +2 |
| Charleston | 68 | 72 | +4 |
| Orlando | 72 | 59 | -13 |
| Tampa | 74 | 56 | -18 |
| Mobile | 50 | 53 | +3 |
| Atlanta | 42 | 36 | -6 |
| Charlotte | 43 | 30 | -12 |
| Tallahassee | 83 | 111 | +28 |

Four synthetic outages occur in this realization. For example, Apalachicola’s model grid peak is about 135 mph while its reported synthetic peak is about 105 mph after the outage model. Missing peaks should not be read as weak winds.

The twenty member-wide maxima span roughly 139–154 mph. The ensemble panels display member exceedance frequencies for 58/74 mph and mean/highest member peak fields. These frequencies depend on assumed perturbations and are not calibrated forecast probabilities.

## Verification and next work

`validate.py` checks canon, stability response, quadrant interpolation/growth, deterministic call order, moving sting and convective proxies, cache/peak/frame consistency, station outages, and ensemble statistics. All eight checks passed for the delivered run. The animation was decoded end-to-end after validating all 165 PNGs; frame files are intentionally omitted from the ZIP, while cached gust fields and the completed MP4 are included.

Next priorities: real analog time-series calibration, supplied terrain/land cover, and sensitivity experiments for jet speed/mixing. Retain honest distinctions between low-level radii, gust thresholds, station peaks, and model maxima. No operational warning polygons or surge/precipitation calculations have been added.

---

# Original v2 handoff (historical reference)

# Handoff: Apalachee Bay Sou'easter wind model (fictional scenario)

You are taking over a working Python gust model, map renderer and animation for a **fictional** weather scenario. Everything is simulated. Keep the "SIMULATED SCENARIO" labeling on every output. Stations are real ASOS locations with **synthetic** values.

## Scenario canon (v2, supersedes any earlier numbers)
- **Date:** Dec 15-19, 2026. Hour 0 = Tue Dec 15 00Z.
- **Background:** strongest El Nino on record (Nino 3.4 ~ +3C); subtropical jet 200+ kt at 250 mb along 28-30N; Gulf very warm (Loop Current eddy, SST ~ +2C); no hurricane touched the Gulf this season. A brief -NAO block (Greenland ridge over a 50/50 low) drives a ~1046 mb Arctic high into Quebec and a strong cold-air-damming wedge (Atlanta 29F, dewpoints in the teens).
- **Storm:** a Gulf low that bombs. Shapiro-Keyser structure: frontal fracture, bent-back front, warm seclusion. It never turns north like a typical nor'easter; it cuts off, loops, and exits east.
- **Pressure curve (mb):** Tue 12Z 1006 -> Wed 06Z 997 -> Wed 12Z 982 -> Wed 18Z 962 -> Thu 00Z 947 -> **Thu 03Z 939 (minimum, just offshore)** -> **Thu 06Z 941 (landfall near St. Marks, FL)** -> Thu 12Z 946 -> Fri 06Z 960 -> Sat 06Z ~972 -> exits off Brunswick, GA ~Sat 12Z at 975. About 56 mb in 24 h at 28-30N = ~4 Bergerons.
- **Track:** Gulf (-96.2, 25.4) -> St. Marks (-84.2, 30.05) -> cyclonic loop over SW Georgia (~-84.2 to -84.9, 30.9-31.7) -> Brunswick (-81.3, 31.4) -> offshore NE. Full waypoint table is `WP` in `stormmodel.py`.
- **Impacts already described to the user:** surge 10-15 ft in Taylor/Dixie counties; sting-jet gusts across south Georgia; overnight squall line with embedded supercells across the Florida peninsula (Wed night, Feb 1998 style); freezing rain 0.75-1.25" in the wedge (NE Georgia, Upstate SC, east metro Atlanta); 12-24" snow with thundersnow over north Alabama/Tennessee/southern Appalachians; 8-12" rain on the coastal plain; 48+ h of SE gales on the GA/SC coast.

## What exists
| File | Role |
|---|---|
| `stormmodel.py` | Parametric model. Grid 3 km on a Lambert conformal Basemap domain (1500 x 1450 km). Holland-style vortex on the track, 6-minute steps, + sting-jet lobe (Thu ~11Z, south GA) + eastward squall-line swath (Wed night, FL peninsula) + cold-air-damming suppression (flat 0.62 factor) + Blue Ridge enhancement (hand-drawn polyline) + land/water contrast + smoothed noise. Caches `peak.npz` (max gust `G`, hour of peak `T`) and `frames.npz` (30-min frames). Exposes `G, T, X, Y, LON, LAT, land, sample(), pressure(), center_xy(), track_xy(), hlabel()`. |
| `plotkit.py` | Shared palette (9-step pink to plum sequential ramp, lightness asserted monotonic), levels 40-120+ mph, colorbar helper. |
| `gustmap2.py` | Still peak-gust map: ~67 stations with greedy collision-free labels, top-reports box, track overlay, 58/74 mph contours. |
| `anim.py` | Frame renderer `python3 anim.py START END`: instantaneous gust field, HUD (pressure, 24 h change in Bergerons, max gust, area >= 58 mph), pressure inset. |
| `reference_outputs/` | Current still map and MP4 for visual comparison. |

## Run
```
pip install -r requirements.txt
python3 stormmodel.py --recompute      # ~1 min; writes peak.npz, frames.npz
python3 gustmap2.py                    # still map
python3 anim.py 0 83 & python3 anim.py 83 165 ; wait   # frames -> ./frames
bash make_video.sh                     # needs ffmpeg
```

## Current outputs (regression targets; changes should be deliberate)
Peak gust max 123 mph. Top reports: Moultrie 115, Valdosta 112 (both Thu 11Z sting jet), St. Simons 103 and Brunswick 101 (Sat 01-02Z as the low exits), Apalachicola 94 (Thu 04Z), Tifton 87, Panama City 85, Jacksonville 85. Savannah 76, Charleston 68, Orlando 72, Tampa 74, Mobile ~50, Atlanta ~42 (wedge), Charlotte ~43.

## Your task: raise realism, in this order
1. **Stability-aware gusts.** Replace the flat wedge factor and fixed land/water gust factors with gust = f(low-level jet speed, inversion strength / mixing depth). Wedge and cold marine layer decouple the surface from the 850 mb jet; sting-jet descent and post-frontal mixing couple fully.
2. **Comma-shaped, quadrant-asymmetric wind field.** Replace the symmetric Holland vortex with separate 34/50/64 kt radii per quadrant, a bent-back front and cold-conveyor-belt wind maximum; gale radius should grow as the low matures (rapid fill of the core, expanding outer winds over the loop).
3. **Ensemble / uncertainty.** Perturb track, intensity and sting-jet placement; output probability-of-exceedance maps (58, 74 mph) and an ensemble-mean/peak panel. Optionally map exceedance onto NWS-style High Wind Warning / Extreme Wind Warning areas.
4. **Streaky convective gusts.** Make the squall line a stochastic set of 5-15 km downburst swaths, bow echoes and line-end vortices instead of one smooth band.
5. **Terrain and land cover** if data can be supplied: roughness length, DEM-based ridge speedups and gap flows instead of the hand-drawn ridge line.
6. **Calibration against analogs:** Big Bend landfalls (Michael, Idalia), the March 1993 Southeast bomb; check inland decay and the wind-pressure relation (a 939 mb extratropical low should not use a tropical relation; current scale factor is 0.72).
7. **Station realism:** correlated turbulence instead of white noise, exposure biases, 3-second gust vs 2-minute mean, outages near peak.

## Constraints
- Keep the module interface and file layout so `gustmap2.py` and `anim.py` still run; update them where outputs change.
- Keep the sequential pink-to-plum ramp, the layout and the disclaimers. Any new colors must keep lightness monotonic (`plotkit.py` asserts it).
- Intended home: the user's `wxsim-toolkit` repo; keep the code reusable for other scenarios (track, pressure curve and canon should stay parameters at the top of the file, not buried in functions).
- Report anything that changes the headline numbers above.
