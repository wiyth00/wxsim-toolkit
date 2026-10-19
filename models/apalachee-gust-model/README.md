# Apalachee Bay sou’easter gust model — v3

**SIMULATED SCENARIO. All December 2026 weather fields and station values in this project are fictional.** Real historical observations appear only in `CALIBRATION_NOTES.md` and are clearly identified there.

Start with `HANDOFF.md` for the scenario and `CHANGELOG.md` for the upgrade. The 939 mb minimum, 941 mb landfall near St. Marks, southwest Georgia loop, Brunswick exit, and original dates are preserved. This package models wind only; surge, rain, ice, snow, and tornado impacts in the handoff are scenario narrative, not calculated outputs.

## Run

Python 3.12 was used for this delivery. Install the dependencies in your own virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 stormmodel.py --recompute
python3 stations.py
python3 gustmap2.py
python3 ensemble.py --members 20 --workers 4 --step .2
python3 ensemble_map.py
python3 validate.py
python3 anim.py 0 165
bash make_video.sh
```

The control uses a 3 km Lambert grid and six-minute time steps. Animation frames are stored every 30 minutes from Wed 06Z through Sat 16Z (165 frames). `anim.py START END` retains the original half-open frame-index interface; ranges such as `0 83` and `83 165` can run separately after the control cache is complete. `make_video.sh` needs ffmpeg; it verifies every input PNG, encodes to a temporary file, and decodes the complete MP4 before replacing the output. It produces a 16.25-second MP4 at 12 fps, including a 2.5-second final hold. Individual PNG frames are reproducible and omitted from the ZIP to avoid duplicating the video.

Twenty independently integrated perturbed members use twelve-minute sampling. Lower `--workers` if memory or CPU resources are limited. Parameters, seeds, and member draws are saved in `ensemble_parameters.json` and `ensemble.npz`. Ensemble map frequencies have 5 percentage-point granularity and are uncalibrated scenario frequencies under the chosen assumptions; they are not real-world forecast probabilities.

## What changed

| Component | v3 behavior |
| --- | --- |
| Surface mixing | Low-level jet speed plus prescribed inversion strength and mixing depth feed an explicit bulk-Richardson-number closure. Cold wedge / marine layers reduce momentum transfer; post-frontal mixing and the imposed descending jet increase it. |
| Wind structure | Separate NE/SE/SW/NW low-level 34/50/64 kt radii, smooth quadrant interpolation, curled bent-back / cold-conveyor wind maximum, translation, and a coastal high/low pressure-gradient component. The core fills while outer radii grow. |
| Intensity | Prescribed low-level jet speeds replace the tropical pressure–wind relation. Pressure and winds are independent scenario parameters in the control; pressure perturbations modestly scale member kinetic energy. |
| Convective gusts | Seeded moving downburst cells with 5–15 km cross-track FWHM, short-lived bow / line-end gust lobes, and a weaker bowing background line. Convective speeds are total gust proxies and are combined by maximum, not added to the vortex. |
| Spatial texture | Smooth spatial fields with continuous time correlation instead of independent white noise at each call. |
| Station reporting | Six-minute sampled gust / mean proxies, fixed exposure biases, AR(1) turbulence with a 30-minute correlation timescale, and synthetic hazard-dependent outages. Separate complete and reported peaks are retained. |
| Uncertainty | Track, pressure, size, timing, inversion, sting placement/amplitude/timing, and random cell realizations vary per member. Ensemble fields come from time-integrated member winds, not a blurred control swath. |
| Cache provenance | Source and optional terrain fingerprints invalidate old caches. Peak and frame keys remain compatible with v2. |

The named durations (3-second gust and 2-minute mean) are closure targets. The model does not resolve three-second turbulence or perform a literal two-minute moving average. The prescribed sting-jet lobe is a scenario proxy; no parcel trajectories diagnose a real sting jet. Quadrant radii describe the **base low-level jet field**, before the conveyor maximum and surface closure; they are **not operational 10 m hurricane wind radii**.

## Files and interfaces

| File | Purpose |
| --- | --- |
| `stormmodel.py` | Scenario configuration and numerical model. Retains `G`, `T`, `X`, `Y`, `LON`, `LAT`, `land`, `sample()`, `pressure()`, `center_xy()`, `track_xy()`, `hlabel()`, `inst_gust()`, and `vortex_gust()`. Adds `Member`, `instant_fields()`, `mixing_gust()`, `quadrant_radii()`, and `member_peak()`. |
| `plotkit.py` | Original sequential pink-to-plum palette and monotonic lightness assertion. |
| `station_catalog.py` / `stations.py` | Inherited real station coordinates and synthetic reporting model; network classifications were not reverified. |
| `gustmap2.py` | Original map layout and station label placement; writes `apalachee_gust_map_v3.png`. |
| `anim.py` / `make_video.sh` / `verify_frames.py` | Frame rendering, PNG validation, and video encoding. |
| `ensemble.py` / `ensemble_map.py` | Independent perturbed integrations and a four-panel uncertainty map. |
| `peak.npz` | Six-minute grid maximum `G`, its time `T`, simulation flag and source fingerprint. |
| `frames.npz` | Half-hour gust fields `F`, hours `ft`, simulation flag and source fingerprint. |
| `station_series.npz` | Six-minute sampled gust and mean proxies, hours and provenance. Column order follows `station_catalog.ST`. |
| `ensemble.npz` | Member peak fields, `p58`, `p74` (percent), `mean_peak`, `max_peak`, `p10`, `p90`, perturbations and provenance. |
| `synthetic_station_reports.csv` | Complete vs reported station peaks, maximum mean proxy, exposure bias, outage window. |
| `v2_v3_comparison.csv` | All station grid/report changes against the reproduced original v2. |
| `compare_v2.py` | Rebuild all v2/v3 comparison reports and run metadata from the included baseline cache. |
| `run_summary.json` / `ensemble_parameters.json` | Run statistics and full assumed-member configuration. |
| `validate.py` | Physics, deterministic behavior, peak/frame, station and ensemble consistency checks. |
| `reference_outputs/` | Original v2 map and video retained for comparison. |

`mean_peak` means the mean of **each member’s time maximum**, not the maximum of the time-varying ensemble mean. `max_peak` is the highest member peak and depends on ensemble size. `sample()` now returns NaN outside the domain rather than silently clamping to a boundary. `hlabel()` rounds to the nearest hour with day rollover.

To use the model in another scenario, edit the explicit top-level waypoint, radius, low-level jet, sting-jet, convection, and ridge parameters. The geographic masks and map domain are still Southeast-specific; those must also change when moving the model to another region. This is a reusable starting point for `wxsim-toolkit`, not a fully generalized meteorological library.

## Terrain input and calibration status

Without terrain data, the model uses coarse open-land/water roughness and a modest schematic ridge enhancement. An optional `terrain.npz` must contain `elevation_m` and `roughness_m`, finite arrays aligned exactly to the model’s X/Y grid. Roughness must be positive and less than 10 m. The loader computes smoothed, flow-dependent slope effects and log-profile exposure; it does not resolve gap flows or validate georeferencing from shape alone.

No DEM or land-cover dataset was supplied. No fitted analog calibration, out-of-sample validation, or operational warning polygons are claimed. See `CALIBRATION_NOTES.md` for the primary-source context and next calibration steps. The 58 and 74 mph map contours are gust thresholds, not issued warnings.

