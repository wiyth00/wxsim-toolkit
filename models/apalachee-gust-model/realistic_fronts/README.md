# Dry thermal-front experiment

**SIMULATED SCENARIO.** These maps and fields are fictional scenario output. The experiment adds a transported potential-temperature field and diagnoses horizontal frontogenesis from its evolving temperature gradient and the resolved horizontal wind. It is a prescribed-flow experiment, not a numerical weather prediction forecast or an analog-calibrated reconstruction.

## Render the stored outputs

The shared model is run separately for each length scale. From the project root:

```bash
.venv/bin/python realistic_fronts/model.py --scale 2.5 --out realistic_fronts/size_2p5x
.venv/bin/python realistic_fronts/model.py --scale 3 --out realistic_fronts/size_3x
.venv/bin/python realistic_fronts/validate.py -v --case realistic_fronts/size_2p5x --case realistic_fronts/size_3x
```

Each run starts at model hour 12 and ends at hour 112. Theta and grid gust peaks advance every 120 seconds; all 69 stations are sampled every six minutes. There are 165 stored half-hour fields from hour 30 through hour 112. Theta and wind components retain float32 precision so the signed frontogenesis archive can be reproduced from its source fields. Pressure and gust frames use float16; peak gusts retain float32 precision. Dependencies reuse the existing project `.venv` and `requirements.txt`.

From the repository root, activate the project environment and run:

```bash
.venv/bin/python realistic_fronts/render.py --case realistic_fronts/size_2p5x
.venv/bin/python realistic_fronts/render.py --case realistic_fronts/size_3x --animation
.venv/bin/python realistic_fronts/render.py --compare realistic_fronts/size_2p5x realistic_fronts/size_3x
```

Each case command writes `front_snapshot.png`, a two-panel gust/pressure and signed-frontogenesis map near model hour 54, and `front_lifecycle.png`, a four-time sequence pairing signed frontogenesis with actual potential-temperature contours. `--hour` selects the nearest stored hour for the snapshot. `--animation` also writes one PNG per half-hour time under `front_frames/` and, when `ffmpeg` is available, `frontogenesis.mp4`. The comparison command writes `realistic_fronts/size_comparison.png`; its rows compare maximum gust proxy and frontogenesis at the requested hour using shared color scales. Rendering reads the saved archives; it does not start or recompute a model run. It requires a `summary.json` with source-change invalidation enabled and matching `source_fingerprint` values in the summary, `fields.npz`, and `peak.npz`; it also compares that fingerprint with the current model.

The frontogenesis map uses a fixed symmetric range of ±2 K per 100 km per 3 h. Blue values are negative frontogenesis (frontolysis), orange values are positive frontogenesis, and white is near zero. Values outside the color range are saturated and marked by colorbar extensions. The three outer grid cells are masked because centered finite differences are least reliable at the domain edge. Potential-temperature contours are labeled in kelvin. Pressure contours are labeled in millibars. Gust shading uses the project’s pink-to-plum palette with its labeled mph scale.

## Meaning and limits

The central pressure follows the original waypoint curve (939 mb minimum and 941 mb at landfall). For reference pressure 1012 mb, the prescribed circular pressure basin is `p(r) = pc + (1012 - pc) exp(-R/r)`, with `R` equal to the original maximum-wind radius parameter times 2.5 or 3. This length scale is a pressure-profile parameter; after including Coriolis, it is not necessarily the radius of maximum wind. The balanced circular component solves `V²/r + fV = (1/rho) dp/dr`, using density 1.15 kg/m³ and Coriolis at the storm center. It replaces both the original independently prescribed jet-speed profile and the later linear pressure-deficit / square-root wind multipliers. Track translation and localized normal strain are additional prescribed flow terms; the total flow is not claimed to satisfy radial balance.

Theta is advected with midpoint trajectory backtracking and bilinear interpolation, with an analytic ambient open boundary and explicit 300 m²/s diffusion. The trajectories are second order; bilinear scalar interpolation introduces additional numerical diffusion. Initial conditions contain one idealized baroclinic zone and a warm-sector tongue. Curled thermal structures arise through transport, but no full Shapiro–Keyser life cycle or self-generated warm seclusion is claimed. The gust closure uses an idealized cold-anomaly stability proxy; it does not measure a vertical inversion. Old ad hoc sting and convective gust boosts are omitted from this experiment.

The diagnostic follows the horizontal kinematic definition used by MetPy: frontogenesis is the rate of change of the magnitude of the horizontal potential-temperature gradient under the horizontal wind. In tensor form, `F = -(∇hθ · ∇hV · ∇hθ) / |∇hθ|`, using the velocity-gradient convention implemented by the model. The dimensional rate in K m⁻¹ s⁻¹ is converted to K per 100 km per 3 h by multiplying by `100,000 m × 10,800 s = 1.08×10⁹`. The renderer does not smooth the diagnostic or turn it into categorical “fronts.” The formula and sign convention are documented in [MetPy’s frontogenesis reference](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.frontogenesis.html).

Potential temperature is transported as a passive, dry scalar by the prescribed two-dimensional flow. The scenario restores the extreme 939 mb pressure minimum, and the model retains the radial wind-balance assumption; the 2.5× and 3× labels describe a changed wind-field length scale, not an exact 34-kt radius. Temperature does not feed back on pressure or wind. There is no vertical motion, moisture, latent heating, precipitation, boundary-layer evolution, or sting-jet diagnosis. The diagnostic can show where this prescribed flow tightens or relaxes the horizontal thermal gradient; it cannot establish that a real atmospheric front or sting jet occurred. The radial-balance assumption is separate from the frontogenesis diagnostic; see [NOAA/AOML’s gradient-wind balance overview](https://www.aoml.noaa.gov/hrd/hrd_sub/gradbal.html?print=yes).

## Case files

- `model.py` provides the shared scenario model and exports the model grid (`X`, `Y`, `DX`) and map projection.
- `size_2p5x/` and `size_3x/` hold each run’s `fields.npz`, `peak.npz`, and `summary.json`.
- `render.py` creates maps and animation without initiating model integration.

## Delivered run and verification

| Length scale | Minimum pressure | Maximum grid gust proxy | Maximum land gust proxy |
| --- | ---: | ---: | ---: |
| 2.5× | 939 mb | 106.63 mph | 95.52 mph |
| 3× | 939 mb | 104.59 mph | 94.61 mph |

These are outputs of the chosen idealized pressure profile and mixing closure, not calibrated estimates for a real 939 mb cyclone. The larger case has a broader 58/74 mph event footprint within the map while its maximum gust decreases slightly. Geographic areas cover only the original Southeast domain, which does not contain the entire larger storm.

`validate.py -v --case size_2p5x --case size_3x` (with the complete paths shown above) passed all 10 manufactured checks and both archive checks. Tests include pressure-gradient balance, frontogenesis signs and units, pure-rotation cancellation, transported-gradient growth under compression, scalar bounds, source provenance, and reproduction of archived frontogenesis from archived theta and wind. Both 165-frame animations were checked as PNGs and decoded end-to-end; each is 20.625 seconds at 8 fps.

`comparison.json` and `station_comparison.csv` contain the complete grid-summary and 69-station comparison. Rebuild them from the project root with `.venv/bin/python realistic_fronts/summarize.py`. Station values come directly from the complete modeled series, without a synthetic exposure/outage reporting layer. `model.py`, `render.py`, `validate.py`, and `summarize.py` are the editable sources. The earlier size-only and extreme pressure-scaled experiments remain separate and preserved.
