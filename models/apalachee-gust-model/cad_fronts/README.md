# Appalachian cold-air-damming corridor experiment

**Fictional Dec 15–19 scenario; not a forecast.** This experiment combines the restored 939 mb cyclone low with a broad 1049 mb high centered in southwest Quebec, outside the Southeast map. The pressure field shown in the maps is the **total pressure**: the cyclone profile plus the broad background high. The 1049 mb value is an off-domain center and must not be read as a 1049 mb local Georgia high.

## Rebuild and render

Run from the repository root. The two length-scale cases are independent full integrations:

```bash
.venv/bin/python cad_fronts/model.py --scale 2.5 --out cad_fronts/size_2p5x
.venv/bin/python cad_fronts/model.py --scale 3 --out cad_fronts/size_3x
.venv/bin/python cad_fronts/validate.py -v --case cad_fronts/size_2p5x --case cad_fronts/size_3x
.venv/bin/python cad_fronts/render.py --case cad_fronts/size_2p5x --animation
.venv/bin/python cad_fronts/render.py --case cad_fronts/size_3x --animation
.venv/bin/python cad_fronts/render.py --compare cad_fronts/size_2p5x cad_fronts/size_3x
```

Model integrations start at hour 12 and end at hour 112. They store 165 half-hour fields from hour 30 through 112; gust peaks are updated at two-minute intervals. The validator must pass for both archives before rendering. The renderer checks the archive fingerprint and model version against the current `cad_fronts/model.py`, then creates `front_snapshot.png`, `front_lifecycle.png`, and an optional `cad_detail.png` in each case directory. `--animation` writes all 165 frame PNGs under `front_frames/`, verifies each with Pillow, encodes an MP4 with ffmpeg, and decodes the full MP4 with ffmpeg before installing it as `frontogenesis.mp4`. `--hour` selects the nearest stored snapshot hour; rendering uses saved archives and does not rerun the integration.

The primary snapshot defaults to model hour 48, matching the supplied Dec 16 evening reference time; `--hour` can select another stored time. It pairs gust shading and total-pressure contours with signed kinematic frontogenesis over potential-temperature contours. Frontogenesis uses a fixed shared ±4 K per 100 km per 3 h range; values beyond it saturate at the colorbar ends. The lifecycle image uses shared potential-temperature contours at 267, 270, 273, 276, 280, 284, 288, 292, 296, and 300 K, with a 265–302 K color range. City labels mark Atlanta, Athens, Gainesville, and Charlotte. The outlined corridor and sparse arrows show the schematic CAD mask and lower-level transport. A map-edge callout locates the 1049 mb center in southwest Quebec, off the plotted domain. No categorical warm/cold-front symbols are drawn.

The CAD detail image uses archived diagnostics when present; otherwise, after checking archive provenance, the renderer recomputes selected-hour upper/lower flow and mixing diagnostics from archived potential temperature and the matching model. It shows θ (K) with dashed sea-level-equivalent dry-temperature proxy contours (°F), prescribed CAD mixing-depth proxy and inversion contours restricted to the CAD mask, and paired upper-flow/lower-layer arrows. The depth and inversion fields are parameterized mixing diagnostics, not a resolved cold-pool depth or atmospheric sounding. Columbus, Macon, Augusta, and Columbia carry nearest-gridpoint **proxy** values sampled from the same field; they are not station temperatures or 2-m forecasts. The proxy uses `T = θ (p / 1000 mb)^0.2854`, converting with the model's total pressure. The pressure field is not actual station pressure at terrain elevation, so the proxy is not suitable for locating a freezing line. No freezing-rain band is diagnosed. Lower-layer arrows are prescribed transport, not 10-m winds. Any archived CAD relaxation or thermal-tendency source term is prescribed forcing; it is distinct from kinematic frontogenesis `F` and must not be labeled frontogenesis.

## Model interpretation and limits

The 2.5× and 3× cases vary the cyclone pressure-profile length scale; neither is an exact radius of maximum 34-kt winds. The gradient-wind equation describes only the cyclone's circular pressure-balanced component. Background-high geostrophic flow, cyclone translation, CAD transport, and surface mixing are separate prescribed components, so the combined flow and pressure field are approximate and are not a complete balanced analysis.

The cold pool, corridor outline, 500–800 m dome depth, 6–10 K inversion, and barrier-parallel jet are schematic parameterizations. The model does not resolve terrain with a DEM, a vertical atmospheric column, drainage, moisture, ice, snowpack, precipitation, latent heating, or two-way thermal/dynamic feedback. The temperature scalar is dry and passive apart from any explicitly documented CAD relaxation term. The gust field is a modeled surface proxy, not an observation or validated local forecast.

Kinematic frontogenesis is the rate of change of the horizontal potential-temperature-gradient magnitude under the horizontal wind, `F = −(∇hθ · ∇hV · ∇hθ) / |∇hθ|`. Positive values mark strengthening and negative values weakening of `|∇hθ|`; this diagnostic does not establish that a real atmospheric front or sting jet occurred. See [MetPy's frontogenesis definition](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.frontogenesis.html) and the [NOAA/AOML source record](https://repository.library.noaa.gov/view/noaa/6930) for the scenario's background reference material.

## Files and results

- `model.py` defines the experiment and archive format.
- `validate.py` checks manufactured behavior, archive integrity, and provenance.
- `render.py` makes case maps, comparison maps, and verified animations.
- `size_2p5x/` and `size_3x/` contain each full run's fields, peak gusts, station series, and metadata.

Case output summaries and verified model results belong here after both current archives pass validation. Do not carry forward values or claims from the earlier realistic-front experiment as results for these CAD cases.

## Vertical separation of the cold-dome ridge

The shallow CAD hydrostatic pressure bulge is retained in total sea-level pressure, but is excluded from the upper/free-flow geostrophic pressure-gradient calculation. Its influence decays with height; applying its sharp surface gradient to the upper layer creates spurious wind spikes. Upper background flow uses only the broad Quebec high. Shallow CAD transport is represented separately by the prescribed barrier jet. The validator checks that varying the cold-dome ridge depth cannot change the upper background geostrophic wind, while its sea-level pressure signature remains. No arbitrary wind-speed cap is applied.

See `SCENARIO.md` for the corrected scenario parameters and `CLAUDE_FIGURE_REVIEW.md` for the supplied image's conceptual strengths, station-versus-city sampling differences, and limits of its CSI/icing annotations.

## Verified CAD delivery

Both full runs passed all 18 manufactured/physical regression tests and both archive checks, including reproducing signed kinematic frontogenesis from saved theta and lower-layer winds. Both 165-frame animations were verified as PNGs and decoded end-to-end before publication. Each runs 20.625 seconds at 8 fps.

| Length scale | Canonical low minimum | Actual total-grid minimum | Maximum gust proxy | Maximum land gust proxy |
| --- | ---: | ---: | ---: | ---: |
| 2.5× | 939 mb | 938.87 mb | 129.96 mph | 124.84 mph |
| 3× | 939 mb | 938.84 mb | 128.52 mph | 123.90 mph |

The small difference between canonical center pressure and the total-field minimum is retained and reported; the field is not clipped to 939 mb. The 1049 mb value specifies the off-domain background-high component.

At hour 48 (7 PM EST Wednesday Dec 16), the 3× case has a modeled upper/free-flow speed near Atlanta of 39.44 m/s (88.2 mph), lower-layer transport of 14.58 m/s, and a surface gust proxy of 37.12 mph beneath the CAD inversion. This is model behavior, not an observed wind profile. The core cold/warm contrast remains at least 10 K between Atlanta and Albany over hours 48–78; those values are potential temperature, not instrument-level air temperatures.

`comparison.json` and `station_comparison.csv` hold the numerical summaries; regenerate with `.venv/bin/python cad_fronts/summarize.py`. Earlier experiments remain intact under their original directories.
