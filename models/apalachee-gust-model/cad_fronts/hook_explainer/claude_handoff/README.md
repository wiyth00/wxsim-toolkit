# Apalachee blue-hook visualization handoff

This folder is a small, portable starting point for revising the blue-hook
visual. It contains an editable standalone renderer, a compact preview-data
archive, and the exact project renderer that produced the source animation.
The project renderer may import the project model; `hook_explorer.py` does not.

## Run the standalone preview

Install Python 3.10+ and the three packages in `requirements.txt`, then run:

```sh
python -m pip install -r requirements.txt
python hook_explorer.py --hour 70 --no-video
```

The resulting PNG is `hook_preview.png`. To use the original 3 km fields
instead of the bundled sampled preview:

```sh
python hook_explorer.py --full-fields /absolute/path/to/fields.npz --hour 70
```

An optional movie can be rendered with `--video hook_preview.mp4` when ffmpeg
is installed. The portable runtime does not need Basemap: projected coast and
state polylines are stored as numeric arrays in `preview_data.npz` and drawn
with Matplotlib `LineCollection`.

## Data and interpretation

The default archive contains 41 half-hour fields from hours 60–80 over the
full original model domain. It samples every third cell of the source 3 km
grid, so the preview spacing is 9 km. These are point samples, not area averages.
The archived frontogenesis values are original 3 km calculations sampled at
those locations; recomputing gradients from the preview's sampled theta and
winds will not reproduce those original gradients. The x/y meter axes are
stored explicitly against the original model-grid origin. The preview tracer
paths are reintegrated from sampled 9 km winds and are approximate; use
`--full-fields` when close reproduction of the source 3 km trajectories matters.

`--full-fields` loads the full domain at its original 3 km spacing after
checking the source fingerprint. Tracer integration uses that full domain;
only the plotted map is zoomed to the hook ROI. Data fields are `theta_K` (K), `F_K_100km_3h`
(K / 100 km / 3 h), and lower-layer transport `u_ms`/`v_ms` (m/s).
Hook-side seeds begin at hour 64 after backtracing the hour-70 hook points;
the supplied RK2 forward integration runs from hour 64 to hour 80.
For optional source-term work, the archive also carries the original-grid
`cad_mask_3km` and `theta_source_target_K_3km`; its metadata records the 21,600 s
CAD restoring time and 300 m²/s diffusivity. Those do not make the 9 km preview
suitable for closing a material gradient budget.

The markers are two-dimensional trajectories through prescribed archived
layer winds. They are not 3-D air parcels. Archived kinematic frontogenesis
F excludes the model's separate explicit CAD restoring source Q, and therefore
does not constitute a complete parcel temperature-gradient budget. This is a
dry schematic experiment, not a forecast, resolved-terrain simulation, or
evidence of precipitation or a sting jet. The model-hour-to-scenario clock is
defined; the screenshot's case and timestamp match remains approximate.

`CLAUDE_PROMPT.md` gives the bounded next-step request: add asinh color
normalization, a gradient mask, and Lagrangian diagnostics while preserving
those distinctions.
