# Claude task: extend the blue-hook visualization

Please work from `hook_explorer.py` and `preview_data.npz`. Keep the current
two-panel comparison: potential temperature with transport arrows and tracer
tracks on the left; archived kinematic frontogenesis F on the right with a
linear, symmetric ±16 K / 100 km / 3 h scale. Preserve the projected axes,
coast/state geometry, city references, storm center, and clear simulated-case
labels.

Implement these three additions:

1. Add a configurable **asinh normalization** for F that reveals weak values
   without hiding the large tails. Keep zero centered, preserve the existing
   linear ±16 scale as an explicit comparison/reference mode, label the
   selected transform and its parameters, and make colorbar ticks intelligible
   in the original linear units. Do not silently clip values.
2. Add a configurable **gradient-magnitude mask** for the F display. Compute
   the mask from horizontal potential-temperature gradient in the active grid,
   with the true x/y spacing and stored crop origin. Make threshold and masked
   appearance explicit. Keep missing/out-of-domain cells distinct from values
   hidden by this mask. Explain that sparse preview gradients are approximate;
   encourage `--full-fields` for diagnostic work.
3. Add **Lagrangian diagnostics along the existing 2-D trajectories**. The
   script already integrates markers with time-interpolated winds using RK2;
   do not duplicate that work. Sample and plot or export theta, |grad(theta)|,
   and F along each path, and add a clearly qualified gradient-change budget
   if it can be supported. The archived F is kinematic and excludes explicit
   CAD restoring source Q. The archive includes exact-grid static
   `cad_mask_3km` and `theta_source_target_K_3km`, plus metadata for
   `cad_restoring_timescale_s=21600` and `scalar_diffusivity_m2_s=300`.
   The model source is `Q = mask * (target - theta) / tau`; diffusion is
   `kappa * laplacian(theta)`. Keep preview limitations clear: the budget
   cannot close on sampled 9 km fields, and even at 3 km it is a numerical
   diagnostic across an advection-diffusion-restoring integration. Do not
   claim a complete material budget from archived F alone.

Keep units visible, options reproducible from the command line, and the default
render useful. Add a concise output file (CSV/NPZ or a small summary) for the
per-tracer time series. Do not imply 3-D parcel motion, observed cold-front
identity, precipitation, or sting-jet structure. Preserve the source
fingerprint and describe any calculation that differs from the original 3 km
archive.

## Source facts

- Model fingerprint: `0a5a143578b010d086bbf58fdf24a57f6ad717d0641fd5b85ada0c317051ed8f`.
- Preview period: 60–80 h at 30-minute intervals; source grid spacing 3 km.
- Hook-side marker origins are backtraced to hour 64; their forward tracks end at hour 80.
- `theta_K` is model potential temperature, not 2 m temperature.
- Archived F units are K / 100 km / 3 h.
- Winds are prescribed lower-layer transport in m/s.
- The CAD corridor is a schematic mask, not resolved terrain.
