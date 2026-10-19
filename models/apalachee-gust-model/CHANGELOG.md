# v3 change record — SIMULATED SCENARIO

Implemented the v2 handoff’s first four priorities and station reporting improvements, with documented terrain and calibration limits. The canonical low track and pressure evolution are unchanged.

The surface closure now uses an inversion/mixing-depth response to low-level wind; the tropical pressure–wind law and flat wedge/land-water gust factors are removed. Low-level 34/50/64 kt radii vary by quadrant and expand during filling. A separate curled conveyor maximum and moving prescribed descending jet supply localized structure. Seeded downburst swaths and short-lived bow/line-end gust lobes replace the smooth severe-wind band; early visual QA led to shorter vortex lifetimes to eliminate artificial long, straight tracks.

The ensemble independently integrates twenty perturbed realizations at twelve-minute intervals and records all member parameters and peak fields. Synthetic station reporting uses six-minute gust/mean proxies, fixed exposure biases, correlated turbulence and four modeled outages in this realization. Original station white noise and the arbitrary 20 mph reporting floor are removed.

The original map layout, palette and simulation labeling are retained. A four-panel uncertainty map is added. Source/terrain fingerprints prevent stale model caches. Video rendering writes verified temporary PNGs atomically; encoding validates every input and stops on decoding/encoding errors. Initial corrupt frames were regenerated before final delivery.

`HANDOFF.md` and `v2_v3_comparison.csv` document every old headline station change. The source, original v2 references, numerical caches, member metadata, maps, video, CSV reports and validation workflow are packaged together.

No fitted analog calibration, real terrain data, resolved gap flow, official warning areas, or surge/ice/rain/snow calculations are claimed.
