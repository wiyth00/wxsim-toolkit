# 2.5× original cyclone radius

**SIMULATED SCENARIO. Fictional weather, not a forecast.**

This separate deterministic run scales the original cyclone radius and diameter by 2.5, giving 6.25 times the geometric area before geography, mixing and map limits. It uses exactly the same scaling definition as the 2× experiment.

Changed: maximum-wind radius, all quadrant low-level 34/50/64 kt radii, outer decay length, bent-back/conveyor arc offset and width, translation-wind decay length. `STORM_SIZE_SCALE = 2.5` in `stormmodel.py` controls the scale relative to the original, not relative to the 2× run.

Preserved: track, timing, 939 mb minimum / 941 mb landfall, prescribed jet speeds, random seeds, geographic masks, terrain, mixing, local sting-jet and downburst dimensions. This is a parametric sensitivity experiment, not a balanced atmospheric simulation. No new ensemble, surge, rain, ice or snow calculation.

The original Southeast domain is retained. The expanded storm extends beyond it; all footprint areas are projected grid areas limited to that map. Low-level radii are not operational 10 m gale radii. Individual stations need not increase monotonically because the strongest ring moves outward.

Outputs: `peak.npz`, `frames.npz`, `station_series.npz`, `synthetic_station_reports.csv`, `apalachee_gust_map_size2p5x.png`, and `apalachee_gust_animation_size2p5x.mp4`. Four-case comparison data, station CSV and map are in `../size_comparison/`.

## Reproduce

From this directory, with the existing parent Python 3.12 `.venv` and project dependencies:

```sh
../.venv/bin/python stormmodel.py --recompute
../.venv/bin/python stations.py
../.venv/bin/python gustmap2.py
../.venv/bin/python anim.py 0 165
bash render_video.sh
cd ..
.venv/bin/python size_sweep.py
```

The comparison validates radii/core scaling, unchanged scenario inputs and grid, all cache fingerprints, six-minute station sampling, 165 half-hour fields, frame/peak consistency, finite fields and deterministic call order. The video script verifies every PNG and decodes the entire video before publishing it.
