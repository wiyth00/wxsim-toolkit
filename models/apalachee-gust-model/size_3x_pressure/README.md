# 3× size and pressure-deficit sensitivity

**SIMULATED SCENARIO. Deliberately extreme fictional experiment, not a realistic forecast.**

Original minimum pressure: 939 mb. Using the model's 1012 mb reference, the original pressure deficit is 73 mb. This experiment imposes `p_new(t) = 1012 - 3 × (1012 - p_original(t))`, giving a 793 mb minimum and 799 mb landfall pressure.

The radius remains 3× the original. Cyclone tangential winds, including the bent-back maximum, are multiplied by `sqrt(3) = 1.732051` relative to the matching size-only case. This is an explicit heuristic inspired by the cyclostrophic wind-squared/pressure-deficit relation; it is not derived from the actual storm's pressure field. No pressure field, gradient-wind balance, thermodynamics or atmospheric mass redistribution is solved. Larger size does not automatically require these pressures.

The scaled structural radius anchors now have velocities `34/50/64 kt × sqrt(3)`; they must not be read as actual 34/50/64 kt radii. Translation winds, geographic pressure-gradient component, local sting boost, convective cells, track, timing, surface closure and random seeds retain their previous settings. Synthetic station reports remain separate from complete grid peaks.

The geographic domain is retained for comparison. Many wind thresholds reach its boundary; reported areas omit the storm beyond the map. No ensemble or surge/precipitation calculation.

Primary-source context for why size, pressure gradient, rotation and wind are coupled in real storms: [NOAA gradient balance](https://www.aoml.noaa.gov/hrd/hrd_sub/gradbal.html?print=yes). That source does not validate this experiment's scaling rule.

## Reproduce

From this directory with the existing parent `.venv`:

```sh
../.venv/bin/python stormmodel.py --recompute
../.venv/bin/python stations.py
../.venv/bin/python gustmap2.py
../.venv/bin/python anim.py 0 165
bash render_video.sh
cd ..
.venv/bin/python pressure_comparison.py
```

Outputs: independent model/cache files, `synthetic_station_reports.csv`, `apalachee_gust_map_size3x_pressure.png`, `apalachee_gust_animation_size3x_pressure.mp4`. The comparison and complete station peaks are in `../pressure_comparison/`.
