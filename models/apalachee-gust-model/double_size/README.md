# Double-size Apalachee Bay sou’easter

**SIMULATED SCENARIO. Fictional weather, not a forecast.**

This separate deterministic experiment doubles cyclone radius and diameter relative to the preserved parent v3 model. A geometrically similar footprint would have four times the area. Actual surface-gust threshold areas differ because the track sweeps across fixed terrain, land/water exposure, cold-air damming, coastal-gradient winds, and a finite map domain.

Changed: radius of maximum base wind, all quadrant low-level 34/50/64 kt radii, the outer exponential decay length, conveyor-arc radial offset and width, and translation-wind decay length. `STORM_SIZE_SCALE = 2.0` in `stormmodel.py` sets the scale. The existing member radius parameter retains its role as an additional perturbation.

Preserved: track, December 2026 timing, 939 mb minimum / 941 mb landfall, prescribed jet speeds, random seeds, local sting-jet dimensions and motion, local downburst/convective swaths, terrain, mixing closure and geographic masks. Pressure and jet speed are independent prescribed parameters in this model. This is a size sensitivity experiment, not a dynamically balanced atmospheric simulation.

The original Southeast map domain is retained for direct comparison; the doubled cyclone extends beyond it. Reported areas are projected grid areas within that domain. Low-level 34 kt radii are not operational 10 m gale radii. Gust contours are not issued warnings. No new ensemble, surge, rain, snow or ice calculation is included.

## Outputs

- `size_comparison.png`: original and doubled peak-gust fields on the same scale.
- `apalachee_gust_map_size2x.png`: detailed map with synthetic station reports.
- `apalachee_gust_animation_size2x.mp4`: half-hour gust evolution.
- `size_comparison.json`: maxima, threshold footprints, radii and provenance.
- `station_size_comparison.csv`: six-minute grid peaks at all 69 stations; uses complete model series to avoid outage bias in this comparison.
- `synthetic_station_reports.csv`: separate synthetic reports with exposure, turbulence and possible outages.
- `peak.npz`, `frames.npz`, `station_series.npz`: independent experiment caches.

## Reproduce

From this directory, use the parent `.venv` created with Python 3.12 and the project's existing `requirements.txt` (NumPy, SciPy, Matplotlib, Basemap/data and Pillow):

```sh
../.venv/bin/python stormmodel.py --recompute
../.venv/bin/python stations.py
../.venv/bin/python gustmap2.py
../.venv/bin/python compare_size.py
../.venv/bin/python anim.py 0 165
../.venv/bin/python verify_frames.py
ffmpeg -y -nostdin -xerror -err_detect explode -loglevel error -framerate 12 -i frames/f_%04d.png -vf 'tpad=stop_mode=clone:stop_duration=2.5,format=yuv420p' -c:v libx264 -threads 2 -crf 20 -preset medium -movflags +faststart apalachee_gust_animation_size2x.mp4
ffmpeg -nostdin -v error -xerror -i apalachee_gust_animation_size2x.mp4 -f null -
```

`compare_size.py` checks original and experiment source fingerprints, doubled radii/core, unchanged scenario inputs, frame/peak consistency, finite fields, and deterministic call order. It expects the parent baseline caches to remain intact.
