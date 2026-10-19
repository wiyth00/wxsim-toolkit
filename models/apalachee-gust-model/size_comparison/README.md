# Storm size sweep

**SIMULATED SCENARIO. Fictional, uncalibrated parametric controls.**

The comparison uses 1×, 2×, 2.5× and 3× the original cyclone radius. Track, pressure, prescribed wind speeds, geographic mixing and local sting/downburst scales remain fixed. The original and 2× files are preserved.

| Radius | Grid peak mph | Largest half-hour 58+ mph footprint km² | Ratio to original |
| --- | ---: | ---: | ---: |
| 1× | 149.69 | 234,018 | 1.000× |
| 2× | 151.67 | 1,020,015 | 4.359× |
| 2.5× | 153.19 | 1,441,674 | 6.161× |
| 3× | 154.04 | 1,520,694 | 6.498× |

Areas cover only the original Southeast domain. The storms extend beyond the map, especially at 3×; area growth is clipped. Event footprints on the map are the union over the full run and differ from the largest simultaneous footprint above. Station comparisons use complete grid series, before synthetic exposure, turbulence and outages.

Outputs: `size_sweep.png`, `size_sweep.json`, `station_size_sweep.csv`. Model sources, detailed maps, 165-frame caches and verified animations are in `../size_2p5x/` and `../size_3x/`. Each has a README with reproduction commands.

Verification: all four models passed unchanged-input/grid checks, exact radius/core scaling, cache provenance, six-minute station sampling, 165 half-hour frames, peak/frame consistency, finite fields and deterministic call-order checks. New animation PNGs were verified individually; the encoding scripts decode the full videos before finalizing them.

Rebuild this comparison from the project root with `.venv/bin/python size_sweep.py`. No new ensemble, balanced atmospheric dynamics, surge or precipitation calculation is included.
