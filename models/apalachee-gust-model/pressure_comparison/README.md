# Pressure-deficit scaling experiment

**EXTREME SIMULATED SCENARIO. These are assumed sensitivity rules, not physically balanced atmospheric solutions.**

The model's reference pressure is 1012 mb and original minimum is 939 mb, a 73 mb deficit. For radius multiplier s, impose `p_new(t) = 1012 - s × (1012 - p_original(t))`. Minimum pressures become 829.5 mb at 2.5× and 793 mb at 3×. Landfall pressures become 834.5 and 799 mb.

Changing pressure alone would not increase this model's control winds, because jet speed is prescribed independently. These separate experiments therefore multiply the entire cyclone tangential wind profile by `sqrt(s)` relative to each matching size-only run. This is an explicit additional assumption. The background coastal-gradient component, translation, local sting boost, downbursts, track, timing and geographic mixing retain their previous settings. Surface gusts respond nonlinearly to mixing.

Structural radii retain their positions, but their original 34/50/64 kt anchors now have speeds multiplied by `sqrt(s)`. They are not literal 34/50/64 kt radii in these experiments.

Larger size alone does not require this pressure scaling. Real wind balance depends on pressure gradients, radius and rotation; see [NOAA gradient balance](https://www.aoml.noaa.gov/hrd/hrd_sub/gradbal.html?print=yes). That source does not validate the chosen rule. The model solves no pressure field, thermodynamics or atmospheric mass redistribution and is not validated for these extreme values.

Outputs: `pressure_comparison.png`, `pressure_comparison.json`, `station_pressure_comparison.csv`. Maps, model sources, independent caches and animations are in `../size_2p5x_pressure/` and `../size_3x_pressure/`. Prior experiments are preserved. All areas remain limited to the original map domain.

Rebuild the comparison from the project root with `.venv/bin/python pressure_comparison.py`. Each scenario README gives full reproduction steps. Verification covers the prescribed pressure rule, radius scaling, unchanged inputs, cache fingerprints, finite fields, 165 half-hour fields, peak consistency and deterministic call order. Video encoders verify every input image and decode the full output before finalizing it.
