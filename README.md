# wxsim — weather scenario graphics toolkit

Reusable generators for simulated severe-weather scenarios: Skew-T soundings, SPC-style convective
outlooks (categorical + tornado/wind/hail probabilities), Mesoscale Discussion graphics, watch graphics,
NWS-style active warning/hazard maps, procedural simulated radar, and a text wrapper for NWS-style products. **All scenario data lives in JSON files; the code never
changes between events.**

Every graphic is labeled as simulated, and none of them use real agency logos.

## Setup

```bash
cd wxsim-toolkit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Boundary data (US states, counties, national outline from the `us-atlas` project) is bundled in `data/`, so
no downloads are needed.

## Render a whole scenario

```bash
python render_all.py scenarios/piedmont_dryline_2027
```

Or one product at a time:

```bash
python -m wxsim.sounding scenarios/piedmont_dryline_2027/sounding_KFFC_12z.json
python -m wxsim.outlook  scenarios/piedmont_dryline_2027/outlook_day1_1300z.json
python -m wxsim.mcd      scenarios/piedmont_dryline_2027/mcd_0487.json
python -m wxsim.watch    scenarios/piedmont_dryline_2027/watch_0124.json
python -m wxsim.hazards  scenarios/piedmont_dryline_2027/hazards_2245z.json
python -m wxsim.radar.render scenarios/piedmont_dryline_2027/radar_KFFC_2245z.json   # add --time 2027-04-17T23:00Z for another frame
python -m wxsim.nws_text scenarios/piedmont_dryline_2027/products/drafts/AFDFFC_draft.txt -o AFDFFC.txt
```

Output lands in `scenarios/<name>/output/` unless you pass `--outdir`.

## Starting a new scenario

1. Copy `scenarios/piedmont_dryline_2027` to `scenarios/<new_name>` and clear its `output/` and `products/`.
2. Edit the JSON files (each one has a `_doc` key explaining its format).
3. Run `python render_all.py scenarios/<new_name>`.

### What each JSON controls

| File | Key fields |
|---|---|
| `sounding_*.json` | `profile` rows `[p hPa, T °C, Td °C, dir °, speed kt]` (surface first), `elevation_m`, optional `modified_surface` (°F) for an afternoon-mixed profile, `annotations` (arrows/labels; `p` can be `"wbz"` / `"fzl"`; text can use `{wbz_kft}`, `{fzl_kft}`, `{lr75}`, `{tc_f}`). Also writes a SHARPpy-format text file. |
| `outlook_*.json` | `categorical` polygons (`TSTM` … `HIGH`), `probabilistic` levels per hazard with an optional `sig` hatch polygon. Polygons are rough `[lon, lat]` control points; they get smoothed, nested, and clipped to land. **The script warns if any probability contour spills outside the categorical risk it implies.** |
| `mcd_*.json` | Sidebar text, `intensities`, `banner`, MCD `polygon` (`[lat, lon]`), and an `environment` block (dryline, optional stable/wedge front, mesolows, towers, cirrus, MLCAPE and shear settings) that drives the synthetic satellite, CAPE contours, and shear barbs. Station `obs` rows: `[id, lat, lon, T°F, Td°F, dir, kt]`. Prints the `LAT...LON` block for the text product. |
| `watch_*.json` | `type` (`Tornado` / `Severe Thunderstorm`), `pds`, the SPC-style `axis` (miles/direction from two cities plus half-width), `state_fips` with optional `state_filters`, and the `probabilities` table. Prints the SEL axis sentence and WWP corner points to paste into the text products. |
| `hazards_*.json` | NWS-style active hazards map for one moment in time. Tropical additions: `county_hazards` entries can be `{"from_tc_file": "tc_*.json"}` (Hurricane/Tropical Storm Warnings from its `inland_ww`), `"style": "hatch"` draws an overlay (a Flood Watch under hurricane warnings), `"pds": true` adds PDS labeling and a banner, a Flash Flood Warning with `"emergency": true` becomes a Flash Flood Emergency, and `colors` overrides hazard colors per map. `watches` (county-based, can point at a `watch_*.json`), `county_hazards` (Red Flag, Wind Advisory, etc., counties as `"Name ST"`), and storm-based `warnings`. Give each warning a `storm` (issue position, motion, duration, back/front widths) and the polygon is built for you, or pass an explicit `polygon`. `elapsed_min` moves the storm marker to the map's valid time. `tags` = tornado / damage / hail / wind. Uses standard NWS hazard colors, shades counties by highest-priority hazard, and prints each warning's UGC line, LAT...LON, and TIME...MOT...LOC for the text product. |

## Storm generator (`wxsim/env/`)

This is a rule-based storm generator, not a numerical cloud model. You describe the **environment**, and storms initiate,
organize, move, split, merge, change character and die on their own. The rules come from the severe-storms
literature: CAPE/shear regimes, Bunkers motion, storm-relative anvil flow for LP↔HP (Rasmussen & Straka), boundary–shear
angle for linear vs. discrete modes, and STP/SHIP for tornado and hail potential.

```bash
python -m wxsim.env.generator scenarios/piedmont_dryline_2027/env_piedmont.json --seed 1   # one realization + maps + radar frames
python -m wxsim.env.generator scenarios/piedmont_dryline_2027/env_piedmont.json --ensemble 20   # 20 seeds, probabilities
python -m wxsim.env.generator scenarios/templates/env_template_gulf_qlcs.json --seed 1   # a very different day
python -m wxsim.env.generator scenarios/seaboard_outbreak_2027/env_seaboard.json --seed 1 --no-maps
python -m wxsim.mrms scenarios/seaboard_outbreak_2027/mrms_seaboard.json   # 2-km MRMS-style composite loop
```

The Seaboard renderer makes a gridded composite from storm primitives at several heights with no radar site or beam. It writes a GIF, 29 high-resolution frames and a pause/scrub HTML viewer under `scenarios/seaboard_outbreak_2027/output/mrms/`. See that scenario's `SCENARIO.md` for times, assumptions and ensemble commands.

**`env_*.json`** (each has a `_doc` key):
- `airmasses`: named parameter sets (MLCAPE, MUCAPE, MLCIN, LCL, PW, mean wind and 0–6 km shear as `[dir, kt]`, 0–1/0–3 km SRH,
  storm-relative anvil wind, 700–500 lapse rate, hodograph curvature 0–1, `elevated` 0–1, WBZ, `linear` tendency, heating-CI rate).
- `boundaries`: each puts its `left` air mass on the left side of the line as drawn, can add a moist `pool` band on its
  right side, and supplies `lift` (with a time curve) for initiation. You can keyframe a boundary's position by hand
  (`track`), or make it **dynamic** with a `dynamic` block (an `initial_line` plus rules), so it moves on its own:
  - `dryline`: synoptic drift (`synoptic_kt`), plus a daytime mixing "jump" (`jump_kt`, lagged about 2 h behind the sun).
    A `speed_max` bulge rides along a jet-streak track, anvil shading slows the dryline near storms, and after sunset it
    retreats west (`retreat_kt`).
  - `wedge_front` (CAD edge): erodes under daytime heating (`erode_day_kt` × `cloud_factor`) and at night
    (`erode_night_kt`, e.g. as the LLJ ramps up), is pushed back by nearby storms' rain-cooled air (`reinforce_kt`), and can be
    pushed with `push_kt`.
  - `cold_front` / `warm_front`: `speed_kt` (piecewise in time). A cold front can also `surge_kt` near convection, so it
    bows out where a QLCS's cold pool pushes it.
  Positive speed moves the line toward its right-hand side (the side opposite `left`). Nodes are smoothed with Taubin
  smoothing, so the line doesn't shrink, and resampled every step.
- `outflows`: **cold pools and outflow boundaries (OFBs) generated by the storms.** A surface-based storm older than 25 min
  feeds a pool just behind itself. Pool strength scales with precipitation, dry low-level air and high bases, and is
  weaker for supercells. Pools spread like gravity currents (`spread_kt` × √strength, slowing over `spread_taper_min` once
  no longer fed), drift with the mean wind (`drift_frac`), merge when they touch, and decay (`decay_min`).
  - *Inside a pool* the air is stabilized: CAPE drops, CIN grows by up to `interior_cin`, LCLs lower, and new storms
    become elevated.
  - *Along the edge (the OFB)* there is lift (`edge_lift`) and extra 0–1 km SRH (`edge_srh01`, the classic boundary
    tornado boost). Where an OFB runs into a dryline or front, the lifts add (`collision_lift`), making intersections
    initiation hot spots.
  - An organized supercell isn't undercut by its own pool, but it is undercut by other storms' pools.
- `regions` (polygon air-mass overrides), `mesolows` (point lift), `time_factors` (diurnal CIN erosion, evening SRH/LLJ
  ramp, falling LCLs, CAPE decay), `initiation` (rate, CIN threshold, spacing, window), and `evolution` (organization
  time, split rate, merge distance, tornado rate/refractory period, and `ef_scale`, which fattens (>1) or thins (<1) the
  EF-rating tail).

**Optional knobs added for the Seaboard super outbreak** (all default to the old behavior):
- `marine`: `{"airmass": "<name>", "blend_km": 30, "weight": 0.9}` blends a (stable) marine air mass over open water, so
  storms that cross the coast go elevated and stop producing tornadoes.
- cold front `dynamic.speed_by_lat`: `[[lat, factor], ...]` scales the front's speed by latitude (e.g. a lagging southern end).
- `outflows.strength` (pool strength multiplier) and `outflows.hard_area_factor` (merged pools fed by many storms are
  eroded once they pass this multiple of `max_area_km2`).
- `evolution`: `tor_end_base` / `tor_end_per_ef` / `tor_end_min` (per-step tornado end chance), `tor_max_min` (longest
  single tornado), `ef_overflow: "resample"` (redraw ratings past EF5 instead of piling them onto EF5), `qlcs_gust_bonus`.
- `wxsim.reports`: `land_only` (default true) drops reports over water. `data/places_us.csv` (name, state, lat, lon) is the
  gazetteer used for report locations.

**What happens each 10-minute step:**
- **Initiation:** Poisson draws wherever lift or heating can overcome CIN, with spacing suppression.
- **Organization and structure:** each storm's supercell organization, LP↔classic↔HP blend, hook/notch strength,
  elevated and high-based character, size and hail core relax toward what its *inflow* environment supports.
- **Motion:** Bunkers right/left movers; ordinary cells move with the mean wind.
- **Interactions:** left-moving splits in straight hodographs, mergers (the survivor turns HP), and upscale growth into QLCS segments where
  the environment favors lines.
- **Boundaries and outflow:** after the storms step, each storm feeds its cold pool, and the pools spread, drift, merge
  and decay. Each dynamic boundary then advances. Every environment sample from then on (initiation, inflow,
  mesoanalysis) sees the updated lines and pools.
- **Decay and hazards:** decay when inflow runs out or the storm reaches its lifetime; tornado episodes from inflow STP (with
  mesocyclone-cycling refractory periods); peak hail from SHIP.

**Outputs:** `storms_gen_seedN.json` (radar and hazards tools read it directly, so warnings can reference generated storms by
`storm_id`), mesoanalysis panels (`MESO_*.png`), a track summary with tornado/hail markers (`GEN_tracks_seedN.png`), an hourly
**boundary-evolution** panel set (`GEN_boundaries_seedN.png`: boundaries, their position an hour earlier, cold pools/OFBs
and live storms), KFFC radar frames, and with `--ensemble`, `ensemble_summary.json` with P(tornado), P(EF2+) and hail counts.

The generated storms file also has a `boundaries` block: every dynamic line's position through time, plus pool outlines
every step. The radar reads it to draw **fine lines**: weak (roughly 5–20 dBZ), grainy, broken returns from insects and
refractivity along the dryline, fronts and OFBs. They only appear where the 0.5° beam is still in the boundary layer
(within about 100 km of the radar). Turn them off with `"fine_lines": false` in the radar config.

Generated storms use the continuous `procedural` archetype by default, and legacy archetypes still work in hand-written
storm files. The opt-in `dynamic` mode below evolves reflectivity from storm state instead of assembling a primitive list.

The `procedural` archetype accepts optional, continuously blended morphology controls. Each is clamped to `0..1`;
omitting them (or leaving them at `0`) preserves the legacy primitive list and appearance. They can be set in `params`
or varied through `keyframes`:

| Parameter | Radar appearance |
|---|---|
| `line` | Grows an isolated supercell into a cell embedded in a broken line and then a linear segment. The forward-flank precipitation broadens into a trailing transition/stratiform zone; mid-range values keep a remnant hook/notch. |
| `bow` | Curves the line into a bow echo. It adds a rear-inflow notch and weak-echo channel behind the apex, with a comma-shaped bookend vortex at the positive-`v` end. Combined `line` + `bow` can show LEWP-like kinks. |
| `tor` | Tightens and wraps the hook, sharpens the inflow notch, and adds a compact high-reflectivity debris ball below 1.5 km. The generator ramps this around tornado episodes. |
| `cycle` | A cyclic mesocyclone phase: the hook occludes and the RFD surges near phase `0.5`, then relaxes as a new hook forms. Phase `0` and `1` meet smoothly; at the default 10-minute cadence, the generator lengthens cycles to about 22–30 minutes to avoid phase aliasing. |
| `lowtop` | Compresses the storm vertically to roughly 7–9 km, reduces its size and anvil, and keeps a compact hook. It can be combined with HP structure for cool-season and coastal storms. |
| `split_core` | A brief shared core pulse on both members as a left-moving split begins. The caller mirrors the left mover. |
| `merge` / `merge_fade` | The survivor briefly broadens while the absorbed core weakens through its short visual fade. These are generator interaction fields. |

The WSR-88D sampler adds a weak three-body hail spike behind sampled cores above about 65 dBZ. Its radial direction is
computed from the radar site, so it is not a storm primitive and does not appear in the site-independent MRMS-style mosaic.

Render the full labeled morphology gallery (dark background, simulated base reflectivity):

```bash
python -m wxsim.radar.gallery
```

The default output is `scenarios/seaboard_outbreak_2027/output/radar_morphology_gallery.png`; pass `--out PATH` to choose another path.

### State-driven dynamic storms

Use `--morphology dynamic` to generate storms whose reflectivity is evaluated directly from storm age, the changing
environment, and a deterministic seeded state field. The same dynamic storm definition works in the WSR-88D sampler and
the site-independent MRMS-style mosaic. This is a lightweight reduced-order simulation, not a full atmospheric fluid
solver. It adds no predefined archetype/primitive recipe to the dynamic path.

```bash
python -m wxsim.env.generator scenarios/seaboard_outbreak_2027/env_seaboard.json --seed 1 --no-maps --morphology dynamic
python -m wxsim.mrms scenarios/seaboard_outbreak_2027/mrms_seaboard.json --storms-file scenarios/seaboard_outbreak_2027/storms_gen_dynamic_seed1.json --outdir scenarios/seaboard_outbreak_2027/output/dynamic_mrms --time 2027-04-29T18:00Z
```

The generated file is `scenarios/seaboard_outbreak_2027/storms_gen_dynamic_seed1.json`; the normal procedural output
keeps its own filename. Omit `--no-maps` to render WSR-88D frames from that dynamic file using the scenario's radar
template. The MRMS command selects the same file and writes to a separate output folder. Use
`python -m wxsim.radar.dynamic_gallery` for one storm at successive ages and for its base and height-composite views.

## Simulated radar (`wxsim/radar/`)

The default procedural mode assembles shape primitives (cores, forward-flank plumes, spiral hook appendages, inflow
notches, anvils, lines) in each storm's relative frame. The optional dynamic mode evaluates a seeded evolving field
without that primitive list. Both are **sampled like a single WSR-88D**: 0.5°-wide beam on a polar grid (0.5° × 250 m gates),
4/3-earth beam height, beam smoothing, and near-radar ground clutter. Each primitive has a vertical extent, so
the beam only sees what's at its height. That's why far-range storms lose their hooks, LP forward flanks (virga)
vanish from base reflectivity but show up in composite, and the anvil shows up in composite.

| File | Job |
|---|---|
| `storms.json` | **Single source of truth for storms.** Each storm has an `archetype`, an `anchor` (time + lat/lon), a `motion`, a `split` (right/left), `params` (`scale`, `dz`, `hail`, `ffd_rot`, `anvil`, plus the morphology controls above), and optional `life` (start/peak/decay/end) and `evolve` (legacy archetype changes over time). The `dynamic` archetype also reads keyframed inflow state such as `mlcape`, `shear_kt`, `rh_low`, and `srh01`. Hazards maps can reference storms by `storm_id` so warnings, markers, and radar all line up. |
| `radar_*.json` | `site`, `time`, `products` (`base`, `composite`), `view`, `extra_views` (zoomed crops), `overlays` (`hazards_file` for warning polygons + watch outline, `storm_ids`, `range_rings`), `clutter`, `seed`. |

Archetypes: `classic_supercell`, `lp_supercell`, `hp_supercell`, `elevated_hail`, `high_based_cell`, `multicell`,
`pulse`, `qlcs_segment`, `stratiform` (see `wxsim/radar/archetypes.py`). The `arc` primitive draws a tapered,
curved comma/bookend segment; like the existing primitives, its horizontal dimensions pass through `_sc()` and left
movers are mirrored by the storm caller. Add archetypes by writing a function that returns primitives and registering it
in `ARCHETYPES`. The `dynamic` storm type is evaluated by `wxsim/radar/dynamic.py` and has no primitive-list entry.

A hazards map can put radar underneath with `"radar_underlay": {"config": "radar_*.json", "product": "base"}`.

**Radar loops (GIF):**
```bash
python -m wxsim.radar.loop scenarios/piedmont_dryline_2027/radar_KFFC_2245z.json --start 2027-04-17T20:00Z --end 2027-04-18T00:00Z --step 10
```
Options: `--view athens_zoom` loops a named extra view; `--product composite`; `--width`; `--frame-ms`; `--keep-frames DIR`
keeps the PNGs. One RadarSim is reused for every frame, so storm texture stays steady. The GIF uses a shared palette with
the exact reflectivity colors pinned in. In an `env_*.json`, `maps.radar_loop` (`start`, `end`, `step_min`) makes the
generator write `GEN_seedN_<site>_BREF_loop.gif` automatically. Generated frames also show a red inverted-triangle
TVS-style marker on storms with an ongoing tornado (`overlays.tornado_markers`).

Still to come: hail and rotation swaths (Phase 2), and velocity couplets plus dual-pol CC/ZDR (Phase 3).

### Tips

- **Physical consistency is on you:** CAPE, shear, and LCLs should all describe the same environment. Run the
  sounding script first. Its console summary (SBCAPE/CIN, PW, lapse rates, convective temp, SRH, shear,
  SHIP) is the source of truth that the text products should quote.
- For text products, write the draft with **one paragraph per line**, then run `wxsim.nws_text` to wrap it to
  69 characters. Headers, tables, and LAT...LON blocks are left untouched.
- The map scripts use a Lambert conformal projection centered on each product's extent, so they work anywhere
  in the CONUS. Change `cities` in each JSON when you move regions (the outlook default is a Southeast list).

## Tropical cyclones (`wxsim/tropical/`)

One `tc_*.json` per NHC advisory drives every tropical product (see `scenarios/delphine_2027/`).

```bash
python -m wxsim.tropical.cone     scenarios/delphine_2027/tc_delphine_adv18.json   # 5-day cone graphic
python -m wxsim.tropical.products scenarios/delphine_2027/tc_delphine_adv18.json   # TCMATx + TCPATx text
python -m wxsim.tropical.coast --snap                                              # re-snap data/tc_breakpoints.json
```

| Key | Contents |
|---|---|
| `name`, `number`, `basin`, `year`, `wmo_slot`, `tz` | Identity; ATCF id (`AL072027`) and WMO headers (`WTNT2x`/`TCMATx`, `WTNT3x`/`TCPATx`) are derived. |
| `advisory` | `number`, `time` (UTC), `motion` (optional; otherwise from the last fix), `next_intermediate`/`next_complete`, `forecaster`, `eye_diameter_nmi`, `position_accuracy_nmi`. |
| `past_track` | Rows `[time, lat, lon, vmax_kt, mslp, status]`, drawn dashed. |
| `forecast` | `tau` 0/12/24/36/48/60/72/96/120 with `lat`, `lon`, `vmax_kt`, `gust_kt`, `status` (`TD/TS/HU/MH/PT/LO/DS`), `inland`, and `radii` (`34`/`50`/`64` nmi by NE/SE/SW/NW; `12ft` seas at tau 0). Tau 0 is the advisory time; others count from the synoptic hour, as NHC does. The loader checks status vs. wind and radii consistency. |
| `coastal_ww` | `{type: HU.W/HU.A/TR.W/TR.A, from, to}` using breakpoint names from `data/tc_breakpoints.json` (TX to NC, snapped to the coastline, so segments follow the shore). |
| `inland_ww` | Optional county lists (`"Name ST"`) shaded on the cone; turn off with `cone.show_inland: false`. |
| `cone` | `radii_nmi` by tau (**illustrative, configurable**), `extent`, `cities`, `output`. |
| `text` | TCP headlines, track/intensity narrative, inland-warning note, and hazard blocks. |

Cone = union of the forecast-point circles plus the convex hull of each consecutive pair; days 1-3 solid, days 4-5 stippled.
Planned for later: TWO hatched areas, wind-speed probabilities/arrival times from a perturbed-track Holland ensemble, and a link from a landfalling TC to the storm generator and radar. Each would be a new module reading `wxsim.tropical.storm.Storm`.

### Wind swath and local warning maps

- `tcwind_*.json` → `python -m wxsim.tropical.windmap`: deterministic peak sustained wind and gust swath, built from the advisory's quadrant wind radii interpolated through time (RMW, tail and gust factors are configurable).
- `tcwarn_*.json` → `python -m wxsim.tropical.local`: zoomed office-area map of inland warnings by county, with hourly center positions.

## WPC products (`wxsim/wpc/`)

All WPC products read one rule-based rainfall field (`wxsim/wpc/rain.py`): predecessor-rain bands, TC rain sampled hourly along a storm file's track (slower storms rain more), and orographic multipliers.

```bash
python -m wxsim.wpc.qpf scenarios/delphine_west_2027/qpf_delphine_west.json      # rainfall map panels
python -m wxsim.wpc.ero scenarios/delphine_west_2027/ero_day1_16z.json          # ERO graphic + QPFERD text
python -m wxsim.tropical.products scenarios/delphine_west_2027/tc_delphine_west_adv22_wpc.json   # WPC post-tropical TCP
```

- **ERO** risk = window rainfall / flash flood guidance, with FFG lowered by antecedent (predecessor) rain; `ratio_thresholds` map that ratio to Marginal/Slight/Moderate/High, or pass `manual` polygons.
- **WPC advisories:** set `advisory.center: "WPC"` in a tc_*.json. You get the KWNH header and WPC office line, no TCM and no cone, and optionally `text.rain_reports`, a storm-total rainfall list by state sampled from the same field.

## Changelog

### 2026-10-11

- `wxsim.hazards`: new hazard types with standard NWS colors and priority — Extreme Wind Warning (polygon or county),
  High Wind Watch, Blizzard Warning, Ice Storm Warning, Winter Storm Warning/Watch, Winter Weather Advisory, and
  Coastal Flood Warning/Watch/Advisory. Watches and county hazards can now pick counties with a `polygon`
  (`[lat, lon]` points, by county interior point) plus an optional `exclude` list, so large regional hazards don't
  need hand-typed county lists. Existing maps render unchanged.
- `wxsim.tropical.storm`: new `EX` (extratropical low) status for non-tropical storms that reuse the storm-file schema.
- `wxsim.tropical.windmap`: optional `modifiers` (feathered multiplier zones, e.g. a cold-air-damming wedge or a
  sting jet), `rmw_nm.extratropical`, `track_labels` (which track points to label), `cbar_label` and `footnote`.
  Existing swaths render unchanged.
- `wxsim.wpc.ero`: optional `exclude` ([lon, lat] rings) removes risk where the rain field doesn't apply, e.g. where
  precipitation falls as snow or ice.
- New scenario `scenarios/apalachee_soueaster_2026/`: a 939 mb Gulf bomb cyclone that comes ashore in the Big Bend
  and loops over southwest Georgia — wind swath, landfall hazards map, WPC QPF and ERO, SPC Day 2 outlook, and TAE
  AFD / High Wind Warning / Extreme Wind Warning text.

### 2026-09-24

- Added `wxsim/wpc/` (rainfall field, QPF maps, ERO), WPC post-tropical advisories, the tropical peak-wind swath and local inland-warning maps, and the fast/west Delphine variants.

- Added `wxsim/tropical/` Phase 1: storm JSON schema, NHC-style 5-day cone, TCM and TCP text products, coastal breakpoints, and the Hurricane Delphine example scenario.

### 2026-09-23

- Added continuous `line`, `bow`, `tor`, `cycle`, and `lowtop` radar morphology, generator-driven split/merge transitions,
  and a radar-site-aware hail spike. Old storms files keep their prior defaults; the morphology gallery is available with
  `python -m wxsim.radar.gallery`.
- Added an opt-in state-driven `dynamic` storm field for WSR-88D and MRMS-style rendering, plus an age-sequence gallery.
