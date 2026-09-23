# wxsim — weather scenario graphics toolkit

Reusable generators for simulated severe-weather scenarios: Skew-T soundings, SPC-style convective
outlooks (categorical + tornado/wind/hail probabilities), Mesoscale Discussion graphics, watch graphics,
and a text wrapper for NWS-style products. **All scenario data lives in JSON files; the code never
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

### Tips

- **Physical consistency is on you:** CAPE, shear, and LCLs should all describe the same environment. Run the
  sounding script first. Its console summary (SBCAPE/CIN, PW, lapse rates, convective temp, SRH, shear,
  SHIP) is the source of truth that the text products should quote.
- For text products, write the draft with **one paragraph per line**, then run `wxsim.nws_text` to wrap it to
  69 characters. Headers, tables, and LAT...LON blocks are left untouched.
- The map scripts use a Lambert conformal projection centered on each product's extent, so they work anywhere
  in the CONUS. Change `cities` in each JSON when you move regions (the outlook default is a Southeast list).
