# Piedmont Dryline Event: Saturday, April 17, 2027 (simulated)

This is a hybrid dryline setup over the Georgia Piedmont. A Mexican Plateau EML was advected across the northern
Gulf and now sits over shallow, modest moisture (sfc Td upper 50s, PW ~0.7"). That gives Plains-style
thermodynamics in the Southeast, with LP supercells, giant hail, dry microbursts, and a narrow tornado corridor
along a retreating (later stalled) cold-air-damming wedge front.

## Key features
- **Aloft:** a negative-tilt shortwave over the Ozarks and Mid-South, a 75–85 kt 500 mb speed max into AL/GA, and a
  coupled jet (right entrance of a 130 kt polar jet plus left exit of the subtropical jet).
- **Surface:** a 991 mb low in western KY deepening toward southern IN, with the cold front lagging in MS/western AL.
- **Hybrid dryline:** combines a Pacific front/dry slot, diurnal mixing into the EML, and downslope drying off the
  Cumberland Plateau (with a lee mesolow in NW GA). It is along the AL/GA line at 18Z with a bulge near
  Carrollton/LaGrange, and near Americus–Griffin–west metro–Calhoun by 21Z.
- **Wedge front:** Gainesville–Athens–Washington–Augusta. Pooled 62–65°F dewpoints and backed SE flow sit
  just south of it (the tornado corridor).
- **12Z FFC:** moist layer ~70 mb deep, cap at 832 mb, 700–500 mb lapse rate 8.4 °C/km, SBCIN −620, conv. temp ~86°F,
  WBZ 8.1 kft. Modified to 86/58: MLCAPE ~2500, ML LCL ~1900 m, DCAPE ~1050, SHIP ~3.

## Timeline (EDT)
| Time | Event |
|---|---|
| 8:24 AM | FFC AFD |
| 7:52 AM CDT | SPC 1300Z Day 1 (Moderate Risk, hail-driven) |
| 1:05 PM CDT | MCD 0487 (Tornado Watch likely, 95%) |
| 2:40 PM | First supercell near Carrollton, 3.5" hail |
| 2:50 PM | PDS Tornado Watch 124 until 11 PM EDT |
| 6:45 PM | Hazards snapshot: 2 Tornado Warnings (Athens area, radar-confirmed/considerable; Anderson SC), 3 Severe Thunderstorm Warnings (4" destructive hail near Madison; elevated hail near Gainesville; 70 mph dry microburst near Thomaston) |
| 6–11 PM | Tornado window along the stalled wedge front (Athens–Commerce–Hartwell–Upstate) |

## Products in this folder
- `products/`: final text products (AFD, SWODY1, MCD 0487, SEL4/WW 124, WWP4). `products/drafts/` has the
  one-paragraph-per-line drafts used with `python -m wxsim.nws_text`.
- `storms.json`: the six storms (S1 to S6) driving both the radar and the warning polygons. `radar_KFFC_2245z.json`: the KFFC base and composite frames plus an Athens zoom.
- `env_piedmont.json`: the environment (air masses, dryline, wedge front, capped south, diurnal factors) for the storm generator.
  - **The dryline and wedge front are dynamic.**
    - Dryline: mixes east through the afternoon (about 84.1°W at 33.4°N by 21Z), bulges under the jet streak, slows
      under anvils, and retreats slowly after dark.
    - Wedge front: erodes north under weak heating, then stalls near the Athens–Commerce–Hartwell axis in the evening as
      storm outflow reinforces it.
  - **Storms make their own cold pools and OFBs.** High-based storms in the dry sector leave small outflow rings. The
    long-lived supercells near Athens build an MCS-scale pool through the evening, which stabilizes the area behind them
    and pushes later cells toward elevated modes. About 1 in 10 storms initiates on an OFB, most often where one runs
    into the dryline or wedge front.
  - **20-member ensemble** (`tor_rate` 0.03, `ef_scale` 1.2): P(any tornado) ≈ 80%, P(EF2+) ≈ 45%, about 5–6 storms
    with 2"+ hail per run. That supports the PDS Tornado Watch alongside the hail-driven Moderate Risk.
  - **Seed 1:** EF2 (G05, 2100Z), EF5 (G03, 2300–2340Z, east of Augusta beyond KFFC's range) and EF0 (G11, 0140Z).
    `GEN_seed1_KFFC_BREF_loop.gif` is the 1930–0130Z KFFC loop.
  - **Figures:** `GEN_boundaries_seedN.png` shows hour-by-hour boundary and pool evolution. Generated radar frames show
    the boundaries as fine lines near KFFC.
- JSON inputs for every graphic; run `python render_all.py scenarios/piedmont_dryline_2027`.
