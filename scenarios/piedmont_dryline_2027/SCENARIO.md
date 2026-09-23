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
| 6–11 PM | Tornado window along the stalled wedge front (Athens–Commerce–Hartwell–Upstate) |

## Products in this folder
- `products/`: final text products (AFD, SWODY1, MCD 0487, SEL4/WW 124, WWP4). `products/drafts/` has the
  one-paragraph-per-line drafts used with `python -m wxsim.nws_text`.
- JSON inputs for every graphic; run `python render_all.py scenarios/piedmont_dryline_2027`.
