# Atlantic Seaboard Super Outbreak: Thursday, April 29, 2027 (simulated)

A classic cool-season "mega" outbreak for the Mid-Atlantic and Carolinas. A deep, negatively tilted trough and a
110–130 kt midlevel jet sit over a very moist warm sector (Td upper 60s–low 70s F) with extreme low-level shear. The
day has two main threats: long-track discrete tornadic supercells over the coastal plain of the Carolinas and SE
Virginia, and an intense, fast-moving QLCS that runs up the I-95 corridor to Philadelphia.

## Key features
- **Aloft:** negative-tilt trough in the Ohio Valley; 110–130 kt 500 mb jet streak over NC/VA by 18z; strong difluence.
- **Surface:** 984 mb low in central PA deepens to about 976 mb over northern NY by 00z. The warm front surges from
  Richmond and the southern Delmarva to near Philadelphia by 20z. The cold front crosses the Appalachians, with a
  pre-frontal trough over the western Piedmont.
- **Warm sector:** MLCAPE 1500–2500 (Carolinas) and 1000–1500 (VA/MD); effective shear 70–90 kt; 60–75 kt LLJ;
  0–1 km SRH 300–500; STP 4–8; LCLs 500–800 m. Storms move north-northeast at 50–60 kt.

## 1300Z Day 1 outlook (Moderate Risk)
- **Area:** a Moderate from SE Georgia through the Carolinas, east-central VA, MD and DE into SE PA/Philadelphia.
- **Tornado:** 15% with Intensity 2 (EF3+) from Florence/Fayetteville–Raleigh–Richmond to Hampton Roads and the lower Delmarva.
- **Wind:** 45% with Intensity 2 (85+ mph) along the whole corridor.
- **Hail:** 30% (no intensity group) only over GA/SC.

## Products in this folder
- `outlook_day1_1300z.json` and output/SPC_Day1_*_2027042913.png
- `outlook_day1_1430z.json` and output/SPC_Day1_*_2027042914.png: special update with two High Risk areas.
- `outlook_day1_1630z.json` and output/SPC_Day1_*_2027042916.png: routine update maintaining both High Risk areas as the threats mature.
- `products/SWODY1.txt`, `products/SWODY1_1430z.txt`, and `products/SWODY1_1630z.txt` (drafts in `products/drafts/`).
- `mcd_0611.json` and `mcd_0612.json`, with matching text and graphics.
- `watch_0157.json` and `watch_0158.json`, with matching SEL7/SEL8 texts, drafts and graphics. Watch 157 is a broad PDS Tornado Watch from coastal GA through the Carolinas into southeast VA, issued at 1450Z and valid to 0000Z. The greatest tornado threat is within its eastern NC/SE VA High Risk core. Watch 158 is a broad PDS Severe Thunderstorm Watch from central VA across the Mid-Atlantic into southeast PA and southern NJ, issued at 1510Z and valid to 2300Z. The greatest destructive wind threat is within its I-95 High Risk core. The watches encompass surrounding Moderate Risk areas and contain distinct VA counties. All products are simulated.

## Simulation environment and MRMS-style loop (super outbreak retune)

`env_seaboard.json` covers coastal Georgia through southeastern Pennsylvania, 1230Z Apr 29 to 0030Z Apr 30 in 10-minute steps.
It was retuned from the first ("meh") version into a true super outbreak:

- **The warm sector no longer collapses at midday.** The Appalachian cold front used to end at the NC/VA line, and the
  generator extends a boundary's cold side past its endpoint, so the post-frontal air mass spread south across the
  Carolinas by 18Z. The front now continues southwest through Georgia and Alabama, and its southern end lags
  (`speed_by_lat`), so it only catches the prefrontal storms in the evening.
- **Storms have room to run.** The Carolina prefrontal trough starts inland (Augusta–Columbia–Charlotte–Greensboro)
  instead of along the coastal plain, so NE-moving supercells cross 150–250 mi of land before they reach the coast.
- **Many rounds of storms.** Open warm-sector heating CI adds discrete supercells ahead of the trough. Four mesolows
  (Midlands, Sandhills/Lumberton, Goldsboro, Southside VA triple point) fire in sequence, and CI continues until 2230Z.
- **Richer air mass and an evening LLJ ramp.** Carolina MLCAPE is 2600 J/kg with 0–1 km SRH around 460 m²/s², and SRH
  rises another 25–30% after 21Z while LCLs fall. Mid-Atlantic MLCAPE is 1600 J/kg.
- **Stable ocean.** A marine-layer air mass over the Atlantic keeps offshore storms from producing tornadoes. Storm
  reports are only generated over land.
- **Weaker, capped cold pools.** On a discrete-supercell day, outflows are weak and merged pools can't blanket the
  warm sector.
- **Tornado behavior.** Tornadoes last longer (with a 130-minute cap per tornado), EF ratings past EF5 are redrawn
  instead of piling up at EF5, and the QLCS produces stronger gusts.

**Seed 1 (representative):** 124 storms. 155 tornadoes (152 touch land): 60 EF0, 45 EF1, 23 EF2, 13 EF3, 7 EF4 and 4 EF5.
13 paths are 50 mi or longer; the longest is about 129 mi. Peak QLCS gust is about 103 kt. SPC-style reports
(`reports_seaboard.json` → `output/storm_reports_20270429.png/.csv`, with the 1630Z outlook overlaid) show 349 tornado,
654 wind and 144 hail reports. `output/storm_reports_before_after.png` compares them with the original run
(53 / 134 / 36).

**Five-member ensemble** (`output/ensemble_summary.json`): 106–128 storms, 125–191 tornadoes, 36–70 EF2+, EF5 in every
member. The extreme tail is deliberate for this fictional scenario and is not a calibrated real-world probability.

`mrms_seaboard.json` uses seed 1 on a 2 km grid (0.5/1.5/3.5/5.5 km slices) and renders a composite every 20 minutes
from 1300Z to 0020Z (35 frames). Open `output/mrms/viewer.html` to play, pause and scrub, or use `output/mrms/Seaboard_MRMS_loop.gif`.
The composite is built from storm primitives, with no beam geometry or MRMS QC. No live weather data is used.

Note: Watch 157 runs to 0000Z and Watch 158 to 2300Z, but storms now continue until about 0030Z, so a late-evening
replacement watch would be a natural next product.

```bash
python -m wxsim.env.generator scenarios/seaboard_outbreak_2027/env_seaboard.json --seed 1 --no-maps
python -m wxsim.env.generator scenarios/seaboard_outbreak_2027/env_seaboard.json --ensemble 5 --no-maps
python -m wxsim.reports scenarios/seaboard_outbreak_2027/reports_seaboard.json
python -m wxsim.mrms scenarios/seaboard_outbreak_2027/mrms_seaboard.json
```

The model is a stochastic conceptual storm generator, not a numerical forecast. Its EF ratings and gusts are scripted
statistical hazards.
