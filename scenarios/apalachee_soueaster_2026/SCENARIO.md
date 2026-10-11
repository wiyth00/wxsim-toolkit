# Apalachee Bay Sou'easter (Dec 15–19, 2026)

Fictional. A record El Niño winter: an extremely strong subtropical jet, a Gulf left unusually warm because no
hurricane mixed it, and a brief −NAO block that drives a 1046 mb Arctic high into Quebec and a strong
cold-air-damming wedge down the east side of the Appalachians. A Gulf low phases with a northern-stream wave and
bombs, comes ashore in the Florida Big Bend, then loops over southwest Georgia instead of turning north like a
nor'easter. Its exit off the Georgia coast gives the coast a long southeast gale, hence "sou'easter".

Hour 0 = Tue Dec 15 00Z. December, so local time is EST (UTC−5).

## Canon

| | |
|---|---|
| Pressure | 1006 mb Tue 12Z → 997 Wed 06Z → 962 Wed 18Z → **939 Thu 03Z** (offshore) → **941 at landfall** near St. Marks FL, Thu 06Z → 951 Thu 18Z → 965 Fri 18Z → 978 Sat 18Z |
| Deepening | 35 mb in 12 h (Wed 06–18Z); 56 mb in 24 h ≈ 4 Bergerons at 28–30°N |
| Track | Western Gulf → St. Marks → cyclonic loop over SW Georgia (Thu–Fri) → exits off Brunswick GA Sat |
| Structure | Shapiro–Keyser: frontal fracture, bent-back front, warm seclusion, sting jet over south Georgia Thu ~5–9 AM EST |
| Wind (swath) | Peak gusts: Perry 126, Valdosta 125, Tallahassee 110, Moultrie 108, Cedar Key 105, Dothan 103, Albany 102, Brunswick 100, Savannah 97, Waycross 97, Panama City 96, Jacksonville 95, Apalachicola 86, Tampa 82, Charleston 77, Macon 73, Augusta 60, Mobile 58, Atlanta 44 (wedge), Birmingham 44 mph |
| Surge | Coastal Flood Warning: 10–15 ft above ground Keaton Beach–Steinhatchee, 8–12 ft St. Marks–Econfina, 4–7 ft Franklin County; multi-tide flooding on the GA/SC coast |
| Convection | Squall line with embedded supercells across the Big Bend and Florida peninsula Wed afternoon–night; SPC Day 2 Enhanced Risk |
| Rain | 6–10 in. Big Bend/south Georgia, locally 12 near Valdosta–Waycross, on soils saturated by a record-wet El Niño autumn (FFG 1.5–2.5 in./3 h). WPC: Day 1 (Wed) Moderate Risk; Day 2 (Thu 12Z–Fri 12Z) **High Risk** from Dothan and Albany through Tallahassee and Valdosta as the low loops; **PDS Flood Watch** across the TAE area (FFATAE) stressing floodwaters full of live power lines, debris and sewage |
| Winter | Ice Storm Warning across most of the wedge: north and central Georgia including Atlanta, the Upstate and Midlands of SC, and the NC Piedmont. Winter Storm Warning to the north (north Alabama, NE Mississippi, Tennessee, the southern Appalachians, southern Virginia: 12–24 in. snow, thundersnow). Winter Weather Advisory for central and east-central Alabama (including Birmingham, near the dry slot) and Mississippi: mostly sleet, light ice. Ice accretion: 1.2–1.5 in. (locally 1.8) Upstate SC, 1.0–1.2 NE Georgia, 0.75–1.0 Charlotte, 0.6–0.9 Atlanta. |

## Files

| File | Product |
|---|---|
| `storm_apalachee.json` | Storm file in the tropical schema with the new `EX` (extratropical) status; WPC Storm Summary 4, Wed 18Z. Quadrant radii keep hurricane-force winds on the southeast flank. |
| `wind_apalachee.json` | `wxsim.tropical.windmap` peak sustained/gust swath, with `modifiers` for the CAD wedge (×0.62) and sting jet (×1.08) |
| `hazards_landfall_06z.json` | Hazards map at landfall (1 AM EST Thu): Extreme Wind Warning polygon, High Wind Warning, Wind Advisory, Ice Storm, Winter Storm, Winter Weather Advisory, Coastal Flood Warning (hatched), Tornado Watch |
| `qpf_apalachee.json` | Storm-total liquid equivalent (`wxsim.wpc.qpf`) |
| `ero_day1_16z.json` | WPC Day 1 ERO 16Z Wed + QPFERD text (`wxsim.wpc.ero`) |
| `ero_day2_21z.json` | WPC Day 2 ERO issued Wed afternoon, High Risk for Thursday; uses `exclude` to drop the snow/ice areas and the post-squall-line peninsula |
| `outlook_day2_1730z.json` | SPC Day 2 outlook issued Tue 1730Z (`wxsim.outlook`) |
| `products/` | AFDTAE (342 PM Wed), NPWTAE High Wind Warning (345 PM Wed), FFATAE PDS Flood Watch (350 PM Wed), EWWTAE Extreme Wind Warning (1240 AM Thu), SWODY2 (Tue). Drafts in `products/drafts/`, wrapped with `wxsim.nws_text`. |

```bash
python -m wxsim.tropical.windmap scenarios/apalachee_soueaster_2026/wind_apalachee.json
python -m wxsim.hazards          scenarios/apalachee_soueaster_2026/hazards_landfall_06z.json
python -m wxsim.wpc.qpf          scenarios/apalachee_soueaster_2026/qpf_apalachee.json
python -m wxsim.wpc.ero          scenarios/apalachee_soueaster_2026/ero_day1_16z.json
python -m wxsim.outlook          scenarios/apalachee_soueaster_2026/outlook_day2_1730z.json
```

## Liberties taken

- **Extreme Wind Warning.** Real EWW criteria are sustained winds of 115+ mph, normally in a major hurricane's
  eyewall. This storm's sustained winds top out near 85–90 mph with gusts to ~125 mph. The scenario has TAE issue one
  anyway, and the product calls it the first EWW for a non-tropical storm. A PDS-style High Wind Warning would be
  the by-the-book choice.
- **Extratropical "advisories".** WPC issues Storm Summaries for big winter storms, not wind-radii advisories. The
  storm file reuses the tropical schema so the swath and rain modules can read it.
- Zone codes (FLZ/GAZ/ALZ/GMZ) in the text products are plausible, not checked against the real zone tables.
