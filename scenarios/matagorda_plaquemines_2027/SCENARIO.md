# Matagorda Bay to lower Plaquemines — coastal-strip variant

*** MOCK / SIMULATION ONLY — NOT AN OFFICIAL NWS PRODUCT ***

Fictional February 24–25, 2027. Separate variant of `gulf_coastal_highrisk_2027`; earlier files are preserved.

HIGH is a 23-mile metric buffer on each side of an approximate coastal corridor, then smoothed and clipped to land. Actual drawn width varies; water portions are omitted. Houston's inland core, Baton Rouge and central New Orleans are outside HIGH. Matagorda, Galveston, Cameron, coastal Terrebonne/Lafourche and lower Plaquemines are the focus.

30% tornado + Intensity 2 supports HIGH. The Intensity 3 core is smaller and west of the delta. Surrounding tornado contours and intensity areas are paired with the categorical bands. Probability is within 25 miles of a point, not the probability of a direct hit.

The mechanism is repeated initiation in a narrow surface-based coastal warm sector. No single storm is assumed to follow the entire curved coastline. The eastern HIGH requires independent surface-based supercells reaching lower Plaquemines; a primarily offshore or rapidly linear event would not justify this footprint.

All meteorological values and timing are invented. Widths describe the schematic scenario rather than observational or model precision.

NOAA/SPC probability–intensity matrix: https://www.weather.gov/media/rah/ConditionalIntensityPresentation.pdf

Render: `python3 -m wxsim.outlook scenarios/matagorda_plaquemines_2027/outlook_day1_1300z.json`
