# Analog review and calibration limits

**SIMULATED SCENARIO — v3 is not quantitatively calibrated.** These primary-source historical facts provide a mechanism and magnitude check, not validation of the fictional December 2026 event.

| Historical source | Evidence used | Implication for this model |
| --- | --- | --- |
| [Clark & Gray (2018), sting-jet review](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3267) | A descending sting jet is distinct from the cold conveyor belt; boundary-layer processes control whether it produces extreme surface winds. | Surface mixing is explicit. A separate moving descent proxy replaces the broad fixed lobe, but prescribed geometry does not diagnose parcel descent. |
| [NWS Tallahassee: Michael (2018)](https://www.weather.gov/tae/HurricaneMichael2018) | Michael’s landfall pressure was 919 mb; Tyndall recorded a 139 mph gust before sensor failure. Donalsonville reported 115 mph well inland. | Extreme inland gusts and missed peaks are possible. A tropical analog does not determine the wind field of a 939 mb extratropical cyclone. |
| [NWS Tallahassee: Idalia (2023)](https://www.weather.gov/tae/HurricaneIdalia2023) | The account describes frequent 80–90 mph gusts in heavily affected areas and widespread 60–70 mph gusts from Valdosta to Waycross; some stations missed peaks due to failure or communications loss. | Exposure and outage effects matter. This scenario’s roughly 120–137 mph synthetic south Georgia station peaks are an imposed extreme case, not fitted Idalia behavior. |
| [NWS Jackson: March 1993 Superstorm](https://www.weather.gov/jan/superstorm_march_1993) | The account gives a 975 mb Florida landfall, 960 mb later minimum, and gusts of 109 mph at Key West and 90 mph at Myrtle Beach. | Convective and broad coastal gusts must be separated from inner-core wind assumptions. A single pressure-to-gust conversion is inadequate. |

The v3 gust closure, jet-speed curve, quadrant radii, inversion fields, exposure spread and outage probabilities remain scenario choices. Similar magnitude to an observed gust does not establish calibration, and a historical station gust is not a spatial storm maximum.

A rigorous next pass would:

1. Ingest quality-controlled station wind/pressure time series and consistent best tracks for Michael, Idalia and the 1993 cyclone, preserving instrument height, gust averaging duration, exposure and outage flags.
2. Supply aligned terrain and land-cover data, then reconstruct each analog in its actual environment rather than fitting the fictional event to isolated headline gusts.
3. Fit mixing and roughness parameters by regime (stable wedge, marine layer, post-frontal mixed air, convective passage), with held-out station/event validation.
4. Evaluate bias, absolute error, inland decay by distance/time since passage, peak-time error and exceedance reliability. Treat censored outage maxima as lower bounds rather than ordinary missing-at-random samples.
5. Examine sensitivity of the extreme 939 mb scenario to uncertain low-level jet speeds and momentum transfer. It remains an extrapolation beyond these analog contexts.

NHC Michael and Idalia tropical cyclone report URLs were identified, but the report fetches were blocked during this session. No uninspected PDF station tables were treated as calibration data. The NWS event pages above were inspected directly.
