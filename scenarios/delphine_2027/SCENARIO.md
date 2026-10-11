# Hurricane Delphine (AL07 2027) — simulated

A fast-moving, Helene-style major hurricane. Delphine lands in the Florida Big Bend early Sat Sep 25 2027 as a
130-kt Category 4 and races north at 25+ mph, so it is still a hurricane when its center passes over metro
Atlanta Saturday afternoon. Inland Hurricane Warnings cover the Atlanta metro area for the first time in the
scenario's history. Post-tropical over the southern Appalachians by Sunday, then a remnant low over the Ohio Valley.

- `tc_delphine_adv18.json` — Advisory 18, 5 PM EDT Fri Sep 24 2027 (2100 UTC)

```bash
python -m wxsim.tropical.cone     scenarios/delphine_2027/tc_delphine_adv18.json   # 5-day cone graphic
python -m wxsim.tropical.products scenarios/delphine_2027/tc_delphine_adv18.json   # TCMAT2 + TCPAT2
python render_all.py scenarios/delphine_2027                                       # both
```

Inland intensity follows a Kaplan-DeMaria-style decay (130 kt at landfall -> ~65 kt nine hours inland) —
plausible only because of the fast forward speed. Cone radii in the JSON are illustrative, not NHC's table.
