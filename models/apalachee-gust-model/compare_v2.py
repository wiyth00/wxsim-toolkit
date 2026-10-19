"""SIMULATED SCENARIO: reproduce v2/v3 comparisons and run provenance."""
from pathlib import Path
import csv
import json
import numpy as np
import stormmodel as s
from station_catalog import ST
from stations import station_reports

reports={r['id']:r for r in station_reports()}
with np.load(Path(s.HERE)/'reference_outputs'/'peak_v2.npz') as d:
    oldG,oldT=d['G'],d['T']
if oldG.shape!=s.G.shape: raise ValueError('v2 comparison requires the original model grid')
rng=np.random.default_rng(42)
rows=[]
for sid,name,lat,lon in ST:
    x,y=s.m(lon,lat)
    if not (0<x<s.m.xmax and 0<y<s.m.ymax): continue
    old_grid=s.sample(oldG,lon,lat)
    old_report=int(round(max(old_grid+rng.normal(0,2.4),20)))
    r=reports[sid];v3=s.sample(s.G,lon,lat)
    rows.append(dict(id=sid,name=name,simulated=True,v2_grid_mph=round(old_grid,2),
      v3_grid_mph=round(v3,2),grid_delta_mph=round(v3-old_grid,2),
      v2_synthetic_report_mph=old_report,v3_synthetic_report_mph=r['reported_gust_mph'],
      synthetic_report_delta_mph=round(r['reported_gust_mph']-old_report,2),
      v3_reported_peak_utc=s.hlabel(r['reported_peak_h']),synthetic_outage=r['outage']))
with open(Path(s.HERE)/'v2_v3_comparison.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with np.load(Path(s.HERE)/'station_series.npz') as d:
    loss=d['gust_mph'].max(axis=0)-d['gust_mph'][::2].max(axis=0)
with np.load(Path(s.HERE)/'ensemble.npz') as e:
    if str(e['model_signature'])!=s.signature():raise ValueError('Regenerate the ensemble first')
    meta={k:e[k].item() for k in ('step_h','n_members','seed','members_json','perturbations_json','model_signature','ensemble_signature')}
    meta['members']=json.loads(meta.pop('members_json'));meta['perturbations']=json.loads(meta.pop('perturbations_json'))
    stats=dict(simulated=True,version=s.MODEL_VERSION,control_grid_max_mph=round(float(s.G.max()),2),
      control_land_max_mph=round(float(s.G[s.land].max()),2),control_water_max_mph=round(float(s.G[~s.land].max()),2),
      v2_grid_max_mph=round(float(oldG.max()),2),v2_land_max_mph=round(float(oldG[s.land].max()),2),
      ensemble_member_max_min_mph=round(float(e['member_peak'].max(axis=(1,2)).min()),2),
      ensemble_member_max_max_mph=round(float(e['member_peak'].max(axis=(1,2)).max()),2),
      control_station_12min_loss_median_mph=round(float(np.median(loss)),3),
      control_station_12min_loss_max_mph=round(float(loss.max()),3),
      synthetic_outages=sum(r['outage'] for r in reports.values()),terrain=s.TERRAIN_MODE,
      reference_status='physics-informed; not quantitatively calibrated to analog observations')
Path(s.HERE,'run_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
Path(s.HERE,'ensemble_parameters.json').write_text(json.dumps(dict(simulated=True,**meta),indent=2)+'\n')
print('saved v2_v3_comparison.csv, run_summary.json and ensemble_parameters.json')
