"""SIMULATED SCENARIO: synthetic reporting at real station coordinates.

Exposure is a fixed station bias. Turbulence is an AR(1) process with a 30 min
correlation timescale on the 6 min model sampling. Durations (3 s gust / 2 min
mean) name closure targets; no 3 s time-resolved flow is computed. Outages are
synthetically placed around hazardous peaks and persist, so reported maxima
can understate model maxima. This does not imply any real station is unreliable.
"""
import csv
import os
import numpy as np
import stormmodel as sm
from station_catalog import ST

EXPOSURE_SD = .045
TURBULENCE_SD = .018
CORRELATION_H = .5
OUTAGE_THRESHOLD_MPH = 90.
OUTAGE_CHANCE = .30


def station_reports():
    with np.load(os.path.join(sm.HERE,'station_series.npz')) as d:
        hours,gust,mean = d['hours'],d['gust_mph'],d['mean_mph']
    rng = np.random.default_rng(42026)
    exposures = np.clip(rng.normal(1.,EXPOSURE_SD,len(ST)),.85,1.08)
    residual = np.zeros_like(gust)
    residual[0] = rng.normal(0,TURBULENCE_SD,len(ST))
    for k in range(1,len(hours)):
        rho = np.exp(-(hours[k]-hours[k-1])/CORRELATION_H)
        residual[k] = rho*residual[k-1]+rng.normal(0,TURBULENCE_SD*np.sqrt(1-rho*rho),len(ST))
    synthetic_gust = gust*exposures[None,:]*(1+residual)
    synthetic_mean = mean*exposures[None,:]
    reports = []
    for j,(sid,name,lat,lon) in enumerate(ST):
        x,y = sm.m(lon,lat)
        if not (0<x<sm.m.xmax and 0<y<sm.m.ymax): continue
        pi = int(np.argmax(synthetic_gust[:,j])); mask = np.ones(len(hours),bool)
        outage = bool(synthetic_gust[pi,j]>=OUTAGE_THRESHOLD_MPH and rng.random()<OUTAGE_CHANCE)
        outage_start = outage_end = None
        if outage:
            # Outage begins before or at the peak, lasts hours; not missing-at-random.
            outage_start = float(hours[pi]-rng.uniform(.1,.8))
            outage_end = outage_start+float(rng.uniform(3.,12.))
            mask = (hours<outage_start)|(hours>outage_end)
        valid = np.flatnonzero(mask)
        rp = int(valid[np.argmax(synthetic_gust[valid,j])])
        reports.append(dict(id=sid,name=name,lat=lat,lon=lon,
          grid_gust_mph=round(float(gust[:,j].max()),2),grid_peak_h=round(float(hours[np.argmax(gust[:,j])]),2),
          synthetic_full_gust_mph=round(float(synthetic_gust[:,j].max()),2),
          reported_gust_mph=round(float(synthetic_gust[rp,j]),2),reported_peak_h=round(float(hours[rp]),2),
          reported_mean_mph=round(float(synthetic_mean[mask,j].max()),2),
          exposure_multiplier=round(float(exposures[j]),4),outage=outage,
          outage_start_h=outage_start,outage_end_h=outage_end,simulated=True))
    return sorted(reports,key=lambda r:-r['reported_gust_mph'])

if __name__=='__main__':
    reports = station_reports()
    with open(os.path.join(sm.HERE,'synthetic_station_reports.csv'),'w',newline='') as fh:
        w = csv.DictWriter(fh,fieldnames=list(reports[0])); w.writeheader(); w.writerows(reports)
    print('SIMULATED SCENARIO:',len(reports),'stations;',sum(r['outage'] for r in reports),'synthetic outages')
    print([(r['id'],r['reported_gust_mph'],sm.hlabel(r['reported_peak_h'])) for r in reports[:8]])
