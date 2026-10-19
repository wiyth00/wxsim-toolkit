"""SIMULATED SCENARIO: assumed-perturbation frequencies, NOT forecast probabilities.

python ensemble.py --members 20 --workers 4 --step .2
Control remains at six-minute resolution; ensemble uses twelve-minute samples.
All fields are integrated independently (track/intensity/timing/mixing/cell seeds),
not perturbed from the finished control swath. peak fields are retained per member.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
import numpy as np
os.environ['APALACHEE_SKIP_CACHE'] = '1'
import stormmodel as sm

ENSEMBLE_SEED = 20261215
PERTURB_SD = dict(track_km=45.,pressure_mb=4.,radius_fraction=.08,timing_h=1.2,
                 sting_km=25.,sting_timing_h=.8,sting_fraction=.20,inversion_fraction=.20)


def generate_members(n,seed=ENSEMBLE_SEED):
    rng = np.random.default_rng(seed)
    def draw(sd): return float(np.clip(rng.normal(),-2.5,2.5)*sd)
    members = []
    for i in range(n):
        members.append(sm.Member(track_dx_km=draw(PERTURB_SD['track_km']),
          track_dy_km=draw(PERTURB_SD['track_km']),pressure_delta_mb=draw(PERTURB_SD['pressure_mb']),
          radius_scale=1+draw(PERTURB_SD['radius_fraction']),timing_h=draw(PERTURB_SD['timing_h']),
          sting_dx_km=draw(PERTURB_SD['sting_km']),sting_dy_km=draw(PERTURB_SD['sting_km']),
          sting_timing_h=draw(PERTURB_SD['sting_timing_h']),sting_scale=1+draw(PERTURB_SD['sting_fraction']),
          inversion_scale=1+draw(PERTURB_SD['inversion_fraction']),seed=sm.SEED+100+i))
    return members


def worker(args):
    index,member,step = args
    peak,_,_,_ = sm.member_peak(member,step=step)
    return index,peak


def run(n=20,workers=4,step=.2):
    if n<2 or step<=0 or workers<1: raise ValueError('Need >=2 members, positive step and >=1 worker')
    members = generate_members(n)
    peaks = np.empty((n,)+sm.X.shape,np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker,(i,member,step)) for i,member in enumerate(members)]
        for k,job in enumerate(as_completed(jobs),1):
            i,peak = job.result(); peaks[i] = peak
            print(f'ensemble {k}/{n}; member {i+1} max {peak.max():.1f} mph',flush=True)
    summary = dict(member_peak=peaks,mean_peak=peaks.mean(axis=0),max_peak=peaks.max(axis=0),
        p58=100.*(peaks>=58).mean(axis=0),p74=100.*(peaks>=74).mean(axis=0),
        p10=np.quantile(peaks,.1,axis=0),p90=np.quantile(peaks,.9,axis=0))
    with open(__file__,'rb') as source:
        ensemble_digest = hashlib.sha256(source.read()).hexdigest()
    meta = dict(simulated=True,model_signature=sm.signature(),
      ensemble_signature=ensemble_digest,
      members_json=json.dumps([asdict(x) for x in members],sort_keys=True),
      perturbations_json=json.dumps(PERTURB_SD,sort_keys=True),step_h=step,seed=ENSEMBLE_SEED,n_members=n)
    np.savez_compressed(os.path.join(sm.HERE,'ensemble.npz'),**summary,**meta)
    print('saved ensemble.npz; uncalibrated synthetic scenario frequencies',flush=True)

if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--members',type=int,default=20); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--step',type=float,default=.2)
    a = p.parse_args(); run(a.members,a.workers,a.step)
