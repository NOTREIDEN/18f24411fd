"""Bounded, reproducible end-to-end physics capacity probe."""
import argparse, importlib.metadata, json, os, platform, time
from pathlib import Path
import numpy as np
from haxballgym import TransitionEngine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='reports/probe.json'); p.add_argument('--seed',type=int,default=0); args=p.parse_args()
    rng=np.random.default_rng(args.seed)
    report={'seed':args.seed,'cpu_count':os.cpu_count(),'platform':platform.platform(),'rayon_threads':os.getenv('RAYON_NUM_THREADS'),'versions':{n:importlib.metadata.version(n) for n in ['haxball-core','haxballgym','numpy']},'benchmarks':[]}
    for team_size in [1,3,10,12]:
        n=256
        engine=TransitionEngine(n,team_size,team_size,tick_skip=2,stadium='classic' if team_size<4 else 'huge')
        state=engine.reset()
        actions=rng.integers(-1,2,(n,team_size*2,3),dtype=np.int64); actions[...,2]=rng.integers(0,2,(n,team_size*2))
        engine.step(actions)
        start=time.perf_counter(); steps=0
        while time.perf_counter()-start<3:
            state=engine.step(actions); steps+=1
        elapsed=time.perf_counter()-start
        assert np.isfinite(state.player_pos).all() and np.isfinite(state.ball_pos).all()
        report['benchmarks'].append({'team_size':team_size,'envs':n,'seconds':elapsed,'physics_ticks_per_second':steps*n*2/elapsed,'decisions_per_second':steps*n/elapsed})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
