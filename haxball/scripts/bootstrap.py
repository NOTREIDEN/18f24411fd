"""Short supervised warmstart; labels come from our transparent predictive baseline."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lab.arena import Arena
from lab.model import Policy,save_checkpoint,load_checkpoint
from lab.schema import action_indices
from lab.teacher import baseline_actions
from lab.maps import futsal_for


def run(seconds=180,output='runs/bootstrap.pt',seed=0,n_envs=64,initial=None):
    torch.set_num_threads(max(1,min(3,int(os.getenv('TORCH_NUM_THREADS','2')))))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=load_checkpoint(initial)[0] if initial else Policy();opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    start=time.monotonic();updates=0;samples=0;ema=None
    formats=[(1,futsal_for(1)),(2,futsal_for(2)),(3,futsal_for(3)),(5,futsal_for(5)),(10,futsal_for(10)),(12,futsal_for(12)),(3,'futsal-classic'),(1,'classic')]
    arenas={};log=[]
    while time.monotonic()-start<seconds or updates==0:
        size,stadium=formats[updates%len(formats)]
        key=(size,stadium)
        if key not in arenas:arenas[key]=Arena(n_envs,size,stadium,seed+updates,tick_skip=2,max_ticks=600)
        env=arenas[key]
        # Fresh states prevent a few stalled kickoff worlds from dominating BC.
        env.randomize_all()
        h=model.initial(n_envs);loss=0.;correct=0;count=0
        for _ in range(8):
            obs=env.obs();actions=baseline_actions(env.state,env.spec,previous_actions=env.previous)
            ids=action_indices(actions,env.state.team)[np.arange(n_envs),env.egos]
            x=[torch.from_numpy(a) for a in obs]
            logits,_,h=model(*x,h)
            labels=torch.from_numpy(ids)
            weight=torch.where(labels>=9,2.0,1.0)
            weight=weight*torch.where(torch.from_numpy(obs[0][:,17])<.01,1.5,.8)
            per=torch.nn.functional.cross_entropy(logits,labels,reduction='none',label_smoothing=.01)
            loss=loss+(per*weight).mean()/8
            correct+=int((logits.argmax(-1)==labels).sum());count+=n_envs
            # DAgger: request teacher labels on states visited by the learner.
            learner=logits.detach().argmax(-1).numpy()
            mixture=min(.8,max(0,(time.monotonic()-start)/max(seconds,1)-.15))
            behavior=np.where(rng.random(n_envs)<mixture,learner,ids)
            _,_,term,trunc,_=env.step(behavior,actions_override=actions)
            h=h*torch.from_numpy(~(term|trunc))[:,None]
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        updates+=1;samples+=count;value=float(loss.detach());ema=value if ema is None else .95*ema+.05*value
        if updates%20==0:
            record={'phase':'imitation','seconds':time.monotonic()-start,'updates':updates,'samples':samples,'loss_ema':ema,'accuracy':correct/count,'team_size':size}
            print(json.dumps(record),flush=True);log.append(record)
            save_checkpoint(output,model,opt,phase='imitation',seed=seed,updates=updates,samples=samples,elapsed=time.monotonic()-start)
    save_checkpoint(output,model,opt,phase='imitation',seed=seed,updates=updates,samples=samples,elapsed=time.monotonic()-start)
    Path(output).with_suffix('.json').write_text(json.dumps(log,indent=2))
    return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=180);p.add_argument('--output',default='runs/bootstrap.pt');p.add_argument('--initial');p.add_argument('--seed',type=int,default=0);p.add_argument('--envs',type=int,default=64);a=p.parse_args()
    run(a.seconds,a.output,a.seed,a.envs,a.initial)
