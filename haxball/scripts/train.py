"""CPU recurrent PPO with a snapshot league and independent checkpoint gates.

Each runner is an autonomous island: no stale/off-policy trajectories are merged,
and checkpoint parameters are never averaged. All buffers are strictly on-policy.
"""
from __future__ import annotations
import argparse,json,os,random,sys,time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lab.arena import Arena
from lab.schema import encode,decode_actions,action_indices
from lab.teacher import baseline_actions
from lab.model import Policy,load_checkpoint,save_checkpoint
from lab.guard import ProgressGuard
from lab.evaluate import evaluate
from lab.maps import futsal_for


def tensor_obs(obs):return tuple(torch.from_numpy(x) for x in obs)


def collect(env,model,horizon,hidden,rng,shadow=None,style='balanced',style_weight=1,full_opponents=False):
    n=env.n
    storage={k:[] for k in ['ego','entities','mask','hidden','actions','logprob','values','rewards','done','reset','teacher']}
    h=hidden if hidden is not None else model.initial(n)
    shadow_h=None
    reset=np.zeros(n,dtype=bool)
    obs=env.obs();rewards=[];style_total=0.;goals_for=0;goals_against=0
    start=time.monotonic()
    for t in range(horizon):
        storage['ego'].append(obs[0]);storage['entities'].append(obs[1]);storage['mask'].append(obs[2])
        storage['hidden'].append(h.numpy().copy());storage['reset'].append(reset.copy())
        a,lp,v,next_h=model.act(obs,h)
        actions=baseline_actions(env.state,env.spec,style,env.previous)
        rows=np.arange(n)
        teacher=action_indices(actions,env.state.team)[rows,env.egos]
        storage['teacher'].append(teacher)
        if shadow is not None:
            myteam=env.state.team[rows,env.egos]
            opposite=np.where(myteam==2,env.team_size,0)
            if full_opponents:
                opposite=opposite[:,None]+np.arange(env.team_size)[None]
            # Historical policies sometimes control the WHOLE opposing team.
            opponent_obs=encode(env.state,env.spec,opposite,env.previous,shadow.observation_revision)
            oa,_,_,shadow_h=shadow.act(opponent_obs,shadow_h)
            if full_opponents:
                rr=np.broadcast_to(rows[:,None],opposite.shape)
                actions[rr,opposite]=decode_actions(oa.numpy().reshape(opposite.shape),env.state.team[rr,opposite])
            else:actions[rows,opposite]=decode_actions(oa.numpy(),env.state.team[rows,opposite])
        next_obs,r,terminated,truncated,info=env.step(a.numpy(),style,style_weight,actions_override=actions)
        if np.any(np.abs(info['style_reward'][info['goal_sign']==0])>1e-9):
            raise RuntimeError('style reward paid without an actual goal')
        if np.any(np.abs(info['style_reward']) > np.abs(info['goal_sign'])+1e-6):
            raise RuntimeError('style reward exceeded the per-goal cap')
        done=terminated|truncated
        if truncated.any():
            # Bootstrap the OLD player's terminal observation before auto-reset.
            with torch.no_grad():
                _,tv,_=model(*tensor_obs(info['terminal_obs']),next_h)
            r=r.copy();r[truncated]+=env.gamma*tv.numpy()[truncated]
        storage['actions'].append(a.numpy());storage['logprob'].append(lp.numpy());storage['values'].append(v.numpy())
        storage['rewards'].append(r);storage['done'].append(done.copy())
        rewards.append(float(r.mean()));style_total+=float(np.abs(info['style_reward']).sum())
        goals_for+=int((info['goal_sign']>0).sum());goals_against+=int((info['goal_sign']<0).sum())
        # Explicit game reset notifications clear temporal memory, including live play.
        reset=done|(info['goal_sign']!=0)
        next_h[reset]=0
        if shadow_h is not None:
            shadow_h[np.repeat(reset,env.team_size) if full_opponents else reset]=0
        h=next_h;obs=next_obs
    with torch.no_grad():_,last_value,_=model(*tensor_obs(obs),h)
    out={k:np.stack(v) for k,v in storage.items()}
    advantage=np.zeros_like(out['rewards']);last_gae=np.zeros(n,dtype=np.float32)
    value_next=last_value.numpy()
    for t in range(horizon-1,-1,-1):
        live=1-out['done'][t].astype(np.float32)
        delta=out['rewards'][t]+env.gamma*value_next*live-out['values'][t]
        last_gae=delta+env.gamma*.97*live*last_gae
        advantage[t]=last_gae;value_next=out['values'][t]
    out['returns']=advantage+out['values']
    out['advantage']=(advantage-advantage.mean())/max(float(advantage.std()),1e-6)
    return out,h,{'rollout_seconds':time.monotonic()-start,'mean_reward':float(np.mean(rewards)),
                  'goals_for':goals_for,'goals_against':goals_against,'style_paid':style_total,
                  'transitions':n*horizon,'physics_ticks':n*horizon*env.tick_skip}


def update(model,opt,data,rng,epochs=2,sequence=16,sequences_per_batch=32,entropy_coef=.012,anchor_coef=.025):
    horizon,n=data['actions'].shape
    if horizon%sequence:raise ValueError('rollout must be divisible by recurrent sequence length')
    chunks=[(t,i) for t in range(0,horizon,sequence) for i in range(n)]
    loss_values=[];kls=[];entropies=[]
    stop=False
    for epoch in range(epochs):
        rng.shuffle(chunks)
        for offset in range(0,len(chunks),sequences_per_batch):
            batch=chunks[offset:offset+sequences_per_batch]
            starts=np.asarray([x[0] for x in batch]);envs=np.asarray([x[1] for x in batch])
            h=torch.from_numpy(data['hidden'][starts,envs].copy())
            total=0.;kl_sum=0.;entropy_sum=0.
            for t in range(sequence):
                idx=starts+t
                clear=torch.from_numpy(data['reset'][idx,envs])
                h=h*(~clear)[:,None]
                x=(torch.from_numpy(data['ego'][idx,envs]),torch.from_numpy(data['entities'][idx,envs]),torch.from_numpy(data['mask'][idx,envs]))
                logits,value,h=model(*x,h)
                dist=torch.distributions.Categorical(logits=logits)
                actions=torch.from_numpy(data['actions'][idx,envs]);oldlp=torch.from_numpy(data['logprob'][idx,envs])
                lp=dist.log_prob(actions);logratio=(lp-oldlp).clamp(-20,20);ratio=logratio.exp()
                adv=torch.from_numpy(data['advantage'][idx,envs])
                policy=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
                target=torch.from_numpy(data['returns'][idx,envs])
                value_loss=torch.nn.functional.smooth_l1_loss(value,target)
                entropy=dist.entropy().mean()
                labels=torch.from_numpy(data['teacher'][idx,envs])
                anchor=torch.nn.functional.cross_entropy(logits,labels)
                total=total+(policy+.5*value_loss-entropy_coef*entropy+anchor_coef*anchor)/sequence
                kl_sum+=float(((ratio-1)-logratio).mean().detach())/sequence
                entropy_sum+=float(entropy.detach())/sequence
            opt.zero_grad(set_to_none=True);total.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite PPO gradient')
            opt.step();loss_values.append(float(total.detach()));kls.append(kl_sum);entropies.append(entropy_sum)
            if kl_sum>.03:stop=True;break
        if stop:break
    return {'loss':float(np.mean(loss_values)),'approx_kl':float(np.mean(kls)),
            'entropy':float(np.mean(entropies)),'early_kl_stop':stop,'optimizer_batches':len(loss_values)}


def train(args):
    threads=max(1,min(os.cpu_count() or 1,args.threads));torch.set_num_threads(threads)
    torch.manual_seed(args.seed);random.seed(args.seed);rng=np.random.default_rng(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);league=out/'league';league.mkdir(exist_ok=True)
    if args.initial:model,initial=load_checkpoint(args.initial)
    else:model,initial=Policy(),{}
    model.observation_revision='player-geometry-v2'
    model.train();opt=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    guard=ProgressGuard(patience=4)
    start=time.monotonic();last_eval=start;last_save=start;last_video=start
    samples=int(initial.get('samples',0)) if args.resume else 0
    iteration=int(initial.get('iteration',0)) if args.resume else 0
    if args.resume and initial.get('phase')=='ppo' and 'optimizer' in initial:
        opt.load_state_dict(initial['optimizer'])
        for group in opt.param_groups:group['lr']=args.lr
    if args.resume and 'numpy_rng_state' in initial:rng.bit_generator.state=initial['numpy_rng_state']
    if args.resume and 'torch_rng_state' in initial:torch.set_rng_state(initial['torch_rng_state'])
    style_weight=1.;entropy=.012;format_weights=np.array([.3,.25,.15,.15,.15])
    formats=[(1,futsal_for(1)),(3,futsal_for(3)),(5,futsal_for(5)),(10,futsal_for(10)),(12,futsal_for(12))]
    arenas={};memories={};shadow=None;last_report=None
    log_path=out/'training.jsonl';log=log_path.open('a',buffering=1)
    def emit(record):
        record={'elapsed':time.monotonic()-start,'iteration':iteration,'samples':samples,**record}
        line=json.dumps(record);log.write(line+'\n');print(line,flush=True)
    def checkpoint(file,**metadata):
        save_checkpoint(file,model,opt,phase='ppo',samples=samples,seed=args.seed,iteration=iteration,
                        numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),**metadata)
    # Atomic resume on a requested stop; GitHub jobs do not lose an entire segment.
    import signal
    stopping=False
    def stop_handler(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    # Honest baseline evaluation before the first policy update.
    first=evaluate(model,seed=90210,n_envs=args.eval_envs,seconds=args.eval_seconds,formats=(1,3,10,12),paired=False)
    if (out/'champion.pt').exists():
        _,old=load_checkpoint(out/'champion.pt')
        if old.get('evaluation'):guard.champion=old['evaluation']
    first['training_reward']=0.;decision=guard.observe(first)
    if decision['promote'] or not (out/'champion.pt').exists():checkpoint(out/'champion.pt',evaluation=first)
    (out/'evaluation-initial.json').write_text(json.dumps(first,indent=2));emit({'event':'evaluation','report':first,'guard':decision})
    while time.monotonic()-start<args.seconds and not stopping:
        fi=int(rng.choice(len(formats),p=format_weights/format_weights.sum()))
        size,stadium=formats[fi]
        if time.monotonic()-start>1800 and rng.random()<.35:
            stadium=['classic','big','rounded','huge','huge'][fi]
        # At least half of batches preserve goals/kickoffs as a real continuous match.
        continuous=iteration%2==0
        key=(size,stadium,continuous)
        if key not in arenas:
            arenas[key]=Arena(args.envs,size,stadium,args.seed+fi*101+int(continuous),tick_skip=args.tick_skip,
                              max_ticks=2700 if continuous else 1200,continuous=continuous)
        env=arenas[key]
        # Fresh arbitrary states for drill fragments; full matches retain their
        # real conceding-team kickoffs. Identical kickoff farming is not a drill.
        if not continuous:
            env.randomize_all();memories.pop(key,None)
        style=['balanced','chase','defensive','passer','shooter'][iteration%5]
        snapshots=sorted(league.glob('*.pt'))+sorted((out/'peers').glob('*.pt'))
        if snapshots and (iteration%4==0 or args.role=='exploiter'):
            path=snapshots[int(rng.integers(len(snapshots)))];shadow,_=load_checkpoint(path)
        elif iteration%4==1:shadow=None
        data,h,rollout=collect(env,model,args.horizon,memories.get(key),rng,shadow,style,style_weight,full_opponents=bool(shadow and iteration%4==0))
        memories[key]=h
        stats=update(model,opt,data,rng,epochs=args.epochs,sequence=args.sequence,entropy_coef=entropy,
                     anchor_coef=max(.005,args.anchor_coef*(1-min((time.monotonic()-start)/7200,1))))
        samples+=rollout['transitions'];iteration+=1
        if iteration%5==0:
            emit({'event':'update','format':size,'continuous':continuous,'opponent':style,'rollout':rollout,'ppo':stats,'metrics':env.tracker.summary()})
        now=time.monotonic()
        if now-last_save>=60 or iteration==1:
            checkpoint(out/'latest.pt')
            emit({'event':'checkpoint','path':str(out/'latest.pt'),'champion':str(out/'champion.pt')})
            last_save=now
        if iteration%40==0:
            save_checkpoint(league/f'snapshot-{iteration:06d}.pt',model,phase='league',samples=samples,iteration=iteration)
            old=sorted(league.glob('*.pt'))
            for path in old[:-12]:path.unlink()
        if now-last_eval>=args.eval_interval:
            report=evaluate(model,seed=90210+(iteration%4)*1000,n_envs=args.eval_envs,seconds=args.eval_seconds,paired=False)
            report['training_reward']=rollout['mean_reward'];decision=guard.observe(report);last_report=report
            emit({'event':'evaluation','report':report,'guard':decision})
            (out/f'evaluation-{iteration:06d}.json').write_text(json.dumps(report,indent=2))
            if decision['promote']:
                checkpoint(out/'champion.pt',evaluation=report)
            if decision['intervention']=='rollback-and-reduce-style':
                champion,_=load_checkpoint(out/'champion.pt');model.load_state_dict(champion.state_dict())
                opt=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5);style_weight*=.5;memories.clear()
                emit({'event':'automatic_intervention','action':'rollback-and-reduce-style','style_weight':style_weight})
            if decision['intervention'] in ('refresh-opponents-and-curriculum','rebalance-regressed-formats'):
                for j,(count,_) in enumerate(formats):
                    score=report['by_format'].get(str(count),{}).get('score',report['score'])
                    format_weights[j]=.1+max(0,1-score)
                entropy=min(.035,entropy*1.25)
                for arena in arenas.values():arena.randomize_all()
                memories.clear()
                emit({'event':'automatic_intervention','action':decision['intervention'],'format_weights':format_weights.tolist(),'entropy':entropy})
            last_eval=time.monotonic()
        if now-last_video>=args.video_interval:
            # A bounded child evaluates the ACTUAL saved checkpoint and emits its clip.
            import subprocess
            video_checkpoint=out/'champion.pt'
            dest=out/f'progress-{iteration:06d}.mp4'
            try:
                subprocess.run([sys.executable,str(Path(__file__).with_name('render_progress.py')),'--checkpoint',str(video_checkpoint),'--output',str(dest),'--seconds','12'],check=True,timeout=180)
                emit({'event':'video','path':str(dest),'checkpoint':str(video_checkpoint)})
            except (subprocess.SubprocessError,OSError) as error:
                emit({'event':'video_warning','error':str(error),'training_continues':True})
            last_video=time.monotonic()
    checkpoint(out/'latest.pt')
    emit({'event':'complete','last_evaluation':last_report,'champion':str(out/'champion.pt')})
    log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=10800);p.add_argument('--initial');p.add_argument('--output',default='runs/main');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--envs',type=int,default=64);p.add_argument('--horizon',type=int,default=128);p.add_argument('--sequence',type=int,default=16);p.add_argument('--epochs',type=int,default=2);p.add_argument('--threads',type=int,default=2);p.add_argument('--tick-skip',type=int,default=2);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--eval-interval',type=float,default=180);p.add_argument('--eval-envs',type=int,default=8);p.add_argument('--eval-seconds',type=int,default=20)
    p.add_argument('--video-interval',type=float,default=1200);p.add_argument('--resume',action='store_true');p.add_argument('--anchor-coef',type=float,default=.04);p.add_argument('--role',choices=['main','exploiter'],default='main')
    train(p.parse_args())
