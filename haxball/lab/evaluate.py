"""Frozen, score-based paired evaluation with individual-impact controls."""
from __future__ import annotations
import time
from dataclasses import replace
import numpy as np
from .arena import Arena
from .schema import action_indices
from .teacher import baseline_actions
from .guard import wilson_interval
from .maps import futsal_for


def _leg(model,size,stadium,seed,n_envs,ticks,style,random_start):
    env=Arena(n_envs,size,stadium,seed,tick_skip=2,max_ticks=ticks+2,random_start=random_start,continuous=True)
    fixed=np.where(np.arange(n_envs)%2==0,0,size)
    env.egos[:]=fixed
    gf=np.zeros(n_envs);ga=np.zeros(n_envs);h=None
    contacts=0;learner_contacts=0;distance_sum=0.;speed_sum=0.
    start=time.monotonic()
    rng=np.random.default_rng(seed)
    styles=np.asarray(['balanced','chase','defensive','passer','shooter'])
    style_ids=rng.permutation(np.arange(n_envs))%len(styles)
    delays=rng.integers(0,3,size=n_envs)
    history=[np.zeros((n_envs,size*2,3),dtype=np.int64) for _ in range(3)]
    for _ in range(ticks//2):
        env.egos[:]=fixed
        # Distinct opponent styles and delays make trials genuinely different;
        # cloning a deterministic kickoff N times does not create N independent games.
        reference=np.empty((n_envs,size*2,3),dtype=np.int64)
        for sid in np.unique(style_ids):
            rows=np.flatnonzero(style_ids==sid);st=env.state
            sub=replace(st,ball_pos=st.ball_pos[rows],ball_vel=st.ball_vel[rows],player_pos=st.player_pos[rows],player_vel=st.player_vel[rows],team=st.team[rows],scored=st.scored[rows],steps=st.steps[rows],ball_pred=None if st.ball_pred is None else st.ball_pred[rows])
            reference[rows]=baseline_actions(sub,env.spec,str(styles[sid]),env.previous[rows])
        if model is None:
            own=baseline_actions(env.state,env.spec,'balanced',env.previous)
            ids=action_indices(own,env.state.team)[np.arange(n_envs),fixed]
        else:
            a,_,_,h=model.act(env.obs(model.observation_revision),h,deterministic=True);ids=a.cpu().numpy()
        from .schema import decode_actions
        reference[np.arange(n_envs),fixed]=decode_actions(ids,env.state.team[np.arange(n_envs),fixed])
        history=[reference,history[0],history[1]]
        applied=np.stack(history)[delays,np.arange(n_envs)]
        delayed_ids=action_indices(applied,env.state.team)[np.arange(n_envs),fixed]
        _,_,term,trunc,info=env.step(delayed_ids,opponent_style=style,style_weight=0,actions_override=applied)
        sign=info['goal_sign'];gf+=sign>0;ga+=sign<0
        contacts+=int(info['events']['contact'].sum())
        learner_contacts+=int((info['events']['contact']&(info['events']['who']==fixed)).sum())
        st=info['terminal_state'];pp=st.player_pos[np.arange(n_envs),fixed]
        distance_sum+=float(np.linalg.norm(pp-st.ball_pos,axis=-1).sum())
        speed_sum+=float(np.linalg.norm(st.player_vel[np.arange(n_envs),fixed],axis=-1).sum())
        if h is not None:
            # Mirrors the live adapter's memory clear after a goal/reset notification.
            done=(info['goal_sign']!=0)|trunc
            if done.any():h[done]=0
    score=(gf>ga).astype(float)+.5*(gf==ga)
    summary=env.tracker.summary()
    return {'games':n_envs,'score':float(score.mean()),'wins':int((gf>ga).sum()),'draws':int((gf==ga).sum()),
            'goals_for':int(gf.sum()),'goals_against':int(ga.sum()),'goal_difference':float((gf-ga).mean()),
            'cluster_fraction':summary['cluster_fraction'],'loop_fraction':summary['loop_fraction'],
            'passes_completed':summary['passes_completed'],'learner_contact_share':learner_contacts/max(contacts,1),
            'mean_ball_distance':distance_sum/max(n_envs*(ticks//2),1),
            'mean_speed':speed_sum/max(n_envs*(ticks//2),1),'wall_seconds':time.monotonic()-start,
            'independent_conditions':len(set(zip(style_ids.tolist(),delays.tolist())))}


def evaluate(model,seed=90210,n_envs=8,seconds=20,formats=(1,3,10,12),paired=True):
    start=time.monotonic();out={};all_scores=[];goal_differences=[];clusters=[];loops=[]
    for size in formats:
        # Rounded geometry and kickoff starts are outside the BC sampling blend.
        stadium=futsal_for(size)
        candidate=_leg(model,size,stadium,seed+size,n_envs,int(seconds*60),'balanced',False)
        if paired:
            baseline=_leg(None,size,stadium,seed+size,n_envs,int(seconds*60),'balanced',False)
            candidate['baseline']=baseline
            candidate['marginal_goal_difference']=candidate['goal_difference']-baseline['goal_difference']
        candidate['score_interval']=wilson_interval(candidate['wins']+.5*candidate['draws'],n_envs)
        out[str(size)]=candidate
        all_scores.append(candidate['score']);goal_differences.append(candidate['goal_difference'])
        clusters.append(candidate['cluster_fraction']);loops.append(candidate['loop_fraction'])
    return {'score':float(np.mean(all_scores)),'goal_difference':float(np.mean(goal_differences)),
            'cluster_fraction':float(np.mean(clusters)),'loop_fraction':float(np.mean(loops)),
            'by_format':out,'seed':seed,'evaluation_seconds':seconds,'wall_seconds':time.monotonic()-start,
            'professional_benchmark':False}
