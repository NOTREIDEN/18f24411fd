"""Local JSON-lines inference worker for an ordinary real-room client.

Only the normal client's observed state enters the actor. The native engine is
used to calculate the SAME free-flight observation as in training, not to move
the real player's disc. Real output is exclusively legal direction/kick input.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import haxball_core
from haxballgym.state import GameState
from lab.schema import FieldSpec,encode,decode_actions,SCHEMA_VERSION
from lab.model import load_checkpoint
from lab.teacher import baseline_actions
from lab.physics import compatible_hbs


class Worker:
    def __init__(self,checkpoint=None):
        torch.set_num_threads(1)
        self.model,self.meta=load_checkpoint(checkpoint) if checkpoint else (None,{})
        self.raw=None;self.core=None;self.spec=None;self.hidden=None;self.ids=None;self.previous=None

    def handle(self,msg):
        if msg['type']=='init':
            self.raw=msg['stadium'];self.core=None;self.hidden=None
            return {'type':'ready','schema':SCHEMA_VERSION,'phase':self.meta.get('phase','predictive-baseline')}
        if msg['type']=='reset':
            self.hidden=None
            if self.previous is not None:self.previous[:]=0
            return {'type':'reset'}
        if msg['type']!='state':raise ValueError('unknown bridge message')
        start=time.perf_counter()
        players=sorted(msg['players'],key=lambda p:(p['team'],p['id']))
        ids=[p['id'] for p in players];teams=np.array([[2 if p['team']==1 else 4 for p in players]],dtype=np.int64)
        nred=int((teams==2).sum());nblue=int((teams==4).sum())
        if not nred or not nblue:return {'type':'action','seq':msg['seq'],'dx':0,'dy':0,'kick':False,'reason':'incomplete-teams'}
        if msg['playerId'] not in ids:return {'type':'action','seq':msg['seq'],'dx':0,'dy':0,'kick':False,'reason':'spectator'}
        if self.core is None or ids!=self.ids:
            self.core=haxball_core.VecEnv.from_hbs(compatible_hbs(self.raw),1,nred,nblue,tick_skip=1)
            self.ids=ids;self.previous=np.zeros((1,len(ids),3),dtype=np.int64);self.hidden=None;self.spec=None
        pp=np.array([[p['pos'] for p in players]],dtype=np.float64)
        pv=np.array([[p['speed'] for p in players]],dtype=np.float64)
        bp=np.array([msg['ball']['pos']],dtype=np.float64);bv=np.array([msg['ball']['speed']],dtype=np.float64)
        if not all(np.isfinite(x).all() for x in (pp,pv,bp,bv)):raise ValueError('nonfinite live state')
        self.core.set_state(bp,bv,pp,pv,None)
        g0,g1,gt=self.core.goals()
        state=GameState(bp,bv,pp,pv,teams,np.array([-1],dtype=np.int8),np.array([msg.get('frame',0)],dtype=np.int64),g0,g1,gt,
                        self.core.wall_segments(),self.core.obstacle_discs(),self.core.predict_ball([2,6,18]),0.)
        if self.spec is None:self.spec=FieldSpec.from_raw(state,self.raw)
        ego=ids.index(msg['playerId'])
        if self.model:
            obs=encode(state,self.spec,np.array([ego]),self.previous,self.model.observation_revision)
            a,_,_,self.hidden=self.model.act(obs,self.hidden,deterministic=True)
            action=decode_actions(a.numpy(),teams[:,ego])[0]
        else:action=baseline_actions(state,self.spec,previous_actions=self.previous)[0,ego]
        self.previous[0,ego]=action
        return {'type':'action','seq':msg['seq'],'dx':int(action[0]),'dy':int(action[1]),'kick':bool(action[2]),'compute_ms':(time.perf_counter()-start)*1000}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');args=p.parse_args();worker=Worker(args.checkpoint)
    print(json.dumps({'type':'loaded','schema':SCHEMA_VERSION}),flush=True)
    for line in sys.stdin:
        try:result=worker.handle(json.loads(line))
        except Exception as exc:result={'type':'error','message':str(exc),'dx':0,'dy':0,'kick':False}
        print(json.dumps(result,separators=(',',':')),flush=True)
