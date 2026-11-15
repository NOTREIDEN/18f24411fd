"""Conservative possession/style attribution. Ambiguous contacts earn no bonus.

Style bonuses are payable ONLY on a real goal and have a per-goal cap. Touch,
wall contact, running and proximity never independently earn positive reward.
"""
from __future__ import annotations
import numpy as np


class EventTracker:
    def __init__(self, n, p, spec):
        self.n,self.p,self.spec=n,p,spec
        self.last_touch=np.full(n,-1,dtype=np.int64)
        self.last_team=np.full(n,-1,dtype=np.int64)
        self.touch_ball=np.zeros((n,2))
        self.touch_age=np.zeros(n,dtype=np.int64)
        self.since_wall=np.zeros(n,dtype=bool)
        self.dribbles=np.zeros((n,p),dtype=np.int64)
        self.selfpasses=np.zeros((n,p),dtype=np.int64)
        self.passes=np.zeros((n,p),dtype=np.int64)
        self.total={'passes_completed':0,'verified_dribbles':0,'verified_selfpasses':0,'goals':0,'wall_goals':0}
        self.episode_contact=np.zeros((n,p),dtype=bool)
        self.opponent_relative=np.zeros((n,p,p))
        self.beaten=np.zeros((n,p,p),dtype=bool)
        self.separation_sum=0.;self.cluster_samples=0;self.samples=0;self.loop_samples=0
        self.travel=np.zeros(n);self.displacement=np.zeros((n,2));self.age=0

    def reset_mask(self, mask):
        for arr in [self.last_touch,self.last_team]: arr[mask]=-1
        for arr in [self.touch_ball,self.touch_age,self.since_wall,self.dribbles,self.selfpasses,self.passes,self.episode_contact,self.opponent_relative,self.beaten,self.travel,self.displacement]:arr[mask]=0

    def update(self, prev, state, actions, ticks):
        n,p=self.n,self.p;row=np.arange(n)
        # Damping-only expected velocity; a wall can also change velocity, so require
        # one unique nearby player and keep boundary events out of contact credit.
        d=np.linalg.norm(state.player_pos-state.ball_pos[:,None],axis=-1)
        oldd=np.linalg.norm(prev.player_pos-prev.ball_pos[:,None],axis=-1)
        near=np.minimum(d,oldd) < self.spec.player_radius+self.spec.ball_radius+5
        change=np.linalg.norm(state.ball_vel-prev.ball_vel*self.spec.ball_damping**ticks,axis=-1)
        near_wall=(np.abs(state.ball_pos)>self.spec.half-self.spec.ball_radius-8).any(-1)
        wall=near_wall & (change>.15)
        self.since_wall |= wall
        unique=near.sum(-1)==1
        if prev.ball_pred is not None:
            residual=np.linalg.norm(state.ball_pos-prev.ball_pred[:,0],axis=-1)
            contact=unique & (residual>.03)
        else:contact=unique & (change>.12) & ~near_wall
        who=near.argmax(-1)
        team=state.team[row,who]
        changed=contact & (who!=self.last_touch)
        same_team=changed & (team==self.last_team) & (self.last_touch>=0)
        passed=same_team & (np.linalg.norm(state.ball_pos-self.touch_ball,axis=-1)>50) & (self.touch_age<180)
        if passed.any():
            self.passes[row[passed],self.last_touch[passed]]+=1
            self.total['passes_completed']+=int(passed.sum())
        selfpass=contact & (who==self.last_touch) & self.since_wall & (self.touch_age>=8) & (np.linalg.norm(state.ball_pos-self.touch_ball,axis=-1)>45)
        if selfpass.any():
            self.selfpasses[row[selfpass],who[selfpass]]+=1;self.total['verified_selfpasses']+=int(selfpass.sum())
        # Passing a nearby defender while retaining contact is a conservative dribble.
        for k in range(p):
            actor=contact & (who==k)
            if not actor.any(): continue
            sign=np.where(state.team[:,k]==2,1.,-1.)
            rel=(state.player_pos[:,:,0]-state.player_pos[:,k,None,0])*sign[:,None]
            opp=state.team!=state.team[:,k,None]
            close=np.linalg.norm(state.player_pos-state.player_pos[:,k,None],axis=-1)<100
            crossed=(self.opponent_relative[:,k]>15)&(rel < -15)&opp&close&self.episode_contact[:,k,None]&~self.beaten[:,k]
            # One count per defender per possession; after crossing the stored sign changes.
            count=crossed.sum(-1)*actor
            self.dribbles[:,k]+=count
            self.total['verified_dribbles']+=int(count.sum())
            self.opponent_relative[actor,k]=rel[actor]
            self.beaten[actor,k] |= crossed[actor]
            self.episode_contact[actor,k]=True
        # On a possession change discard old style credit: it cannot be farmed earlier.
        turnover=changed & (team!=self.last_team)
        self.dribbles[turnover]=0;self.selfpasses[turnover]=0;self.passes[turnover]=0
        self.beaten[turnover]=False
        self.touch_age+=ticks
        # Refresh origin only on a new player or a recovered self-pass; holding contact
        # must not erase the distance needed to verify a self-pass.
        refresh=changed|selfpass
        self.touch_ball[refresh]=state.ball_pos[refresh];self.touch_age[refresh]=0;self.since_wall[refresh]=False
        self.last_touch[contact]=who[contact];self.last_team[contact]=team[contact]
        goal=state.scored>=0
        self.total['goals']+=int(goal.sum());self.total['wall_goals']+=int((goal&self.since_wall).sum())
        diff=state.player_pos[:,:,None]-state.player_pos[:,None]
        ds=np.linalg.norm(diff,axis=-1)
        friendly=state.team[:,:,None]==state.team[:,None]
        cluster=(ds<42)&friendly&~np.eye(p,dtype=bool)[None]
        # Contact contests excluded: nearby teammates may legitimately intervene.
        clustered=cluster.any(-1)&(d>80)
        self.cluster_samples+=int(clustered.sum());self.samples+=n*p
        movement=state.ball_pos-prev.ball_pos
        self.travel+=np.linalg.norm(movement,axis=-1);self.displacement+=movement;self.age+=ticks
        if self.age>=180:
            loops=(self.travel>100)&(np.linalg.norm(self.displacement,axis=-1)<25)
            self.loop_samples+=int(loops.sum());self.travel[:]=0;self.displacement[:]=0;self.age=0
        return {'contact':contact,'who':who,'wall':wall,'goal':goal,'clustered':clustered}

    def style_on_goal(self, state):
        # A standard goal always pays its base reward. Creative finishes receive
        # extra preference; straight wall rebound shots receive zero style credit.
        winner=np.where(state.scored==2,4,2)
        own=state.team==winner[:,None]
        creative=np.minimum(self.dribbles,2)*.15+np.minimum(self.selfpasses,1)*.1+np.minimum(self.passes,2)*.05
        return np.minimum((creative*own).sum(-1),.5)*(state.scored>=0)

    def summary(self):
        return {**self.total,'cluster_fraction':self.cluster_samples/max(self.samples,1),
                'loop_fraction':self.loop_samples/max(self.samples/max(self.p,1)/90,1)}
