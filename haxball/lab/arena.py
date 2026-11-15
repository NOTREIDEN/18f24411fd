"""Native batched matches with one independently controlled learning slot."""
from __future__ import annotations
import numpy as np
from .physics import Engine
from .schema import FieldSpec, encode, decode_actions
from .teacher import baseline_actions
from .metrics import EventTracker


class Arena:
    def __init__(self,n_envs=64,team_size=1,stadium='classic',seed=0,tick_skip=2,max_ticks=1800,gamma=.999,random_start=True,continuous=False):
        self.rng=np.random.default_rng(seed);self.n=n_envs;self.team_size=team_size
        self.tick_skip=tick_skip;self.max_ticks=max_ticks;self.gamma=gamma;self.stadium=stadium
        self.continuous=continuous
        self.engine=Engine(n_envs,team_size,team_size,tick_skip=tick_skip,stadium=stadium,predict_offsets=[tick_skip,6,18])
        self.state=self.engine.reset();self.spec=FieldSpec.from_state(self.state,stadium)
        self.p=2*team_size;self.egos=self.rng.integers(0,self.p,size=n_envs)
        self.age=np.zeros(n_envs,dtype=np.int64)
        self.previous=np.zeros((self.n,self.p,3),dtype=np.int64)
        self.tracker=EventTracker(self.n,self.p,self.spec)
        self.goal_for=0;self.goal_against=0;self.episodes=0;self.full_matches=0
        if random_start:self.randomize_all()

    def randomize_all(self):
        """Only call at a batch boundary: set_state resets ALL hidden kick states."""
        hx,hy=self.spec.half
        bp=self.rng.uniform([-hx*.65,-hy*.7],[hx*.65,hy*.7],(self.n,2))
        # Deliberate wall-escape drills, with room for the PLAYER outside the
        # ball-only boundary. These are real legal states of the selected map.
        wall_case=self.rng.random(self.n)<.25
        bp[wall_case,1]=self.rng.choice([-1,1],int(wall_case.sum()))*(hy-self.spec.ball_radius-2)
        bv=self.rng.normal(0,.6,(self.n,2))
        pp=self.rng.uniform([-hx*.85,-hy*.8],[hx*.85,hy*.8],(self.n,self.p,2))
        pv=self.rng.normal(0,.25,(self.n,self.p,2))
        # Distribute roles while retaining contact-rich starts for the learning slot.
        for i in range(self.p):
            for _ in range(6):
                d=np.linalg.norm(pp[:,i,None]-pp[:,:i],axis=-1)
                bad=(d<2*self.spec.player_radius+3).any(-1)|(np.linalg.norm(pp[:,i]-bp,axis=-1)<self.spec.player_radius+self.spec.ball_radius+3)
                if not bad.any():break
                pp[bad,i]=self.rng.uniform([-hx*.85,-hy*.8],[hx*.85,hy*.8],(int(bad.sum()),2))
        rows=np.arange(self.n);side=np.where(self.state.team[rows,self.egos]==2,1.,-1.)
        contact=(self.rng.random(self.n)<.55)|wall_case
        offset=self.rng.uniform(29,100,self.n)
        angles=self.rng.uniform(-np.pi,np.pi,self.n)
        around=np.stack([np.cos(angles),np.sin(angles)],-1)*offset[:,None]
        pp[rows[contact],self.egos[contact]]=bp[contact]+around[contact]
        player_half=self.spec.player_half if self.spec.player_half is not None else self.spec.half
        np.clip(pp,-player_half+self.spec.player_radius+3,player_half-self.spec.player_radius-3,out=pp)
        # Reject impossible overlaps introduced by placing the selected learner.
        for _ in range(12):
            mine=pp[rows,self.egos]
            d=np.linalg.norm(pp-mine[:,None],axis=-1);d[rows,self.egos]=1e9
            bad=(d<2*self.spec.player_radius+2).any(-1)|(np.linalg.norm(mine-bp,axis=-1)<self.spec.player_radius+self.spec.ball_radius+2)
            if not bad.any():break
            angle=self.rng.uniform(-np.pi,np.pi,int(bad.sum()));off=self.rng.uniform(28,110,int(bad.sum()))
            pp[rows[bad],self.egos[bad]]=bp[bad]+np.stack([np.cos(angle),np.sin(angle)],-1)*off[:,None]
            np.clip(pp,-player_half+self.spec.player_radius+3,player_half-self.spec.player_radius-3,out=pp)
        self.state=self.engine.set_state(bp,bv,pp,pv)
        self.age[:]=0;self.previous[:]=0;self.tracker.reset_mask(np.ones(self.n,dtype=bool))

    def obs(self,observation_revision='player-geometry-v2'):
        return encode(self.state,self.spec,self.egos,self.previous,observation_revision)

    def potential(self,state):
        row=np.arange(self.n);teams=state.team[row,self.egos]
        side=np.where(teams==2,1.,-1.)
        progress=state.ball_pos[:,0]*side/self.spec.half[0]
        d=np.linalg.norm(state.player_pos-state.ball_pos[:,None],axis=-1)
        same=state.team==teams[:,None]
        friend=np.where(same,d,1e9).min(-1)
        enemy=np.where(~same,d,1e9).min(-1)
        control=np.tanh((enemy-friend)/100)
        return .15*np.tanh(progress)+.08*control

    def step(self,indices,opponent_style='balanced',style_weight=1.0,actions_override=None):
        before=self.state;oldphi=self.potential(before)
        actions=baseline_actions(before,self.spec,opponent_style,self.previous) if actions_override is None else actions_override.copy()
        row=np.arange(self.n);teams=before.team[row,self.egos]
        actions[row,self.egos]=decode_actions(indices,teams)
        terminal=self.engine.step(actions)
        events=self.tracker.update(before,terminal,actions,self.tick_skip)
        self.age+=self.tick_skip
        goal=terminal.scored>=0
        timeout=self.age>=self.max_ticks
        terminated=goal&~np.full(self.n,self.continuous,dtype=bool)
        truncated=timeout&~terminated
        sign=np.where(terminal.scored==teams,-1.,1.)*goal
        style=self.tracker.style_on_goal(terminal)*max(0,min(style_weight,2))
        nextphi=np.where(terminated,0,self.potential(terminal))
        reward=sign*(4.+style)+self.gamma*nextphi-oldphi
        self.goal_for+=int((sign>0).sum());self.goal_against+=int((sign<0).sum())
        self.previous=actions
        terminal_obs=encode(terminal,self.spec,self.egos,self.previous)
        done=terminated|truncated
        self.episodes+=int(done.sum())
        if self.continuous:
            self.full_matches+=int(timeout.sum())
            self.tracker.reset_mask(goal)
        if done.any():
            # reset_mask, NOT set_state, preserves ongoing worlds' kick cooldowns.
            self.engine.reset_mask(done)
            self.age[done]=0;self.previous[done]=0
            self.egos[done]=self.rng.integers(0,self.p,size=int(done.sum()))
            self.tracker.reset_mask(done)
            self.state=self.engine.snapshot()
        else:self.state=terminal
        return self.obs(),reward.astype(np.float32),terminated,truncated,{
            'terminal_obs':terminal_obs,'events':events,'style_reward':sign*style,
            'goal_sign':sign,'actions':actions,'terminal_state':terminal,
        }
