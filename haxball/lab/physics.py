"""Coordinate compatibility layer for the pinned native HaxBall port.

The upstream port reflects stadium geometry into y-up coordinates. A native
client observes y-down. Symmetric zero-bias maps hid this difference; BFF's
one-sided walls exposed it. Pre-reflect only fields the native loader reflects,
while leaving goals and observed dynamic state in real game coordinates.
"""
from __future__ import annotations
import copy,json
import numpy as np
import haxball_core
from haxballgym import TransitionEngine
from haxballgym.stadium import stadium_text


def compatible_hbs(raw):
    data=copy.deepcopy(raw)
    if not isinstance(data.get('traits',{}),dict):data['traits']={}
    for vertex in data.get('vertexes',[]):vertex['y']=-vertex['y']
    for disc in data.get('discs',[]):
        if 'pos' in disc:disc['pos'][1]=-disc['pos'][1]
    for plane in data.get('planes',[]):plane['normal'][1]=-plane['normal'][1]
    for key in ('redSpawnPoints','blueSpawnPoints'):
        for pos in data.get(key,[]):pos[1]=-pos[1]
    for props in data.get('segments',[])+list(data.get('traits',{}).values()):
        if 'curve' in props:props['curve']=-props['curve']
        props.pop('curveF',None)
    return json.dumps(data,separators=(',',':'))


class Engine(TransitionEngine):
    def __init__(self,n_envs,n_red=1,n_blue=1,step_limit=2000,tick_skip=2,stadium='classic',predict_offsets=None):
        self._predict_offsets=list(predict_offsets) if predict_offsets else None
        if self._predict_offsets and (self._predict_offsets[0]<1 or any(b<=a for a,b in zip(self._predict_offsets,self._predict_offsets[1:]))):
            raise ValueError('prediction offsets must be positive and strictly ascending')
        self._e=haxball_core.VecEnv.from_hbs(compatible_hbs(json.loads(stadium_text(stadium))),n_envs,n_red,n_blue,step_limit=step_limit,tick_skip=tick_skip)
        self.stadium=stadium;self.n_envs=self._e.n_envs;self.n_players=self._e.n_players
        self.player_max_speed=self._e.player_max_speed
        self._no_goal=np.full(n_envs,-1,dtype=np.int8)
        self._teams=self._e.teams()
        self._goal_p0,self._goal_p1,self._goal_team=self._e.goals()
        self._walls=self._e.wall_segments();self._obstacles=self._e.obstacle_discs()
