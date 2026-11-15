"""One versioned observation/action contract for native training and live play.

No ordering-dependent player features. All movement is in an x-mirrored frame
for blue; actions are mirrored back exactly once in ``decode_actions``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from haxballgym.obs import GeoObs
from haxballgym.stadium import stadium_text

SCHEMA_VERSION = 3
EGO_DIM = 60
ENTITY_DIM = 10
MAX_PLAYERS = 24
ACTION_TABLE = np.asarray([(x, y, k) for k in (0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=np.int64)


@dataclass
class FieldSpec:
    name: str
    half: np.ndarray
    player_radius: float = 15.0
    ball_radius: float = 10.0
    acceleration: float = 0.1
    damping: float = 0.96
    ball_damping: float = 0.99
    kick_strength: float = 5.0
    ball_inv_mass: float = 1.0
    ball_bounce: float = .5
    player_bounce: float = .5
    kicking_acceleration: float = .07
    kicking_damping: float = .96
    ball_gravity: tuple = (0.,0.)
    kickback: float = 0.
    player_half: np.ndarray | None = None
    player_walls: np.ndarray | None = None
    player_obstacles: np.ndarray | None = None

    @property
    def max_speed(self):
        # Upstream's getter divides by (1-accel), which is NOT terminal speed.
        return self.acceleration * self.damping / max(1-self.damping, 1e-6)

    @classmethod
    def from_state(cls, state, stadium='classic'):
        raw = json.loads(stadium_text(stadium))
        return cls.from_raw(state,raw,stadium)

    @classmethod
    def from_raw(cls,state,raw,name='live'):
        pp = raw.get('playerPhysics') or {}
        bp = raw.get('ballPhysics') or {}
        if not isinstance(bp, dict):
            bp = (raw.get('discs') or [{}])[0]
        pts = np.asarray(state.walls).reshape(-1, 2)
        useful = pts[np.abs(pts).max(axis=1) < 2999]
        half = np.maximum(np.abs(useful if len(useful) else pts).max(axis=0), 50.0)
        traits=raw.get('traits') or {}
        if not isinstance(traits,dict):traits={}
        def props(item):return {**traits.get(item.get('trait'),{}),**item}
        flags={'ball':1,'red':2,'blue':4,'redKO':8,'blueKO':16,'wall':32,'all':63,'kick':64,'score':128}
        def bits(value,default):
            if value is None:return default
            result=0
            for x in value:result |= flags.get(x,0)
            return result
        def player_collides(item):
            item=props(item)
            return bool(bits(item.get('cMask'),63)&6 and bits(item.get('cGroup'),32)&39)
        player_half=np.full(2,np.inf)
        walls=[];obstacles=[]
        for plane in raw.get('planes',[]):
            if not player_collides(plane):continue
            normal=np.asarray(plane['normal'],dtype=float);dist=float(plane['dist'])
            if abs(normal[0])>1e-8 and abs(normal[1])<1e-8:player_half[0]=min(player_half[0],abs(dist/normal[0]))
            if abs(normal[1])>1e-8 and abs(normal[0])<1e-8:player_half[1]=min(player_half[1],abs(dist/normal[1]))
        player_half=np.where(np.isfinite(player_half),player_half,half+30)
        hx,hy=player_half
        walls.extend([[-hx,-hy,hx,-hy],[hx,-hy,hx,hy],[hx,hy,-hx,hy],[-hx,hy,-hx,-hy]])
        vertices=raw.get('vertexes',[])
        for segment in raw.get('segments',[]):
            if player_collides(segment):
                a,b=vertices[segment['v0']],vertices[segment['v1']]
                walls.append([a['x'],a['y'],b['x'],b['y']])
        ball_index=int(raw.get('ballPhysics','')[4:]) if isinstance(raw.get('ballPhysics'),str) and raw['ballPhysics'].startswith('disc') else -1
        for i,disc in enumerate(raw.get('discs',[])):
            d=props(disc)
            if i==ball_index or d.get('invMass',1)!=0 or not player_collides(d):continue
            pos=d.get('pos',[0,0]);obstacles.append([pos[0],pos[1],d.get('radius',10)])
        return cls(name, half, float(pp.get('radius', 15)), float(bp.get('radius', 10)),
                   float(pp.get('acceleration', .1)), float(pp.get('damping', .96)),
                   float(bp.get('damping', .99)), float(pp.get('kickStrength', 5)),
                   float(bp.get('invMass',1)),float(bp.get('bCoef',.5)),float(pp.get('bCoef',.5)),
                   float(pp.get('kickingAcceleration',.07)),float(pp.get('kickingDamping',.96)),
                   tuple(bp.get('gravity',[0.,0.])),float(pp.get('kickback',0)),
                   player_half,np.asarray(walls,dtype=float),np.asarray(obstacles,dtype=float).reshape(-1,3))

    def as_dict(self):
        return {**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.__dict__.items()}, 'max_speed': self.max_speed}


def action_indices(world_actions, teams):
    a = np.asarray(world_actions, dtype=np.int64).copy()
    a[..., 0] *= np.where(teams == 2, 1, -1)
    return (a[..., 1]+1)*3 + (a[..., 0]+1) + 9*a[..., 2]


def decode_actions(indices, teams):
    out = ACTION_TABLE[np.asarray(indices, dtype=np.int64)].copy()
    out[..., 0] *= np.where(np.asarray(teams) == 2, 1, -1)
    return out


def _rays(origins, state, scale,walls=None,obstacles=None):
    shape = origins.shape[:-1]
    o = origins.reshape(-1, 2)
    ang = np.arange(8)*np.pi/4
    dirs = np.broadcast_to(np.stack([np.cos(ang), np.sin(ang)], -1), (len(o), 8, 2))
    walls = np.asarray(state.walls if walls is None else walls)
    cap = float(2*np.linalg.norm(scale))
    if len(walls):
        result = GeoObs._raycast(o, dirs, walls[:, :2], walls[:, 2:], cap)
    else:
        result = np.ones((len(o), 8))
    obs = np.asarray(state.obstacles if obstacles is None else obstacles)
    if len(obs):
        result = np.minimum(result, GeoObs._raycast_discs(o, dirs, obs[:, :2], obs[:, 2], cap))
    return result.reshape(*shape, 8)


def encode(state, spec: FieldSpec, egos=None, previous_actions=None,observation_revision='player-geometry-v2'):
    """Return arrays (B,60), (B,P,10), (B,P), with B=N*number of chosen egos.

    ``egos`` is (N,) or (N,L). Absent means every player. Entity P is truly
    variable: model reductions use a mask, without truncating distant threats.
    """
    n, p = state.player_pos.shape[:2]
    if p > MAX_PLAYERS:
        raise ValueError(f'observation contract supports <= {MAX_PLAYERS} players, got {p}')
    if egos is None:
        egos = np.broadcast_to(np.arange(p), (n, p))
    egos = np.asarray(egos, dtype=np.int64)
    if egos.ndim == 1:
        egos = egos[:, None]
    rows = np.broadcast_to(np.arange(n)[:, None], egos.shape)
    ep = state.player_pos[rows, egos]
    ev = state.player_vel[rows, egos]
    et = state.team[rows, egos]
    side = np.where(et == 2, 1.0, -1.0)
    mirror = np.stack([side, np.ones_like(side)], -1)
    target = state.goal_center(True)[rows, egos]
    own = state.goal_center(False)[rows, egos]
    ball = state.ball_pos[:, None]
    bv = np.broadcast_to(state.ball_vel[:, None], ep.shape)
    half = spec.half
    rel = ball-ep
    dist = np.linalg.norm(rel, axis=-1)
    vmax = max(spec.max_speed, .1)
    ego = np.zeros((*egos.shape, EGO_DIM), dtype=np.float32)
    ego[..., 0:2] = ep*mirror/half
    ego[..., 2:4] = ev*mirror/vmax
    ego[..., 4:6] = rel*mirror/half
    ego[..., 6:8] = bv*mirror/10
    ego[..., 8:10] = (target-ep)*mirror/half
    ego[..., 10:12] = (own-ep)*mirror/half
    ego[..., 12] = dist/np.linalg.norm(half)
    ego[..., 13] = dist < spec.player_radius+spec.ball_radius+4
    tg0, tg1 = state.goal_line(True)
    ego[..., 14] = np.linalg.norm(tg1[rows, egos]-tg0[rows, egos], axis=-1)/(2*half[1])
    same = state.team[:, None] == et[..., None]
    ego[..., 15] = same.sum(-1)/12
    ego[..., 16] = (~same).sum(-1)/12
    all_dist = np.linalg.norm(state.player_pos-state.ball_pos[:, None], axis=-1)
    mine = dist[..., None]
    ego[..., 17] = ((all_dist[:, None] < mine-1e-4) & same).sum(-1)/12
    ego[..., 18] = ((all_dist[:, None] < mine-1e-4) & ~same).sum(-1)/12
    if previous_actions is not None:
        pa = np.asarray(previous_actions)[rows, egos]
        ego[..., 19:21] = pa[..., :2]*mirror
        ego[..., 21] = pa[..., 2]
    # Rays in world frame; reorder, rather than negate distances, for blue.
    ball_rays = _rays(state.ball_pos, state, half)
    br = np.broadcast_to(ball_rays[:, None], (*egos.shape, 8))
    er = _rays(ep,state,half) if observation_revision=='ball-geometry-v1' else _rays(ep, state, half,spec.player_walls,spec.player_obstacles)
    flip = np.array([4, 3, 2, 1, 0, 7, 6, 5])
    ego[..., 22:30] = np.where((side < 0)[..., None], br[..., flip], br)
    ego[..., 30:38] = np.where((side < 0)[..., None], er[..., flip], er)
    ego[..., 38] = half[0]/600
    ego[..., 39] = half[1]/300
    ego[..., 40] = vmax/5
    ego[..., 41] = spec.ball_damping
    ego[..., 42] = spec.player_radius/30
    ego[..., 43] = spec.ball_radius/20
    ego[..., 44] = spec.kick_strength/10
    ego[..., 45] = spec.acceleration
    ego[..., 46] = spec.damping
    ego[..., 47] = np.clip(state.steps[:, None]/10800, 0, 1)
    if state.ball_pred is not None and state.ball_pred.shape[1]>=3:
        ego[..., 48:50] = (state.ball_pred[:,None,1]-ep)*mirror/half
        ego[..., 50:52] = (state.ball_pred[:,None,2]-ep)*mirror/half
    else:
        for start,ticks in [(48,6),(50,18)]:
            drag=(1-spec.ball_damping**ticks)/max(1-spec.ball_damping,1e-6)
            ego[...,start:start+2]=(ball+bv*drag-ep)*mirror/half
    ego[...,52]=spec.ball_inv_mass/5
    ego[...,53]=spec.ball_bounce/2
    ego[...,54]=spec.player_bounce/2
    ego[...,55]=spec.kicking_acceleration/max(spec.acceleration,1e-6)
    ego[...,56]=spec.kicking_damping
    ego[...,57:59]=np.asarray(spec.ball_gravity)[None,None]*mirror
    ego[...,59]=spec.kickback/5
    entity = np.zeros((*egos.shape, p, ENTITY_DIM), dtype=np.float32)
    entity[..., 0:2] = (state.player_pos[:, None]-ep[..., None, :])*mirror[..., None, :]/half
    entity[..., 2:4] = state.player_vel[:, None]*mirror[..., None, :]/vmax
    entity[..., 4] = same
    entity[..., 5] = np.arange(p)[None, None] == egos[..., None]
    entity[..., 6:8] = (state.ball_pos[:, None, None]-state.player_pos[:, None])*mirror[..., None, :]/half
    entity[..., 8] = all_dist[:, None]/np.linalg.norm(half)
    entity[..., 9] = np.linalg.norm(state.player_pos[:, None]-own[..., None, :], axis=-1)/np.linalg.norm(half)
    np.clip(ego, -8, 8, out=ego)
    np.clip(entity, -8, 8, out=entity)
    return ego.reshape(-1, EGO_DIM), entity.reshape(-1, p, ENTITY_DIM), np.ones((egos.size, p), dtype=bool)
