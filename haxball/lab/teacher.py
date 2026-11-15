"""Predictive bootstrapping policy with role assignment and pass-lane geometry.

This is deliberately a transparent baseline, not a claim of professional play.
It is also an opponent and a guard against regressions in the learned policy.
"""
from __future__ import annotations
import numpy as np
from .schema import FieldSpec


def _unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def baseline_actions(state, spec: FieldSpec, style='balanced',previous_actions=None):
    pp, pv, bp, bv = state.player_pos, state.player_vel, state.ball_pos, state.ball_vel
    n, p = pp.shape[:2]
    target_goal, own_goal = state.goal_center(True), state.goal_center(False)
    attack = _unit(target_goal-own_goal)
    lateral = np.stack([-attack[..., 1], attack[..., 0]], -1)
    same = state.team[:, :, None] == state.team[:, None, :]
    eye = np.eye(p, dtype=bool)[None]
    to_ball = bp[:, None]-pp
    distances = np.linalg.norm(to_ball, axis=-1)
    reach = distances/np.maximum(spec.max_speed, .1)-np.sum(pv*_unit(to_ball), -1)*5
    canonical_y=pp[...,1]*np.where(state.team==2,1.,-1.)
    role_key=reach+canonical_y*1e-5+pp[...,0]*1e-8
    team_rank = ((role_key[:, None, :] < role_key[:, :, None]-1e-9) & same).sum(-1)
    active = team_rank == 0
    # Predict an interception with damping; keep inside approximate play bounds.
    ticks = np.clip(distances/np.maximum(spec.max_speed, .1)*.55, 2, 28)
    drag = (1-np.power(spec.ball_damping, ticks))/max(1-spec.ball_damping, 1e-6)
    future = bp[:, None]+bv[:, None]*drag[..., None]
    if state.ball_pred is not None and state.ball_pred.shape[1]>=3:
        # Interpolate between real wall-aware free-flight predictions.
        w=np.clip((ticks-6)/12,0,1)[...,None]
        predicted=state.ball_pred[:,None,1]*(1-w)+state.ball_pred[:,None,2]*w
        future=np.where((ticks>=6)[...,None],predicted,future)
    future = np.clip(future, -spec.half+spec.ball_radius, spec.half-spec.ball_radius)
    # Score shoot targets at left/centre/right of the actual goal mouth.
    g0, g1 = state.goal_line(True)
    shots = np.stack([g0*.8+g1*.2, (g0+g1)/2, g0*.2+g1*.8], -2)
    shot_vec = shots-bp[:, None, None]
    player_from_ball = pp[:, None, :, None]-bp[:, None, None, None]
    denom = np.maximum((shot_vec*shot_vec).sum(-1), 1e-6)
    t = (player_from_ball*shot_vec[:, :, None]).sum(-1)/denom[:, :, None]
    foot = bp[:, None, None, None]+t[..., None]*shot_vec[:, :, None]
    clearance = np.linalg.norm(pp[:, None, :, None]-foot, axis=-1)
    blocks = (~same)[..., None] & (t > .02) & (t < 1.02)
    open_shot = np.where(blocks, clearance, 1e6).min(-2)
    best_shot = open_shot.argmax(-1)
    rr, ii = np.indices((n, p))
    aim = shots[rr, ii, best_shot]
    best_clearance = open_shot[rr, ii, best_shot]
    # Candidate passes lead a moving teammate, reject interception-prone lanes.
    travel = np.linalg.norm(pp-bp[:, None], axis=-1)/max(spec.kick_strength*spec.ball_inv_mass, 1)
    receivers = pp+pv*np.clip(travel, 0, 40)[..., None]
    pass_vec = receivers[:, None]-bp[:, None, None]
    pd = np.maximum((pass_vec*pass_vec).sum(-1), 1e-6)
    rel_def = pp[:, None, None]-bp[:, None, None, None]
    pt = (rel_def*pass_vec[:, :, :, None]).sum(-1)/pd[..., None]
    pf = bp[:, None, None, None]+pt[..., None]*pass_vec[:, :, :, None]
    gap = np.linalg.norm(pp[:, None, None]-pf, axis=-1)
    defender = (~same)[:, :, None] & (pt > .03) & (pt < 1.05)
    lane = np.where(defender, gap, 1e6).min(-1)
    progress = ((receivers[:, None]-bp[:, None, None])*attack[:, :, None]).sum(-1)
    valid = same & ~eye & (np.linalg.norm(pass_vec, axis=-1)>60) & (lane > spec.player_radius+spec.ball_radius+12)
    pass_score = progress + np.minimum(lane, 100)*.8 - np.linalg.norm(pass_vec, axis=-1)*.08
    pass_score = np.where(valid, pass_score, -1e9)
    receiver_idx = pass_score.argmax(-1)
    pass_best = pass_score.max(-1)
    goal_dist = np.linalg.norm(target_goal-bp[:, None], axis=-1)
    should_pass = (pass_best > 45) & ((best_clearance < 38) | (goal_dist > spec.half[0]*1.1))
    if style == 'shooter': should_pass[:] = False
    if style == 'passer': should_pass = pass_best > 15
    aim = np.where(should_pass[..., None], receivers[rr, receiver_idx], aim)
    heading = _unit(aim-bp[:, None])
    behind = future-heading*(spec.player_radius+spec.ball_radius+2)
    # Approach from behind rather than drive the ball toward our own goal.
    contact_line = np.sum(_unit(to_ball)*heading, -1)
    near = distances < spec.player_radius+spec.ball_radius+12
    active_target = np.where((near & (contact_line > .75))[..., None], future+heading*22, behind)
    # Teammates spread into support lanes. Rank by lateral position is ID invariant.
    lateral_pos = np.sum(pp*lateral, -1)
    lane_rank = ((lateral_pos[:, None, :] < lateral_pos[:, :, None]) & same).sum(-1)
    teammates = same.sum(-1)
    lane_y = (lane_rank-(teammates-1)/2)*np.minimum(75, 2*spec.half[1]/np.maximum(teammates, 2))
    support = bp[:, None]-attack*np.minimum(130, spec.half[0]*.32)+lateral*lane_y[..., None]
    # Deepest available teammate covers goal; closer players support without chasing.
    depth = np.sum((pp-own_goal)*attack, -1)
    keeper_key=depth+canonical_y*1e-5+pp[...,0]*1e-8
    is_deepest = keeper_key <= np.where(same, keeper_key[:, None, :], 1e9).min(-1)+1e-9
    cover = own_goal + attack*55 + lateral*np.clip(np.sum((bp[:, None]-own_goal)*lateral,-1)*.35,-65,65)[..., None]
    support = np.where((is_deepest & (teammates>1))[..., None], cover, support)
    desired = np.where(active[..., None], active_target, support)
    if style == 'chase': desired = future
    if style == 'defensive': desired = np.where((distances < spec.half[0]*.5)[...,None],active_target,cover)
    # Soft personal space only for off-ball positioning, never a hard exclusion.
    diff = pp[:, :, None]-pp[:, None]
    ds = np.linalg.norm(diff, axis=-1)
    repel = (_unit(diff)*np.where(same & ~eye, np.clip(60-ds,0,60)/60,0)[...,None]).sum(-2)*30
    desired += np.where(active[...,None],0,repel)
    player_half=spec.player_half if spec.player_half is not None else spec.half
    desired = np.clip(desired,-player_half+spec.player_radius+3,player_half-spec.player_radius-3)
    # Velocity-aware steering avoids jitter/overshoot near the contact location.
    delta = desired-pp-pv*7
    dominant = np.abs(delta).max(-1, keepdims=True)
    directions = np.where((np.abs(delta) < 2.0) | (np.abs(delta) < dominant*.41421356), 0, np.sign(delta)).astype(np.int64)
    can_kick = distances < spec.player_radius+spec.ball_radius+3.6
    aligned = np.sum(_unit(to_ball)*heading, -1) > .93
    good_shot = best_clearance > spec.player_radius+spec.ball_radius+4
    kick = can_kick & aligned & (should_pass | good_shot | (goal_dist < 130))
    if style == 'chase': kick = can_kick
    # A held button fires only once. Explicit release/repress makes the rival
    # genuinely contest repeated contacts instead of becoming a stationary prop.
    if previous_actions is not None:kick &= np.asarray(previous_actions)[...,2]==0
    else:kick &= (state.steps[:,None]%4)<2
    return np.concatenate([directions,kick[...,None]],-1).astype(np.int64)
