import numpy as np
from lab.arena import Arena
from lab.teacher import baseline_actions
from lab.maps import futsal_for


def test_reference_releases_kick_and_rearms_in_contact():
    e=Arena(1,1,futsal_for(1),random_start=False)
    bp=np.array([[0.,0.]]);bv=np.zeros((1,2));pp=np.array([[[-100.,40.],[22.,0.]]]);pv=np.zeros_like(pp)
    e.state=e.engine.set_state(bp,bv,pp,pv)
    released=np.zeros((1,2,3),dtype=np.int64)
    first=baseline_actions(e.state,e.spec,'chase',released)
    assert first[0,1,2]==1
    second=baseline_actions(e.state,e.spec,'chase',first)
    assert second[0,1,2]==0


def test_random_starts_are_not_horizontal_replicas():
    e=Arena(128,3,futsal_for(3),seed=81)
    delta=e.state.player_pos[np.arange(e.n),e.egos]-e.state.ball_pos
    assert (np.abs(delta[:,1])>25).mean()>.45
    assert len(np.unique(np.round(delta,1),axis=0))>120


def test_futsal_player_can_position_outside_ball_boundary():
    e=Arena(1,1,futsal_for(1),random_start=False)
    assert e.spec.player_half[1]>e.spec.half[1]+e.spec.player_radius
    bp=np.array([[0.,e.spec.half[1]-e.spec.ball_radius-1]])
    pp=np.array([[[-26.,bp[0,1]-35],[120.,0.]]])
    e.state=e.engine.set_state(bp,np.zeros((1,2)),pp,np.zeros_like(pp))
    old=e.obs('ball-geometry-v1')[0]
    new=e.obs('player-geometry-v2')[0]
    assert np.isfinite(new).all()
    assert not np.array_equal(old,new), 'player lidar must reflect its own walls'
