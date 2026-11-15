import numpy as np
from haxballgym import TransitionEngine

def test_large_matches_are_finite_and_actionable():
    for size in (1,3,10,12):
        e=TransitionEngine(4,size,size,tick_skip=2,stadium='huge' if size>3 else 'classic')
        initial=e.reset()
        acts=np.zeros((4,size*2,3),dtype=np.int64);acts[:,:size,0]=1;acts[:,size:,0]=-1
        for _ in range(10): state=e.step(acts)
        assert state.player_pos.shape==(4,size*2,2)
        assert np.isfinite(state.player_pos).all()
        assert np.any(np.abs(initial.player_pos-state.player_pos)>0)

def test_reset_mask_does_not_change_other_worlds():
    e=TransitionEngine(3,1,1,tick_skip=1)
    e.reset(); a=np.ones((3,2,3),dtype=np.int64)
    for _ in range(10): e.step(a)
    before=e.snapshot(); e.reset_mask(np.array([True,False,False]));after=e.snapshot()
    np.testing.assert_array_equal(before.player_pos[1:],after.player_pos[1:])
    assert after.steps[0]==0
