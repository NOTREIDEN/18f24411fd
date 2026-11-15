import numpy as np
import torch
from haxballgym import TransitionEngine
from lab.schema import FieldSpec,encode,decode_actions,action_indices,EGO_DIM
from lab.model import Policy


def test_terminal_speed_uses_damping_not_acceleration():
    spec=FieldSpec('test',np.array([420.,200.]))
    assert abs(spec.max_speed-2.4)<1e-12


def test_action_roundtrip_for_both_teams():
    idx=np.arange(18)
    for team in (2,4):
        teams=np.full(18,team)
        np.testing.assert_array_equal(idx,action_indices(decode_actions(idx,teams),teams))


def test_policy_accepts_large_games_and_entity_permutations():
    torch.set_num_threads(1);model=Policy().eval()
    for size in (1,3,10,12):
        engine=TransitionEngine(2,size,size,tick_skip=2)
        state=engine.reset();spec=FieldSpec.from_state(state)
        obs=encode(state,spec,egos=np.array([0,size]))
        assert obs[0].shape==(2,EGO_DIM)
        with torch.no_grad():
            original=model(*(torch.from_numpy(x) for x in obs),model.initial(2))[0]
            perm=np.random.default_rng(1).permutation(size*2)
            altered=(obs[0],obs[1][:,perm],obs[2][:,perm])
            reordered=model(*(torch.from_numpy(x.copy()) for x in altered),model.initial(2))[0]
        torch.testing.assert_close(original,reordered,atol=1e-6,rtol=1e-5)
