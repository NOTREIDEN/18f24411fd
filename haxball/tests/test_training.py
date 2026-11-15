import numpy as np
import torch
from lab.arena import Arena
from lab.model import Policy
from scripts.train import collect,update


def test_recurrent_update_with_entire_learned_opposing_team():
    torch.set_num_threads(1)
    model=Policy();opponent=Policy().eval();env=Arena(2,3,'big',seed=77)
    rng=np.random.default_rng(7)
    data,hidden,metrics=collect(env,model,8,None,rng,opponent,full_opponents=True)
    assert data['actions'].shape==(8,2)
    assert hidden.shape==(2,128)
    assert np.isfinite(data['returns']).all()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    result=update(model,opt,data,rng,epochs=1,sequence=4,sequences_per_batch=4)
    assert np.isfinite(result['loss'])
    assert metrics['physics_ticks']==32
