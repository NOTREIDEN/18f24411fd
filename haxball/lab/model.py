"""Permutation-invariant recurrent policy, small enough for CPU and browser use."""
from __future__ import annotations
import math
import torch
from torch import nn
from .schema import EGO_DIM, ENTITY_DIM, SCHEMA_VERSION


class Policy(nn.Module):
    hidden_size = 128

    def __init__(self):
        super().__init__()
        self.observation_revision='player-geometry-v2'
        self.ego = nn.Sequential(nn.Linear(EGO_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.entity = nn.Sequential(nn.Linear(ENTITY_DIM, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.query = nn.Linear(64, 32)
        self.fuse = nn.Sequential(nn.Linear(160, 128), nn.Tanh())
        self.memory = nn.GRUCell(128, 128)
        self.actor = nn.Linear(128, 18)
        self.critic = nn.Linear(128, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1)

    def initial(self, n, device=None):
        return torch.zeros(n, self.hidden_size, device=device or next(self.parameters()).device)

    def forward(self, ego, entities, mask, hidden):
        e = self.ego(ego)
        tokens = self.entity(entities)
        scores = (tokens*self.query(e)[:, None]).sum(-1)/math.sqrt(32)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        attended = (scores.softmax(-1)[..., None]*tokens).sum(1)
        friendly = (entities[..., 4] > .5) & mask.bool()
        hostile = ~friendly & mask.bool()
        fm = (tokens*friendly[..., None]).sum(1)/friendly.sum(1, keepdim=True).clamp_min(1)
        hm = (tokens*hostile[..., None]).sum(1)/hostile.sum(1, keepdim=True).clamp_min(1)
        h = self.memory(self.fuse(torch.cat([e, attended, fm, hm], -1)), hidden)
        return self.actor(h), self.critic(h).squeeze(-1), h

    @torch.no_grad()
    def act(self, obs, hidden=None, deterministic=False):
        device = next(self.parameters()).device
        tensors = [torch.as_tensor(a, device=device) for a in obs]
        if hidden is None: hidden = self.initial(len(tensors[0]), device)
        logits, value, h = self(*tensors, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, h


def save_checkpoint(path, policy, optimizer=None, **metadata):
    from pathlib import Path
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    import os
    payload = {'schema': SCHEMA_VERSION, 'model': policy.state_dict(),'observation_revision':policy.observation_revision,
               'code_revision':os.getenv('GITHUB_SHA','local-refinement'), **metadata}
    if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
    tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)


def load_checkpoint(path, device='cpu'):
    obj = torch.load(path, map_location=device, weights_only=False)
    if obj.get('schema')==2 and SCHEMA_VERSION==3:
        # Preserve the trained policy exactly while adding physical context.
        old=obj['model']['ego.0.weight'];expanded=torch.zeros(old.shape[0],EGO_DIM,device=old.device)
        expanded[:,:old.shape[1]]=old;obj['model']['ego.0.weight']=expanded
        obj['schema']=3;obj['migrated_from_schema']=2;obj.pop('optimizer',None)
    if obj.get('schema') != SCHEMA_VERSION: raise ValueError('checkpoint observation schema mismatch')
    model = Policy().to(device); model.load_state_dict(obj['model']); model.eval()
    model.observation_revision=obj.get('observation_revision','ball-geometry-v1')
    return model, obj
