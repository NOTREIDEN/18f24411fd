"""Differential checks of the native core against the real JS engine."""
import json,subprocess
from pathlib import Path
import numpy as np
import pytest
import haxball_core
from lab.physics import compatible_hbs

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'node_modules/node-haxball').exists(),reason='npm ci installs JS oracle')
def test_player_contacts_kicks_and_large_games_match_js():
    data=json.loads(subprocess.check_output(['node','scripts/player_oracle.cjs'],cwd=ROOT))
    for sc in data:
        n=sc['size'];e=haxball_core.VecEnv.from_hbs(compatible_hbs(sc['stadium']),1,n,n,tick_skip=1)
        e.set_kick_rate_limit(sc['minKick'],0,1)
        st=sc['initial']
        e.set_state(np.array([st['bp']],dtype=float),np.array([st['bv']],dtype=float),np.array([st['pp']],dtype=float),np.array([st['pv']],dtype=float),None)
        worst=0.
        for actions,expected in zip(sc['actions'],sc['trajectory'],strict=True):
            e.physics_step(np.array([actions],dtype=np.int64));bp,bv,pp,pv,_=e.snapshot()
            for actual,key in [(bp[0],'bp'),(bv[0],'bv'),(pp[0],'pp'),(pv[0],'pv')]:
                delta=float(np.abs(actual-np.asarray(expected[key])).max());worst=max(worst,delta)
                assert delta<1e-7,f"{sc['name']} {key} diverged by {delta}"
        print(sc['name'],n,'players per team; max difference',worst)
