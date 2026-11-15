import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lab.model import load_checkpoint
from lab.evaluate import evaluate

p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',default='reports/evaluation.json');p.add_argument('--envs',type=int,default=8);p.add_argument('--seconds',type=int,default=20);p.add_argument('--formats',default='1,3,10,12');p.add_argument('--unpaired',action='store_true');a=p.parse_args()
torch.set_num_threads(1)
model,meta=load_checkpoint(a.checkpoint) if a.checkpoint else (None,{})
report=evaluate(model,n_envs=a.envs,seconds=a.seconds,formats=tuple(map(int,a.formats.split(','))),paired=not a.unpaired)
report.update({'checkpoint':a.checkpoint,'checkpoint_samples':meta.get('samples',0)})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
