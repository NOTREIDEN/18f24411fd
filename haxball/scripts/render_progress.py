"""Render actual native-engine evaluations; never synthesize successful actions."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from lab.arena import Arena
from lab.schema import action_indices,encode,decode_actions
from lab.teacher import baseline_actions
from lab.maps import futsal_for


def render(output,checkpoint=None,seconds=12,formats=(1,3,10),seed=137,fps=30):
    model=None;meta={}
    if checkpoint:
        import torch
        from lab.model import load_checkpoint
        torch.set_num_threads(1);model,meta=load_checkpoint(checkpoint)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    width,height=960,640
    encoder=shutil.which('ffmpeg')
    if encoder is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        encoder=get_ffmpeg_exe()
    pipe=subprocess.Popen([encoder,'-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
        title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
    except OSError:font=small=title=ImageFont.load_default()
    reports=[];traces={}
    for size in formats:
        stadium=futsal_for(size)
        env=Arena(1,size,stadium,seed+size,tick_skip=2,max_ticks=6000,random_start=True,continuous=True)
        env.egos[:]=0
        h=None;score=[0,0];trace=[];my_activity=[];opponent_activity=[];my_kicks=0;opponent_kicks=0;last_keys=np.zeros((env.p,3),dtype=np.int64)
        scale=min((width-80)/(2*env.spec.half[0]),(height-170)/(2*env.spec.half[1]))
        center=np.array([width/2,110+(height-170)/2])
        def xy(v):return tuple(np.asarray(v)*scale+center)
        for frame in range(int(seconds*fps)):
            state=env.state
            if model:
                obs=env.obs(model.observation_revision);a,_,_,h=model.act(obs,h,deterministic=True);indices=a.numpy()
            else:
                acts=baseline_actions(state,env.spec,previous_actions=env.previous)
                indices=action_indices(acts,state.team)[:,0]
            _,_,term,trunc,info=env.step(indices)
            if info['goal_sign'][0]!=0:
                score[0 if info['goal_sign'][0]>0 else 1]+=1
            if ((term|trunc)[0] or info['goal_sign'][0]!=0) and h is not None:h.zero_()
            env.egos[:]=0
            state=info['terminal_state']
            speed=np.linalg.norm(state.player_vel[0],axis=-1)
            other=state.team[0]!=state.team[0,0]
            my_activity.append(float(speed[0]>.15*env.spec.max_speed))
            opponent_activity.append(float((speed[other]>.15*env.spec.max_speed).mean()))
            keys=info['actions'][0]
            rising=(keys[:,2]>0)&(last_keys[:,2]==0)
            my_kicks+=int(rising[0]);opponent_kicks+=int(rising[other].sum());last_keys=keys.copy()
            trace.append(np.concatenate([state.ball_pos[0],state.ball_vel[0],state.player_pos[0].ravel(),state.player_vel[0].ravel(),keys.ravel()]).astype(np.float32))
            image=Image.new('RGB',(width,height),(15,23,35));draw=ImageDraw.Draw(image)
            draw.text((22,16),'NOTREIDEN | HaxBall Lab',font=title,fill=(240,244,250))
            stage=('IMITACION INICIAL' if meta.get('phase')=='imitation' else 'POLITICA PPO') if model else 'CONTROLADOR INICIAL'
            label=Path(stadium).stem if '/' in stadium else stadium
            draw.text((22,48),f'{stage} | {size}v{size} | {label} | {frame/fps:.1f}s | {score[0]} : {score[1]}',font=font,fill=(180,202,222))
            draw.text((22,74),f'Actividad rival: {np.mean(opponent_activity):.0%} | pulsaciones de patada rivales: {opponent_kicks}',font=small,fill=(180,202,222))
            x0,y0=xy(-env.spec.half);x1,y1=xy(env.spec.half)
            draw.rectangle((x0,y0,x1,y1),fill=(37,89,72),outline=(120,155,144),width=2)
            draw.line((xy([0,-env.spec.half[1]]),xy([0,env.spec.half[1]])),fill=(74,123,103),width=1)
            radius=75*scale;cx,cy=xy([0,0]);draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=(74,123,103))
            for wall in state.walls:
                if np.abs(wall).max()<1500:draw.line((xy(wall[:2]),xy(wall[2:])),fill=(202,213,219),width=2)
            for a,b,team in zip(state.goal_p0,state.goal_p1,state.goal_team):
                draw.line((xy(a),xy(b)),fill=(245,100,104) if team==2 else (105,164,245),width=4)
            for obstacle in state.obstacles:
                x,y=xy(obstacle[:2]);r=obstacle[2]*scale;draw.ellipse((x-r,y-r,x+r,y+r),fill=(210,216,220))
            for k,pos in enumerate(state.player_pos[0]):
                x,y=xy(pos);r=max(2.5,env.spec.player_radius*scale)
                if k==0:draw.ellipse((x-r-3,y-r-3,x+r+3,y+r+3),outline=(255,218,90),width=3)
                draw.ellipse((x-r,y-r,x+r,y+r),fill=(235,83,94) if state.team[0,k]==2 else (79,141,232),outline=(18,29,39),width=2)
                draw.text((x-4,y-7),str(k+1),font=small,fill=(255,255,255))
            x,y=xy(state.ball_pos[0]);r=max(1.5,env.spec.ball_radius*scale)
            draw.ellipse((x-r,y-r,x+r,y+r),fill=(250,250,247),outline=(12,22,31),width=1)
            summary=env.tracker.summary()
            draw.text((22,height-58),'Anillo amarillo: nuestro unico jugador. Resto: rivales/companeros de referencia.',font=small,fill=(188,202,218))
            draw.text((22,height-37),f'Pases confirmados: {summary["passes_completed"]} | agrupamiento: {summary["cluster_fraction"]:.1%} | checkpoint: {Path(checkpoint).name if checkpoint else "baseline"}',font=small,fill=(188,202,218))
            pipe.stdin.write(image.tobytes())
        reports.append({'format':size,'score':score,'metrics':env.tracker.summary(),'stadium':stadium,
                        'my_motion_fraction':float(np.mean(my_activity)),'opponent_motion_fraction':float(np.mean(opponent_activity)),
                        'my_kick_presses':my_kicks,'opponent_kick_presses':opponent_kicks})
        traces[str(size)]=np.stack(trace)
    pipe.stdin.close()
    if pipe.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
    report={'checkpoint':str(checkpoint) if checkpoint else None,'phase':meta.get('phase','baseline'),'training_samples':meta.get('samples',0),'source':'actual native-engine trajectories; evaluation visualization, not official client recording','formats':reports}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(output.with_suffix('.npz'),**traces)
    print(json.dumps({'video':str(output),'report':report}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/progress-000.mp4');p.add_argument('--checkpoint');p.add_argument('--seconds',type=int,default=12);p.add_argument('--formats',default='1,3,10');p.add_argument('--seed',type=int,default=137);a=p.parse_args()
    render(a.output,a.checkpoint,a.seconds,tuple(map(int,a.formats.split(','))),a.seed)
