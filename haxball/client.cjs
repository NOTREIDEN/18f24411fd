'use strict';
const fs=require('node:fs');
const path=require('node:path');
const {spawn}=require('node:child_process');
const readline=require('node:readline');
const API=require('node-haxball')();

function snapshot(room,playerId,seq,frame){
 const state=room.state,gs=room.gameState;
 if(!gs||!state)return null;
 const players=state.players.filter(p=>p.disc&&p.team.id!==0).map(p=>({id:p.id,team:p.team.id,pos:[p.disc.pos.x,p.disc.pos.y],speed:[p.disc.speed.x,p.disc.speed.y]}));
 const b=gs.physicsState.discs[0];
 return {type:'state',seq,frame,playerId,players,ball:{pos:[b.pos.x,b.pos.y],speed:[b.speed.x,b.speed.y]}};
}

class Bridge {
 constructor({python='python3',checkpoint,onAction,onError}){
  const args=[path.join(__dirname,'scripts/infer.py')];if(checkpoint)args.push('--checkpoint',checkpoint);
  this.child=spawn(python,args,{stdio:['pipe','pipe','inherit']});this.onAction=onAction;this.onError=onError;
  this.pending=false;this.ready=false;this.started=0;this.seq=0;this.timings=[];
  this.child.on('error',e=>onError?.(e));this.child.on('exit',code=>{this.ready=false;if(!this.closing)onError?.(new Error('inference exited '+code));});
  readline.createInterface({input:this.child.stdout}).on('line',line=>{
   let m;try{m=JSON.parse(line)}catch{return onError?.(new Error('invalid worker output'));}
   if(m.type==='loaded')this.loaded=true;
   if(m.type==='ready'){this.ready=true;this.pending=false;}
   if(m.type==='error'){this.pending=false;return onError?.(new Error(m.message));}
   if(m.type==='action'){
    this.pending=false;const latency=performance.now()-this.started;this.timings.push(latency);if(this.timings.length>300)this.timings.shift();
    // A slow result must never keep applying an old movement plan.
    if(m.seq===this.seq&&latency<100)this.onAction(m);
   }
  });
 }
 send(msg){if(this.child.stdin.writable)this.child.stdin.write(JSON.stringify(msg)+'\n');}
 initialize(stadium){this.seq++;this.ready=false;this.pending=false;this.send({type:'init',stadium});}
 reset(){this.seq++;this.pending=false;this.send({type:'reset'});}
 request(msg){
  if(!this.ready||this.pending)return false;
  this.pending=true;this.started=performance.now();msg.seq=++this.seq;this.send(msg);return true;
 }
 close(){this.closing=true;this.child.kill('SIGTERM');}
}

function makePlugin(options){
 const {Plugin,AllowFlags,Utils}=API;
 function PlayerPlugin(){
  Object.setPrototypeOf(this,Plugin.prototype);
  Plugin.call(this,'notreiden_player',true,{version:'0.1.0',author:'NOTREIDEN HaxBall Lab',description:'Local recurrent policy; normal player inputs only',allowFlags:AllowFlags.JoinRoom});
  let bridge,frame=0,signature='',lastApplied=0;
  const release=()=>{try{this.room?.setKeyState(0);}catch{}};
  const reset=()=>{release();bridge?.reset();};
  this.initialize=()=>{
   bridge=new Bridge({...options,onAction:m=>{if(!this.room)return;this.room.setKeyState(Utils.keyState(m.dx,m.dy,m.kick));lastApplied=performance.now();},onError:e=>{release();console.error('[agent]',e.message);}});
  };
  this.onGameTick=()=>{
   frame++;const room=this.room;
   if(!room?.gameState||!room.currentPlayer?.disc)return release();
   if(performance.now()-lastApplied>250)release();
   const players=room.state.players.filter(p=>p.disc).map(p=>p.id+':'+p.team.id).join(',');
   const next=room.state.stadium.name+'|'+players;
   if(next!==signature){signature=next;bridge.initialize(JSON.parse(Utils.exportStadium(room.state.stadium)));return;}
   if(frame%2)return;
   const msg=snapshot(room,room.currentPlayerId,0,frame);if(msg)bridge.request(msg);
  };
  this.onPositionsReset=()=>{frame=0;reset();};
  this.onTeamGoal=reset;this.onGameStop=reset;
  this.onStadiumChange=()=>{signature='';reset();};
  this.onGamePauseChange=reset;
  this.finalize=()=>{release();bridge?.close();};
 }
 return new PlayerPlugin();
}

async function main(){
 const argv=process.argv.slice(2);const get=(name,def)=>{let i=argv.indexOf('--'+name);return i<0?def:argv[i+1];};
 const roomArg=get('room');
 if(!roomArg){console.error('Usage: node client.cjs --room <room URL or ID> --checkpoint runs/main/champion.pt --python <python> [--name NAME] [--password PASSWORD] [--auth-file PATH]');process.exitCode=2;return;}
 const id=roomArg.includes('?')?new URL(roomArg).searchParams.get('c'):roomArg;
 if(!id)throw new Error('invalid room ID');
 const authFile=path.resolve(get('auth-file',path.join(__dirname,'runs/client-auth.json')));
 let authObj;
 if(fs.existsSync(authFile)){
  const key=JSON.parse(fs.readFileSync(authFile,'utf8')).key;authObj=await API.Utils.authFromKey(key);
 }else{
  const [key,auth]=await API.Utils.generateAuth();authObj=auth;fs.mkdirSync(path.dirname(authFile),{recursive:true});fs.writeFileSync(authFile,JSON.stringify({key}),{mode:0o600});
 }
 let room;
 const plugin=makePlugin({python:get('python','python3'),checkpoint:get('checkpoint')});
 const joining=API.Room.join({id,password:get('password',null),authObj,token:process.env.HAXBALL_CLIENT_TOKEN},
  {storage:{player_name:get('name','NOTREIDEN IA'),avatar:'AI'},plugins:[plugin],
   onOpen:r=>{room=r;console.error('[agent] joined real room; player ID',r.currentPlayerId);},
   onClose:err=>{console.error('[agent] room closed',err?.message||'');process.exitCode=err?1:0;}});
 const stop=()=>{try{room?.setKeyState(0);room?.leave();joining?.cancel();}finally{process.exit(0);}};
 process.on('SIGINT',stop);process.on('SIGTERM',stop);
}

module.exports={snapshot,Bridge,makePlugin,API};
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
