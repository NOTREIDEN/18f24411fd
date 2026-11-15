'use strict';
const assert=require('node:assert/strict');
const {API,Bridge,snapshot}=require('../client.cjs');
const room=API.Room.sandbox({},{});room.setSimulationSpeed(0);
for(let i=1;i<=6;i++){room.playerJoin(i,'test'+i,'',null,'','');room.setPlayerTeam(i,i<=3?1:2,0);}
room.startGame(0);
const python=process.env.HAXBALL_PYTHON||'python3';
let applied=0,tick=0,bridge;
const timeout=setTimeout(()=>{console.error('bridge selftest timed out');process.exit(1)},15000);
bridge=new Bridge({python,checkpoint:process.env.HAXBALL_CHECKPOINT,onError:e=>{throw e;},onAction:a=>{
 assert.ok([-1,0,1].includes(a.dx));assert.ok([-1,0,1].includes(a.dy));assert.equal(typeof a.kick,'boolean');
 room.playerInput(API.Utils.keyState(a.dx,a.dy,a.kick),1);room.runSteps(1);applied++;
 if(applied===8){clearTimeout(timeout);clearInterval(interval);console.log(JSON.stringify({ok:true,appliedInputs:applied,player:room.state.players[0].disc.pos,inferenceRoundTripMs:bridge.timings}));room.destroy();bridge.close();setTimeout(()=>process.exit(0),100);}
}});
bridge.initialize(JSON.parse(API.Utils.exportStadium(room.state.stadium)));
const interval=setInterval(()=>{tick++;const msg=snapshot(room,1,0,tick);bridge.request(msg)},30);
