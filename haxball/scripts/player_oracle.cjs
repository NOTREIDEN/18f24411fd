'use strict';
// Differential trajectories from the actual JS engine, including player input.
const {Room,Utils}=require('node-haxball')();
let seed=92717;const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
const scenarios=[['Classic',1,100,2],['Big',3,100,6],['Huge',10,80,2],['Rounded',3,100,2]];
scenarios.push(['Asymmetric validation',3,120,2]);
const result=[];
if(require('node:fs').existsSync('assets/maps/futsal-bff-v3.hbs'))scenarios.push(['Futsal BFF v3',3,120,2]);
for(const size of [1,2,3,12])scenarios.push(['Futsal profile '+size,size,size>3?80:120,2]);
for(const [name,size,ticks,minKick] of scenarios){
 const room=Room.sandbox({},{});room.setSimulationSpeed(0);
 try{
  let stadium;
  if(name.startsWith('Futsal profile '))stadium=Utils.parseStadium(require('node:fs').readFileSync(`assets/maps/futsal-${size}v${size}${size>3?'-training':''}.hbs`,'utf8'));
  else if(name==='Futsal BFF v3')stadium=Utils.parseStadium(require('node:fs').readFileSync('assets/maps/futsal-bff-v3.hbs','utf8'));
  else if(name==='Asymmetric validation'){
   const raw=JSON.parse(Utils.exportStadium(Utils.getDefaultStadiums()[0]));raw.name=name;
   for(const pl of raw.planes)if(pl.cMask?.includes('ball')&&pl.normal[1]===1)pl.dist=-135;
   raw.discs.push({pos:[15,55],radius:9,invMass:0,bCoef:.7});
   stadium=Utils.parseStadium(JSON.stringify(raw));
  }else stadium=Utils.getDefaultStadiums().find(x=>x.name===name);
  room.setCurrentStadium(stadium,0);
  for(let k=0;k<2*size;k++){room.playerJoin(k+1,'p'+k,'',null,'','');room.setPlayerTeam(k+1,k<size?1:2,0);}
  if(room.setKickRateLimit)room.setKickRateLimit(minKick,0,0,0);
  room.startGame(0);room.setDiscProperties(0,0,{xspeed:1},0);
  // In this API runSteps(x) advances one step with multiplier x. Repeated 1s
  // are essential; a single 2 leaves kickoff collision masks active.
  for(let settle=0;settle<3;settle++)room.runSteps(1);
  const pp=[],pv=[];
  for(let k=0;k<2*size;k++){
   const pos=[(k%5-2)*44+rand()*5,(Math.floor(k/5)-Math.floor(size/5)/2)*42-42];
   const vel=[(rand()-.5)*1.2,(rand()-.5)*1.2];pp.push(pos);pv.push(vel);
   room.setDiscProperties(k+1,1,{x:pos[0],y:pos[1],xspeed:vel[0],yspeed:vel[1]},0);
  }
  // Ball starts near a player so that contact and kick paths are exercised.
  const bp=[pp[0][0]+26,pp[0][1]],bv=[.2,-.1];
  room.setDiscProperties(0,0,{x:bp[0],y:bp[1],xspeed:bv[0],yspeed:bv[1]},0);
  // Network event setters may quantize values. Compare from the actual state,
  // rather than pretending the requested decimal survived serialization.
  const initialBall=room.gameState.physicsState.discs[0];
  const actualInitial={bp:[initialBall.pos.x,initialBall.pos.y],bv:[initialBall.speed.x,initialBall.speed.y],pp:room.state.players.map(p=>[p.disc.pos.x,p.disc.pos.y]),pv:room.state.players.map(p=>[p.disc.speed.x,p.disc.speed.y]),masks:room.state.players.map(p=>p.disc.cMask)};
  const actions=[],trajectory=[];
  for(let t=0;t<ticks;t++){
   const a=[];
   for(let k=0;k<2*size;k++){
    const b=room.gameState.physicsState.discs[0],d=room.state.players[k].disc;
    const dx=t<25?Math.sign(b.pos.x-d.pos.x):Math.floor(rand()*3)-1;
    const dy=t<25?Math.sign(b.pos.y-d.pos.y):Math.floor(rand()*3)-1;
    const kick=(t%4===0||rand()<.15)?1:0;a.push([dx,dy,kick]);
    room.playerInput(Utils.keyState(dx,dy,!!kick),k+1);
   }
   room.runSteps(1);actions.push(a);
   const ball=room.gameState.physicsState.discs[0];
   trajectory.push({bp:[ball.pos.x,ball.pos.y],bv:[ball.speed.x,ball.speed.y],pp:room.state.players.map(p=>[p.disc.pos.x,p.disc.pos.y]),pv:room.state.players.map(p=>[p.disc.speed.x,p.disc.speed.y])});
  }
  result.push({name,size,minKick,stadium:JSON.parse(Utils.exportStadium(stadium)),initial:actualInitial,actions,trajectory});
 }finally{room.destroy();}
}
console.log(JSON.stringify(result));
