'use strict';
const API = require('node-haxball')();
const room = API.Room.sandbox({}, {controlledPlayerId: 1});
try {
  room.setSimulationSpeed(0);
  room.playerJoin(1, 'probe-red', 'ar', null, '', '');
  room.playerJoin(2, 'probe-blue', 'ar', null, '', '');
  room.setPlayerTeam(1, 1, 0); room.setPlayerTeam(2, 2, 0);
  room.startGame(0); room.playerInput(API.Utils.keyState(1,0,false),1);
  for(let tick=0;tick<60;tick++)room.runSteps(1);
  const b=room.gameState.physicsState.discs[0];
  console.log(JSON.stringify({ok:true,stadium:room.state.stadium.name,players:room.state.players.map(p=>({id:p.id,team:p.team.id,pos:p.disc?.pos})),ball:{pos:b.pos,speed:b.speed},goalCount:room.state.stadium.goals.length}));
} finally {room.destroy();}
