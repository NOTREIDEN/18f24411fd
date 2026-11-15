'use strict';
// A local JavaScript Action receives artifact runtime credentials without ever
// writing them to logs or checkpoints. Python remains the independent learner.
const fs=require('node:fs');const path=require('node:path');const {spawn,execFileSync}=require('node:child_process');
const readline=require('node:readline');const {DefaultArtifactClient}=require('@actions/artifact');
const root=path.join(process.env.GITHUB_WORKSPACE||path.resolve(__dirname,'../..'),'haxball');
process.chdir(root);
const seed=Number(process.env.HAXBALL_SEED||0),run=process.env.GITHUB_RUN_ID,repo=process.env.GITHUB_REPOSITORY;
const artifact=new DefaultArtifactClient();let queue=Promise.resolve(),lastPeer=0,lastImport=0;
const out=path.join(root,'runs/island');fs.mkdirSync(path.join(out,'peers'),{recursive:true});
function gh(args){return execFileSync('gh',args,{encoding:'utf8',maxBuffer:8*1024*1024,env:process.env,timeout:90000});}
function listArtifacts(which=run){return JSON.parse(gh(['api','--paginate','--slurp',`repos/${repo}/actions/runs/${which}/artifacts?per_page=100`])).flatMap(p=>p.artifacts||[]);}
async function upload(name,files){
 files=files.filter(f=>fs.existsSync(f));if(!files.length)return;
 const result=await artifact.uploadArtifact(name,files,root,{retentionDays:1,compressionLevel:6});
 console.log(JSON.stringify({event:'artifact',name,id:result.id,url:`https://github.com/${repo}/actions/runs/${run}/artifacts/${result.id}`}));return result;
}
async function publishPeer(record){
 const source=path.join(out,'champion.pt');if(!fs.existsSync(source))return;
 // Copy an immutable upload snapshot while the learner continues atomically saving.
 const dir=path.join(root,`reports/peer-${seed}`);fs.mkdirSync(dir,{recursive:true});
 const copy=path.join(dir,'champion.pt');fs.copyFileSync(source,copy);
 fs.writeFileSync(path.join(dir,'metadata.json'),JSON.stringify({seed,run,iteration:record.iteration,samples:record.samples}));
 const prefix=`peer-${seed}-`;
 await upload(prefix+record.iteration,[copy,path.join(dir,'metadata.json')]);
 const old=listArtifacts().filter(a=>a.name.startsWith(prefix)).sort((a,b)=>b.id-a.id);
 for(const a of old.slice(2))gh(['api','--method','DELETE',`repos/${repo}/actions/artifacts/${a.id}`]);
}
async function importPeer(){
 const choices=listArtifacts().filter(a=>a.name.startsWith('peer-')&&!a.name.startsWith(`peer-${seed}-`));
 if(!choices.length)return;
 const pick=choices[Math.floor(Math.random()*Math.min(choices.length,30))];
 const temp=path.join(out,'peer-download');fs.rmSync(temp,{recursive:true,force:true});fs.mkdirSync(temp,{recursive:true});
 gh(['run','download',run,'--repo',repo,'--name',pick.name,'--dir',temp]);
 function find(dir){for(const d of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,d.name);if(d.isDirectory()){const q=find(p);if(q)return q;}else if(d.name==='champion.pt')return p;}}
 const cp=find(temp);if(cp){const destination=path.join(out,'peers',pick.name+'.pt');fs.copyFileSync(cp,destination);const peers=fs.readdirSync(path.dirname(destination)).filter(x=>x.endsWith('.pt'));for(const file of peers.slice(0,Math.max(0,peers.length-12)))fs.unlinkSync(path.join(out,'peers',file));}
}
function safeTask(fn){queue=queue.then(fn).catch(e=>console.log(JSON.stringify({event:'artifact-warning',message:String(e.message).slice(0,500)})));}
async function main(){
 let initial=path.join(root,'assets/bootstrap.pt'),resume=false;
 if(process.env.HAXBALL_PREVIOUS_RUN){
  const previous=process.env.HAXBALL_PREVIOUS_RUN,temp=path.join(root,'runs/resume');
  fs.mkdirSync(temp,{recursive:true});gh(['run','download',previous,'--repo',repo,'--name',`final-island-${seed}`,'--dir',temp]);
  function locate(dir,name){for(const x of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,x.name);if(x.isDirectory()){const hit=locate(p,name);if(hit)return hit;}else if(x.name===name)return p;}}
  const latest=locate(temp,'latest.pt'),champion=locate(temp,'champion.pt');
  if(latest){initial=latest;resume=true;}if(champion)fs.copyFileSync(champion,path.join(out,'champion.pt'));
 }
 const remaining=Math.floor((Date.parse(process.env.HAXBALL_DEADLINE)-Date.now())/1000)-600;
 const seconds=Math.max(0,Math.min(Number(process.env.HAXBALL_DURATION||7200),remaining));
 if(seconds<60)throw new Error('task deadline leaves insufficient training time');
 if(!resume){
  const warm=path.join(root,'runs/futsal-warmstart.pt');
  const boot=spawn('python',['scripts/bootstrap.py','--initial',initial,'--seconds','120','--output',warm,'--seed',String(seed),'--envs','64'],{cwd:root,env:process.env,stdio:'inherit'});
  const result=await new Promise(resolve=>boot.on('exit',resolve));if(result!==0)throw new Error('futsal warmstart failed '+result);initial=warm;
 }
 const envs=[64,128,256][seed%3],threads=[1,2,4][seed%3],lr=[.0001,.0002,.0003][Math.floor(seed/3)%3];
 const args=['scripts/train.py','--initial',initial,'--output',out,'--seconds',String(seconds),'--seed',String(seed),'--envs',String(envs),'--threads',String(threads),'--horizon','128','--sequence','16','--lr',String(lr),'--eval-envs','12','--eval-seconds','30','--eval-interval','180','--role',seed>=16?'exploiter':'main'];
 if(resume)args.push('--resume');
 console.log(JSON.stringify({event:'island-start',seed,seconds,envs,threads,lr,owner:repo.split('/')[0],resume}));
 const child=spawn('python',args,{cwd:root,env:process.env,stdio:['ignore','pipe','inherit']});
 readline.createInterface({input:child.stdout}).on('line',line=>{
  console.log(line);let record;try{record=JSON.parse(line)}catch{return;}
  if(record.event==='checkpoint'&&Date.now()-lastPeer>600000){lastPeer=Date.now();safeTask(()=>publishPeer(record));}
  if(record.event==='checkpoint'&&Date.now()-lastImport>600000){lastImport=Date.now();safeTask(importPeer);}
  if(record.event==='video'){
   const file=path.resolve(root,record.path);safeTask(()=>upload(`video-${seed}-${record.iteration}`,[file,file.replace(/\.mp4$/,'.json')]));
  }
 });
 process.on('SIGTERM',()=>child.kill('SIGTERM'));process.on('SIGINT',()=>child.kill('SIGINT'));
 const exit=await new Promise(resolve=>child.on('exit',(code)=>resolve(code)));
 await queue;
 const reports=fs.existsSync(out)?fs.readdirSync(out).filter(x=>x.endsWith('.json')).map(x=>path.join(out,x)):[];
 await upload(`final-island-${seed}`,[path.join(out,'champion.pt'),path.join(out,'latest.pt'),path.join(out,'training.jsonl'),...reports]);
 if(exit!==0)throw new Error('learner exited '+exit);
}
main().catch(e=>{console.error(e.stack||e);process.exitCode=1;});
