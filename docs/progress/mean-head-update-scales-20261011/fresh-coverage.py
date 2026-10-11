from pathlib import Path
import os, subprocess, json, hashlib
c=Path('/home/sagan/.cache/bevy-gym-quality-validation');q=Path('/home/sagan/Code/github.com/sagan-software/bevy-gym-quality');out=c/'mean-head-fresh-coverage-v3-source-match';out.mkdir(exist_ok=False)
llvm=Path('/nix/store/llz63b99xvpsk2bhrzbh7z1grxxfa4h9-rust-mixed/lib/rustlib/x86_64-unknown-linux-gnu/bin')
source=q/'src/training/recurrent_ppo/mod.rs';lines=source.read_text().splitlines();start=next(i+1 for i,x in enumerate(lines) if 'let previous_head = self' in x);end=next(i+1 for i,x in enumerate(lines[start:],start) if 'self.project_exploration();' in x)
objects=[]
for label,pattern,filter_name in [('private','bevy_gym-*','action_mean_step_scaling'),('public','recurrent_ppo_bytes-*','action_mean_update_scales')]:
 candidates=[p for p in (c/'coverage-target/debug/deps').glob(pattern) if p.is_file() and p.suffix=='' and os.access(p,os.X_OK)]
 objects.append((label,max(candidates,key=lambda p:p.stat().st_mtime),filter_name))
summary={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'objects':[],'groups':{}};regions={};branches={}
for label,obj,filter_name in objects:
 with (out/f'{label}.log').open('w') as log:
  subprocess.run([str(obj),filter_name,'--test-threads=1'],cwd=q,env={**os.environ,'LLVM_PROFILE_FILE':str(out/f'{label}-%p-%m.profraw')},stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([str(llvm/'llvm-profdata'),'merge','-sparse',*map(str,out.glob(f'{label}-*.profraw')),'-o',str(out/f'{label}.profdata')],check=True)
 with (out/f'{label}.json').open('w') as data,(out/f'{label}.stderr').open('w') as err:
  subprocess.run([str(llvm/'llvm-cov'),'export',str(obj),'-instr-profile='+str(out/f'{label}.profdata')],stdout=data,stderr=err,check=True)
 assert not (out/f'{label}.stderr').read_text()
 summary['objects'].append({'path':str(obj),'sha256':hashlib.sha256(obj.read_bytes()).hexdigest()})
 for f in json.loads((out/f'{label}.json').read_text())['data'][0]['functions']:
  if '5tests' in f['name'] or '::tests::' in f['name']:continue
  name=next((n for n in ['set_action_mean_update_scales','scale_action_mean_step'] if n in f['name']),None)
  for r in f['regions']:
   if (q/f['filenames'][r[5]]).resolve()!=source:continue
   group=name or ('update_hook' if start<=r[0]<=r[2]<=end else None)
   if not group:continue
   key=(group,*r[:4]);regions[key]=max(regions.get(key,0),r[4])
  for r in f['branches']:
   if (q/f['filenames'][r[6]]).resolve()!=source:continue
   group=name or ('update_hook' if start<=r[0]<=r[2]<=end else None)
   if not group:continue
   key=(group,*r[:4]);old=branches.get(key,(0,0));branches[key]=(max(old[0],r[4]),max(old[1],r[5]))
for group in sorted({k[0] for k in regions}):
 rr={k:v for k,v in regions.items() if k[0]==group}
 summary['groups'][group]={'hit':sum(v>0 for v in rr.values()),'total':len(rr),'zero_regions':[list(k[1:]) for k,v in rr.items() if not v],'branches':[{'range':list(k[1:]),'hits':list(v)} for k,v in branches.items() if k[0]==group]}
assert summary['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['groups'],indent=2))
