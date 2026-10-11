from pathlib import Path
import os,subprocess,json,hashlib
c=Path('/home/sagan/.cache/bevy-gym-quality-validation');q=Path('/home/sagan/Code/github.com/sagan-software/bevy-gym-quality');out=c/'droid-parallel-fresh-coverage-v3';out.mkdir(exist_ok=False)
llvm=Path('/nix/store/llz63b99xvpsk2bhrzbh7z1grxxfa4h9-rust-mixed/lib/rustlib/x86_64-unknown-linux-gnu/bin')
candidates=[x for x in (c/'coverage-target/debug/deps').glob('droid_parallel_rollout-*') if x.is_file() and x.suffix=='' and os.access(x,os.X_OK)]
objects=[('tests',max(candidates,key=lambda x:x.stat().st_mtime)),('cli',c/'coverage-target/debug/examples/droid-standing')]
summary={'objects':[]};regions={};branches={}
for label,obj in objects:
 args=[] if label=='tests' else ['train','--updates','1','--seed','281','--reward-profile','posture-height-v1','--optimizer-profile','conservative-v1','--output',str(c/'droid-parallel-cli-smoke-fresh-v3')]
 with (out/f'{label}.log').open('w') as log:run=subprocess.run([str(obj),*args],cwd=q,env={**os.environ,'LLVM_PROFILE_FILE':str(out/f'{label}-%p-%m.profraw')},stdout=log,stderr=subprocess.STDOUT)
 assert run.returncode==(0 if label=='tests' else 1)
 if label=='cli':assert 'Standing exhausted its update budget without passing' in (out/'cli.log').read_text()
 subprocess.run([str(llvm/'llvm-profdata'),'merge','-sparse',*map(str,out.glob(f'{label}-*.profraw')),'-o',str(out/f'{label}.profdata')],check=True)
 with (out/f'{label}.json').open('w') as data,(out/f'{label}.stderr').open('w') as err:subprocess.run([str(llvm/'llvm-cov'),'export',str(obj),'-instr-profile='+str(out/f'{label}.profdata')],stdout=data,stderr=err,check=True)
 assert not (out/f'{label}.stderr').read_text()
 summary['objects'].append({'path':str(obj),'sha256':hashlib.sha256(obj.read_bytes()).hexdigest()})
 for f in json.loads((out/f'{label}.json').read_text())['data'][0]['functions']:
  if '5tests' in f['name'] or '::tests::' in f['name']:continue
  for r in f['regions']:
   path=str((q/f['filenames'][r[5]]).resolve());rel=str(Path(path).relative_to(q)) if Path(path).is_relative_to(q) else ''
   selected=(rel=='examples/robots/standing/parallel_rollout.rs' or (rel=='examples/robots/learning/rollout.rs' and any(name in f['name'] for name in ['collect_lane','collectors','collect_into','with_decoder','7collect','12with_actions'])) or (rel=='examples/robots/standing/mod.rs' and 'parallel_batch_with_recipe' in f['name']))
   if not selected:continue
   key=(rel,*r[:4]);regions[key]=max(regions.get(key,0),r[4])
  for r in f['branches']:
   path=str((q/f['filenames'][r[6]]).resolve());rel=str(Path(path).relative_to(q)) if Path(path).is_relative_to(q) else ''
   if rel not in ['examples/robots/learning/rollout.rs','examples/robots/standing/parallel_rollout.rs']:continue
   key=(rel,*r[:4]);old=branches.get(key,(0,0));branches[key]=(max(old[0],r[4]),max(old[1],r[5]))
for rel in sorted({k[0] for k in regions}):
 rr={k:v for k,v in regions.items() if k[0]==rel};summary[rel]={'hit':sum(v>0 for v in rr.values()),'total':len(rr),'zero_regions':[list(k[1:]) for k,v in rr.items() if not v],'branches':[{'range':list(k[1:]),'hits':list(v)} for k,v in branches.items() if k[0]==rel],'source_sha256':hashlib.sha256((q/rel).read_bytes()).hexdigest()}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='objects'},indent=2))
