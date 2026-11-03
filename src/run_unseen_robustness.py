import sys,pathlib,json,csv,hashlib,collections,time,joblib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import external_kinship as k
from run_consistent_kinship import eval_rows
from local_kinship_repair import evaluate
P=ROOT/'results/unseen_robustness_v1';O=P/'reproduction';O.mkdir(exist_ok=True)
h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
prior=[]
for p in (ROOT/'third_party/CLUTRR_external').glob('*.csv'):prior.extend(csv.DictReader(p.open(encoding='utf-8-sig')))
ids={r['id'] for r in prior};fps={k.story_fingerprint(r['story']) for r in prior};seen=set();groups=collections.defaultdict(list);exc=collections.Counter()
raw=list(csv.DictReader((P/'robustness_raw.csv').open(encoding='utf-8-sig')))
for r in raw:
 f=k.story_fingerprint(r['story'])
 if r['id'] in ids:exc['prior_id']+=1;continue
 if f in fps:exc['prior_fingerprint']+=1;continue
 if f in seen:exc['within_duplicate']+=1;continue
 seen.add(f);groups[r['task_name']].append(r)
selected=[]
for task,rr in sorted(groups.items()):selected.extend(sorted(rr,key=lambda r:hashlib.sha256(('NS026|'+k.story_fingerprint(r['story'])).encode()).hexdigest())[:40])
audit={'raw':len(raw),'excluded':dict(exc),'eligible':{g:len(v) for g,v in groups.items()},'selected':dict(collections.Counter(r['task_name'] for r in selected)),'n':len(selected),'prior_source_rows':len(prior),'source_sha256':h(P/'robustness_raw.csv'),'selected_ids':[r['id'] for r in selected]}
(O/'selection_audit.json').write_text(json.dumps(audit,indent=2));(O/'heldout.json').write_text(json.dumps(selected));print('PRE_SCORE_SELECTION',json.dumps(audit),flush=True)
assert selected
fr=json.loads((ROOT/'results/consistent_kinship_v1/FROZEN_SELECTION.json').read_text());old=json.loads((ROOT/'results/local_kinship_repair_v1/FROZEN_SELECTION.json').read_text());pred=[];hashes={};start=time.time()
for seed in [42,43,44]:
 f=fr[str(seed)];p=ROOT/f'results/consistent_kinship_v1/model_{seed}.joblib';hashes[str(p.relative_to(ROOT))]=h(p);assert h(p)==f['model_sha256'];m=joblib.load(p)
 pred.extend(eval_rows(m,selected,f['rules'],f['reciprocal'],seed,'NS023'))
 f=old[str(seed)];p=ROOT/f'results/local_kinship_repair_v1/{f["selected"]}_{seed}.joblib';hashes[str(p.relative_to(ROOT))]=h(p);assert h(p)==f['model_sha256'];m=joblib.load(p);pred.extend(evaluate(m,selected,f['rules'],seed,'NS022'))
 p=ROOT/f'results/external_kinship_v1/model_{seed}.joblib';hashes[str(p.relative_to(ROOT))]=h(p);m=joblib.load(p)
 for r in selected:
  a=m.prepare(r['story'],k.parse_query(r['query']))['direct'];pred.append({'id':r['id'],'seed':seed,'method':'direct','hops':int(r['task_name'].split('.')[-1]),'correct':int(a==r['target_text']),'prediction':a,'gold':r['target_text'],'answered':1,'ref_edges':0,'recovered':0,'invalid_proof':0,'limited':0})
 (O/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred));print('SEED_DONE',seed,'seconds',time.time()-start,flush=True)
byid={r['id']:r for r in selected};summary={'selection':audit,'model_hashes':hashes,'runtime_seconds':time.time()-start,'metrics':[],'paired':[],'no_retraining':True,'scope':'fresh same-corpus robustness,not independent real-world domain or long5to10 confirmation'}
for method in ['NS023','NS022','direct']:
 for task in ['all']+sorted(groups):
  rr=[x for x in pred if x['method']==method and (task=='all' or byid[x['id']]['task_name']==task)]
  if rr:summary['metrics'].append({'method':method,'task':task,'n':len(rr),'accuracy_percent':100*np.mean([x['correct'] for x in rr]),'invalid_proofs':sum(x['invalid_proof'] for x in rr)})
pm={(x['method'],x['seed'],x['id']):x for x in pred}
for b in ['NS022','direct']:
 d=np.array([np.mean([pm['NS023',s,r['id']]['correct']-pm[b,s,r['id']]['correct'] for s in [42,43,44]]) for r in selected]);rng=np.random.default_rng(26026);v=np.concatenate([rng.choice(d,(1000,len(d)),replace=True).mean(1) for _ in range(10)])*100
 summary['paired'].append({'baseline':b,'delta_pp':float(d.mean()*100),'ci95':list(np.quantile(v,[.025,.975]))})
(O/'summary.json').write_text(json.dumps(summary,indent=2));print('FINAL_SUMMARY',json.dumps(summary,indent=2),flush=True)
