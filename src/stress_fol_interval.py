"""Frozen-code transfer to larger operator families and fresh logical cases.

Synthetic cases are computational diagnostics, not natural-language benchmarks.
"""
import csv,hashlib,json,random,time
from pathlib import Path
from collections import Counter
from fol_certifier import parse,ambiguity_paths
from fol_model_reuse import run as baseline
from fol_interval_certifier import run as interval
ROOT=Path(__file__).resolve().parents[1]
METHODS=['incremental_early_stop','interval_gate','selector_only','selector_model_reuse','incremental_model_reuse','branch_model_reuse']
def build():
 data=[json.loads(x) for x in (ROOT/'third_party/FOLIO_curated/dataset_rawpairs.jsonl').read_text().splitlines()];stories={x['id'].split('_')[1]:x for x in data if x['id'].startswith('story_')};cases=[]
 for x in data:
  if not x['id'].startswith('concl_'):continue
  k=stories[x['id'].split('_')[1]]['FOL_sentence_new'];q=x['FOL_sentence_new'];n=len(ambiguity_paths(parse(k)))+len(ambiguity_paths(parse(q)))
  if n>4:cases.append({'id':x['id'],'source':'curated_larger_family','theory':k,'question':q,'max_sites':7})
 for seed in range(16000,16032):
  rng=random.Random(seed);atoms=[f'U{seed}_{i}(a)' for i in range(8)];clauses=[]
  for j in range(6):
   a,b=rng.sample(atoms,2);a=('¬' if rng.random()<.25 else '')+a;b=('¬' if rng.random()<.25 else '')+b;clauses.append(f'({a} ∨ {b})')
  if rng.random()<.25:clauses.append(rng.choice(atoms))
  cases.append({'id':f'synthetic_{seed}','source':'fresh_synthetic','theory':' ∧ '.join(clauses),'question':rng.choice(atoms),'max_sites':6})
 return cases

def main():
 out=ROOT/'results/folio_interval_stress';out.mkdir(exist_ok=True);cases=build();(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
 protocol={'experiment':'NS-FOL-STRESS-016','before_outcome':False,'revision':'Add stronger model-reuse controls after initial stress outcomes; algorithm and cases unchanged; initial results retained in history/folio_stress_initial','synthetic_seeds':[16000,16031],'synthetic_n':32,'methods':METHODS,'repetitions':3,'timeout_ms':1500,'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/fol_interval_certifier.py','src/fol_model_reuse.py','src/stress_fol_interval.py']},'scope':'larger families of previously seen curated questions + fresh synthetic computational diagnostics; neither is an independent end-to-end NLU corpus'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
 for rep in range(3):
  for i,case in enumerate(cases):
   k,q=parse(case['theory']),parse(case['question']);labels=[];shift=(rep+i)%len(METHODS)
   for method in METHODS[shift:]+METHODS[:shift]:
    r=interval(k,q,max_sites=case['max_sites']) if method=='interval_gate' else baseline(k,q,method,max_sites=case['max_sites']);r.pop('reuse_evidence');r.pop('obligations',None);r.setdefault('certificate_kind','exact_or_selector');labels.append(r['label']);rows.append({'id':case['id'],'source':case['source'],'rep':rep,'method':method,**r})
   if len(set(labels))!=1:raise AssertionError((case['id'],labels))
   if (i+1)%10==0:print('rep',rep,'cases',i+1,flush=True)
 with (out/'runs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={}
 for source in sorted({r['source'] for r in rows}):
  result[source]={}
  for method in METHODS:
   rr=[r for r in rows if r['source']==source and r['method']==method];result[source][method]={'runs':len(rr),'mean_ms':sum(r['runtime_seconds'] for r in rr)/len(rr)*1000,'mean_solver_checks':sum(r['solver_checks'] for r in rr)/len(rr),'certificate_disagreements':0,'labels':dict(Counter(r['label'] for r in rr))}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
