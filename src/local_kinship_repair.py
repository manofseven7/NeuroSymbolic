"""NS-022: fixed local grounder repair; selected by old admission validation only."""
import csv,hashlib,itertools,json,re,time
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
import external_kinship as k
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/local_kinship_repair_v1'
def local_text(story,pair):
 a,b=pair
 ss=[s.strip() for s in re.split(r'[.!?]',story) if s.strip()]
 def both(s):return '['+a+']' in s and '['+b+']' in s
 hits=[s for s in ss if both(s)]
 if not hits:hits=[' '.join(ss[i:i+2]) for i in range(len(ss)-1) if both(' '.join(ss[i:i+2]))]
 if not hits:return 'NO_LOCAL_PAIR'
 s=min(hits,key=len)
 return re.sub(r'\[([^\]]+)\]',lambda m:' SUBJECT ' if m[1]==a else ' OBJECT ' if m[1]==b else ' PERSON ',s)
class LocalGrounder:
 def __init__(self,seed=42,mask_inverse=False):self.seed=seed;self.mask_inverse=mask_inverse
 def fit(self,rows):
  texts=[];labels=[]
  for r in rows:
   f,_=k.supervision(r)
   for a,b in itertools.permutations(k.names(r['story']),2):
    if self.mask_inverse and (a,b) not in f and (b,a) in f:continue
    texts.append(local_text(r['story'],(a,b)));labels.append(f.get((a,b),k.NONE))
  self.vec=k.features();X=self.vec.fit_transform(texts)
  self.clf=LogisticRegression(C=2,max_iter=300,class_weight='balanced',random_state=self.seed).fit(X,labels)
  return self
 def prepare(self,story,query):
  pairs=list(itertools.permutations(k.names(story),2));options=[]
  if pairs:
   probs=self.clf.predict_proba(self.vec.transform([local_text(story,p) for p in pairs]))
   for p in probs:
    order=np.argsort(-p);options.append([(str(self.clf.classes_[j]),float(p[j])) for j in order[:2]])
  return {'story':story,'query':tuple(query),'pairs':pairs,'options':options}
def read(name):return list(csv.DictReader((ROOT/'third_party/CLUTRR_external'/name).open(encoding='utf-8-sig')))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evaluate(model,rows,rules,seed,method):
 out=[]
 for r in rows:
  p=model.prepare(r['story'],k.parse_query(r['query']));f=k.selected_facts(p);ex=k.execute(f,p['query'],rules);ref,_=k.supervision(r)
  out.append({'id':r['id'],'seed':seed,'method':method,'hops':int(r['task_name'].split('.')[-1]),'correct':int(ex['preferred']==r['target_text']),'prediction':ex['preferred'],'gold':r['target_text'],'answered':int(bool(ex['relations'])),'ref_edges':len(ref),'recovered':sum(pair in f and f[pair][0]==rel for pair,rel in ref.items()),'invalid_proof':int(bool(ex['proof']) and not k.verify_proof(ex['proof'],f,rules)),'limited':int(ex['limited'])})
 return out
def summarize(rows):
 out={}
 for method in sorted({r['method'] for r in rows}):
  out[method]={}
  for group,lo,hi in [('all',2,10),('short',2,3),('long',4,10)]:
   rr=[r for r in rows if r['method']==method and lo<=r['hops']<=hi];n=len(rr)
   out[method][group]={'decisions':n,'unique_questions':len({r['id'] for r in rr}),'accuracy':sum(r['correct'] for r in rr)/n,'coverage':sum(r['answered'] for r in rr)/n,'reference_edge_recall':sum(r['recovered'] for r in rr)/sum(r['ref_edges'] for r in rr),'invalid_proofs':sum(r['invalid_proof'] for r in rr)}
 return out
def write_rows(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True)
 protocol={'experiment':'NS-LOCAL-GROUNDING-022','variants':['local','local_mask_inverse'],'selection':'maximum correct on archived128admissionvalidation; tie local; no test tuning','training':'same256programs/seed as NS020','seeds':[42,43,44],'C':2,'rule_set':'unchanged bundle','test_status':'old test seen during diagnosis; regression only','stop':'two fixed variants; then frozen confirmation; no further search','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 tr={r['id']:r for r in read('train.csv')};va={r['id']:r for r in read('validation.csv')};old=ROOT/'results/external_kinship_v1';frozen={}
 for seed in [42,43,44]:
  fr=json.loads((old/f'frozen_{seed}.json').read_text());rules=fr['bundle_rules'];scores={};models={}
  for name,mask in [('local',False),('local_mask_inverse',True)]:
   model=LocalGrounder(seed,mask).fit([tr[i] for i in fr['training_ids']]);models[name]=model;joblib.dump(model,OUT/f'{name}_{seed}.joblib')
   scores[name]=sum(x['correct'] for x in evaluate(model,[va[i] for i in fr['gate_ids']],rules,seed,name))
  selected=max(scores,key=scores.get);audit=evaluate(models[selected],[va[i] for i in fr['audit_ids']],rules,seed,selected)
  frozen[str(seed)]={'selected':selected,'gate_correct':scores,'audit_correct':sum(x['correct'] for x in audit),'gate_n':128,'audit_n':128,'model_sha256':digest(OUT/f'{selected}_{seed}.joblib'),'rules':rules}
  print('selected',seed,frozen[str(seed)],flush=True)
 (OUT/'FROZEN_SELECTION.json').write_text(json.dumps(frozen,indent=2))
 source={r['id']:r for r in read('test.csv')};ids=json.loads((old/'test_split_audit.json').read_text())['test_ids'];rows=[source[i] for i in ids];results=[]
 for seed in [42,43,44]:
  fr=frozen[str(seed)];model=joblib.load(OUT/f'{fr["selected"]}_{seed}.joblib');results+=evaluate(model,rows,fr['rules'],seed,'repaired')
  print('regression',seed,flush=True)
 write_rows(OUT/'predictions.csv',results);(OUT/'summary.json').write_text(json.dumps(summarize(results),indent=2))
if __name__=='__main__':main()
