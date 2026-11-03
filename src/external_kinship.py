"""Bounded, source-separated CLUTRR adapter. No evaluation annotation in predict().
Transfer concerns reused chain schema and certification contract, not learned
source neural weights or unrestricted real-world domain generalization.
"""
import ast,hashlib,itertools,json,re,math
from collections import Counter,defaultdict
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
NONE='__NO_DIRECT_EDGE__'

def names(story):return list(dict.fromkeys(re.findall(r'\[([^\]]+)\]',story)))
def pair_text(story,pair):
 a,b=pair
 def sub(m):return ' SUBJECT ' if m[1]==a else ' OBJECT ' if m[1]==b else ' PERSON '
 text=re.sub(r'\[([^\]]+)\]',sub,story)
 # Endpoint-conditioned local spans plus full delexicalized story.
 hits=[]
 for sent in re.split(r'[.!?]',text):
  if 'SUBJECT' in sent and 'OBJECT' in sent:hits.append(sent)
 return ' '.join(hits)+' LOCAL_END '+text

def parse_query(s):return tuple(ast.literal_eval(s))
def supervision(row):
 """Training/evaluation harness ONLY. Never called by predict or family()."""
 steps=ast.literal_eval(row['proof_state']);heads={tuple(h) for step in steps for h in step};leaves={tuple(b) for step in steps for bb in step.values() for b in bb}-heads
 facts={}
 for a,r,b in sorted(leaves):facts[a,b]=r
 rules=[]
 for step in steps:
  for h,body in step.items():
   if len(body)!=2:continue
   for b1,b2 in [body,body[::-1]]:
    if b1[0]==h[0] and b1[2]==b2[0] and b2[2]==h[2]:rules.append((b1[1],b2[1],h[1]));break
 return facts,rules

def features():return FeatureUnion([('w',TfidfVectorizer(ngram_range=(1,2),max_features=6000,sublinear_tf=True)),('c',TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=10000,sublinear_tf=True))])

class Grounder:
 def __init__(self,seed=42):self.seed=seed
 def fit(self,rows):
  texts=[];labels=[]
  for r in rows:
   f,_=supervision(r)
   for a,b in itertools.permutations(names(r['story']),2):texts.append(pair_text(r['story'],(a,b)));labels.append(f.get((a,b),NONE))
  self.vec=features();X=self.vec.fit_transform(texts);self.clf=LogisticRegression(C=2,max_iter=300,class_weight='balanced',random_state=self.seed).fit(X,labels)
  self.direct_vec=features();xx=[pair_text(r['story'],parse_query(r['query'])) for r in rows];self.direct_X=self.direct_vec.fit_transform(xx);self.direct_y=np.array([r['target_text'] for r in rows]);self.direct_clf=LogisticRegression(C=2,max_iter=300,class_weight='balanced',random_state=self.seed).fit(self.direct_X,self.direct_y);return self
 def prepare(self,story,query):
  pairs=list(itertools.permutations(names(story),2))
  if not pairs:return {'story':story,'query':tuple(query),'pairs':[],'options':[],'direct':'__ABSTAIN__','prototype':'__ABSTAIN__'}
  probs=self.clf.predict_proba(self.vec.transform([pair_text(story,p) for p in pairs]));options=[]
  for p in probs:
   order=np.argsort(-p);options.append([(str(self.clf.classes_[j]),float(p[j])) for j in order[:2]])
  X=self.direct_vec.transform([pair_text(story,query)]);direct=str(self.direct_clf.predict(X)[0]);proto=str(self.direct_y[cosine_similarity(X,self.direct_X).argmax()])
  return {'story':story,'query':tuple(query),'pairs':pairs,'options':options,'direct':direct,'prototype':proto}

def selected_facts(prepared,choices=None):
 choices=choices or {};facts={}
 for i,pair in enumerate(prepared['pairs']):
  rel,prob=prepared['options'][i][choices.get(i,0)]
  if rel!=NONE:facts[pair]=(rel,max(1e-9,prob),i)
 return facts

def execute(facts,query,rules,path_limit=64):
 """Enumerate simple paths, then all binary compositions along each path.
A limit blocks a certificate; a found derivation remains a valid support object.
"""
 a,b=query;adj=defaultdict(list)
 for key,(r,p,i) in facts.items():
  u,v=key[:2];adj[u].append((v,r,p,i))
 paths=[];limited=False
 def dfs(u,vertices,edges):
  nonlocal limited
  if limited:return
  if u==b and edges:
   if len(paths)>=path_limit:limited=True;return
   paths.append((vertices,edges));return
  for v,r,p,i in sorted(adj[u]):
   if v not in vertices:dfs(v,vertices+[v],edges+[(r,p,i)])
 dfs(a,[a],[])
 bypair=defaultdict(list)
 for ri,(r1,r2,r3) in enumerate(rules):bypair[r1,r2].append((r3,ri))
 found={}
 for vertices,edges in paths:
  n=len(edges);chart={}
  for i,(r,p,sid) in enumerate(edges):chart[i,i+1]={r:{'atom':[vertices[i],r,vertices[i+1]],'kind':'fact','pair_index':sid,'cost':-math.log(p),'support':[sid],'rule_ids':[]}}
  for length in range(2,n+1):
   for i in range(n-length+1):
    j=i+length;cell={}
    for k in range(i+1,j):
     for r1,p1 in chart[i,k].items():
      for r2,p2 in chart[k,j].items():
       for r3,ri in bypair[r1,r2]:
        cost=p1['cost']+p2['cost']+.1
        if r3 not in cell or cost<cell[r3]['cost']:cell[r3]={'atom':[vertices[i],r3,vertices[j]],'kind':'rule','rule_id':ri,'parents':[p1,p2],'cost':cost,'support':sorted(set(p1['support']+p2['support'])),'rule_ids':sorted(set(p1['rule_ids']+p2['rule_ids']+[ri]))}
    chart[i,j]=cell
  for rel,pf in chart[0,n].items():
   if rel not in found or pf['cost']<found[rel]['cost']:found[rel]=pf
 preferred=min(found,key=lambda r:(found[r]['cost'],r)) if found else '__ABSTAIN__'
 return {'relations':sorted(found),'preferred':preferred,'proof':found.get(preferred),'limited':limited}

def verify_proof(proof,facts,rules):
 if not proof:return False
 a,r,b=proof['atom']
 if proof['kind']=='fact':return (a,b) in facts and facts[a,b][0]==r and facts[a,b][2]==proof['pair_index']
 ri=proof['rule_id']
 if not 0<=ri<len(rules):return False
 p,q=proof['parents'];x,r1,y=p['atom'];yy,r2,z=q['atom']
 return x==a and z==b and y==yy and tuple(rules[ri])==(r1,r2,r) and verify_proof(p,facts,rules) and verify_proof(q,facts,rules)

def family(prepared,rules,width=4):
 facts=selected_facts(prepared);preferred=execute(facts,prepared['query'],rules);opts=prepared['options'];vary=sorted([i for i in range(len(opts)) if opts[i][0][0]!=opts[i][1][0]],key=lambda i:(opts[i][0][1]-opts[i][1][1],i))[:width]
 # Intersection/union path semantics: relation alternatives are independent
 # edge facts in upper; common singleton edge choices are lower.
 # Upper accepts multiple labels per pair; positive simple-path closure is
 # monotone. Equal singleton lower/upper query sets certify every program.
 # Otherwise exact retained-family fallback preserves the decision.
 lower={};upper={}
 for i,pair in enumerate(prepared['pairs']):
  choices=opts[i] if i in vary else [opts[i][0]]
  for rel,prob in choices:
   if rel!=NONE:upper[tuple(pair)+(rel,)]=(rel,max(1e-9,prob),i)
  if i not in vary and choices[0][0]!=NONE:lower[pair]=(choices[0][0],max(1e-9,choices[0][1]),i)
 lo=execute(lower,prepared['query'],rules);hi=execute(upper,prepared['query'],rules)
 if not lo['limited'] and not hi['limited'] and len(lo['relations'])==1 and lo['relations']==hi['relations']:
  return {'preferred':preferred['preferred'],'proof':preferred['proof'],'facts':facts,'certified':True,'certified_answer':lo['relations'][0],'family_size':2**len(vary),'checked_programs':0,'certificate_kind':'must_may','limited':False,'vary':vary,'sets':[tuple(lo['relations'])]}
 outcomes=[];limited=False;sets=[]
 for bits in itertools.product([0,1],repeat=len(vary)):
  ex=execute(selected_facts(prepared,dict(zip(vary,bits))),prepared['query'],rules);limited|=ex['limited'];sets.append(tuple(ex['relations']));outcomes.append(ex)
  if limited or len(sets[-1])!=1 or len(set(sets))>1:break
 certified=not limited and bool(sets) and len(set(sets))==1 and len(sets[0])==1
 return {'preferred':preferred['preferred'],'proof':preferred['proof'],'facts':facts,'certified':certified,'certified_answer':sets[0][0] if certified else '__ABSTAIN__','family_size':2**len(vary),'checked_programs':len(sets),'certificate_kind':'exact_fallback','limited':limited,'vary':vary,'sets':sets}

def correct(prepared,rules,oracle,budget=2,policy='uncertainty',seed=42,stop_on_certificate=True):
 """Harness passes a semantic pair oracle. It returns directly asserted edge
 semantics or NO_DIRECT, never the requested task-answer relation.
 """
 p={**prepared,'options':[[*o] for o in prepared['options']]};trace=[];rng=np.random.default_rng(seed)
 got=family(p,rules)
 for _ in range(budget):
  if stop_on_certificate and got['certified']:break
  allowed=[i for i in got['vary'] if i not in [x['pair_index'] for x in trace]]
  if not allowed:break
  i=min(allowed,key=lambda j:(p['options'][j][0][1]-p['options'][j][1][1],j)) if policy=='uncertainty' else int(rng.choice(allowed));pair=p['pairs'][i];rel=oracle(pair);p['options'][i]=[(rel,1.),(rel,1.)];trace.append({'pair_index':i,'pair':pair,'semantic_label':rel});got=family(p,rules)
 got['annotations']=trace;return got

GENDER={r:'m' for r in ['son','son-in-law','father','father-in-law','brother','grandfather','uncle','nephew','grandson','husband']}
GENDER.update({r:'f' for r in ['daughter','daughter-in-law','mother','mother-in-law','sister','sister-in-law','grandmother','aunt','niece','granddaughter','wife']})
def story_fingerprint(story):return re.sub(r'\[[^\]]+\]','[ENTITY]',story).lower().strip()
def valid_schema(rule):
 # Constructor fixes connected, range-restricted x-y-z shape; target-gender
 # constraint and functional composition reject incompatible instantiations.
 return len(rule)==3 and all(isinstance(r,str) and r in GENDER for r in rule) and GENDER[rule[1]]==GENDER[rule[2]]
def admit_rules(prepared,labels,candidates):
 accepted=[];log=[]
 def score(rs):return sum(execute(selected_facts(p),p['query'],rs)['preferred']==y for p,y in zip(prepared,labels))
 base=score(accepted)
 for rule,count in candidates:
  valid=valid_schema(rule) and not any(r[:2]==rule[:2] and r[2]!=rule[2] for r in accepted)
  trial=score(accepted+[rule]) if valid else base
  # Logical structure/range/join checked by construction; validation resolves
  # conflicting consequences. This is not an external truth verifier.
  accept=valid and trial>base
  log.append({'rule':rule,'training_support':count,'schema_valid':valid,'before_correct':base,'trial_correct':trial,'admitted':accept})
  if accept:accepted.append(rule);base=trial
 return accepted,log

def proof_interventions(got,query,rules):
 pf=got['proof'];facts=got['facts']
 if not pf:return None
 rel=got['preferred'];sup=set(pf['support']);restricted={p:v for p,v in facts.items() if v[2] in sup};removed={p:v for p,v in facts.items() if v[2] not in sup}
 sufficient=rel in execute(restricted,query,rules)['relations'];necessary=rel not in execute(removed,query,rules)['relations']
 deletions=[]
 for ri in pf['rule_ids']:
  subset=[r for j,r in enumerate(rules) if j!=ri];deletions.append({'rule_id':ri,'removes_conclusion':rel not in execute(facts,query,subset)['relations']})
 return {'proof_valid':verify_proof(pf,facts,rules),'support_sufficient':sufficient,'support_set_deletion_removes':necessary,'individual_rule_deletions':deletions}


def admit_rule_bundle(prepared,labels,candidates):
 """Single validation-diagnosed repair before test: admit a coherent bundle.
 Preserves complementary rules with no individually immediate payoff.
 """
 coherent=[];rejected=[]
 for rule,count in candidates:
  valid=valid_schema(rule) and not any(r[:2]==rule[:2] and r[2]!=rule[2] for r in coherent)
  if valid:coherent.append(rule)
  else:rejected.append(rule)
 def score(rs):return sum(execute(selected_facts(p),p['query'],rs)['preferred']==y for p,y in zip(prepared,labels))
 base=score([]);trial=score(coherent);accepted=coherent if trial>base else []
 return accepted,{'baseline_correct':base,'bundle_correct':trial,'candidate_count':len(candidates),'coherent_count':len(coherent),'rejected_rules':rejected,'admitted':trial>base,'validation_n':len(labels)}
