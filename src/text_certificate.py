"""Frozen neural text front end + must/may certificate over retained programs.

Horn specialization of the formal-core contract, not unrestricted NL-to-FOL.
Sign is a predicate component: explicit negative facts do not introduce classical
contraposition. Program uncertainty abstention is distinct from logical unknown.
"""
import hashlib,json
import learned_parser as l
import text_reasoner as t
from evaluate_learned_parser import beam,execute
LABELS={0:'true',1:'false',2:'unknown',3:'both',4:'abstain'}

def freeze(a):
 return tuple(freeze(x) for x in a) if isinstance(a,(list,tuple)) else a

def clause_key(a):return json.dumps(a,separators=(',',':'))

def closure(clauses,queries):
 rr=[('b'+str(i),'bound clause',a) for i,a in enumerate(clauses)]
 entities=sorted({v for q in queries for v in q[1:3] if v and v!='?x'})
 for i,v in enumerate(entities):rr.append(('domain'+str(i),'domain bookkeeping',('fact',(),('__bridge_domain__',v,'',True))))
 return t.run(rr,queries[0])[2]

def gate(texts,options,width=16):
 states=beam(options,width)
 if not states:return {'prediction':4,'bank_size':0,'certificate':'no_complete_program','records':[],'query':None,'proof':[],'states':[]}
 programs=[];queries=[]
 for path,score in states:
  rr=[(f's{i+1}',s,options[i][path[i]][0]) for i,s in enumerate(texts[:-1])];q=options[-1][path[-1]][0][2];programs.append(rr);queries.append(q)
 sets=[{clause_key(a) for _,_,a in rr} for rr in programs];common=set.intersection(*sets);union=set.union(*sets)
 lower=closure([freeze(json.loads(s)) for s in sorted(common)],queries)
 upper=closure([freeze(json.loads(s)) for s in sorted(union)],queries)
 def bit(qs):
  if all(q in lower for q in qs):return 1
  if all(q not in upper for q in qs):return 0
  return None
 pos,neg=bit(queries),bit([t.opposite(q) for q in queries]);kind='horn_must_may'
 if pos is not None and neg is not None:prediction=3 if pos and neg else 0 if pos else 1 if neg else 2
 else:
  labels=[t.run(rr,q)[0] for rr,q in zip(programs,queries)];prediction=labels[0] if len(set(labels))==1 else 4;kind='exact_retained_family'
 rr,q=programs[0],queries[0];top,pf,_=t.run(rr,q)
 return {'prediction':prediction,'top1':top,'bank_size':len(states),'certificate':kind,'records':rr,'query':q,'proof':pf if prediction!=4 else [],'states':[p for p,_ in states]}

def infer(model,context,statement,budget=0,width=16,annotator=None,cache=None):
 texts=[s.strip() for s in context.split('.') if s.strip()]+[statement.strip().rstrip('.')]
 cache={} if cache is None else cache
 for s in texts:
  if s not in cache:
   if not l.tokens(s):cache[s]=([],0.);continue
   _,p=model.distributions(s);cache[s]=(model.candidates(s,3),float(p.max(axis=1).min()))
 options=[list(cache[s][0]) for s in texts];options[-1]=[v for v in options[-1] if v[0][0]=='fact' and '?x' not in v[0][2]]
 trace=[];result=gate(texts,options,width)
 for _ in range(budget):
  if result['prediction']!=4:break
  allowed=[i for i in range(len(texts)) if i not in [q['sentence'] for q in trace]]
  if not allowed or annotator is None:break
  missing=[i for i in allowed if not options[i]]
  if missing:allowed=missing
  elif result['states']:
   variable=[i for i in allowed if len({clause_key(options[i][p[i]][0]) for p in result['states']})>1]
   if variable:allowed=variable
  j=min(allowed,key=lambda i:(cache[texts[i]][1],i));ast=annotator(j,texts[j]);options[j]=[(ast,0.)];trace.append({'sentence':j,'text':texts[j],'interpretation':ast});result=gate(texts,options,width)
 result.update(texts=texts,annotations=trace,options=options)
 return result

def fol_program(records,q):
 """Lossless signed-Horn embedding; an audit route to the existing FOL engine."""
 def symbol(prefix,s):return prefix+hashlib.sha256(s.encode()).hexdigest()[:16]
 def atom(a):
  pred,sub,obj,sign=a;args=[sub]+([obj] if obj else []);args=tuple('x' if v=='?x' else symbol('c',v) for v in args)
  return ('atom',symbol('p',pred)+('_pos' if sign else '_neg'),args)
 def conjunction(xs):
  a=xs[0]
  for b in xs[1:]:a=('∧',a,b)
  return a
 clauses=[]
 for _,_,(kind,body,head) in records:
  a=atom(head) if kind=='fact' else ('→',conjunction([atom(b) for b in body]),atom(head))
  if any('?x' in b for b in (*body,head)):a=('∀','x',a)
  clauses.append(a)
 if not clauses:clauses=[('→',atom(q),atom(q))]
 return conjunction(clauses),atom(q),atom(t.opposite(q))

def audit_fol(records,q):
 from fol_certifier import solve
 k,p,n=fol_program(records,q);a,b=solve(k,p),solve(k,n)
 if a['label'] not in ('True','Uncertain') or b['label'] not in ('True','Uncertain'):return {'valid':False,'statuses':[a['label'],b['label']]}
 pos,neg=a['label']=='True',b['label']=='True';label=3 if pos and neg else 0 if pos else 1 if neg else 2
 return {'valid':label==t.run(records,q)[0],'prediction':label,'statuses':[a['label'],b['label']]}

def explanation(result):
 y=result['prediction']
 if y==4:return {'answer':'abstain','reason':'Retained interpretations disagree or no complete program exists.','steps':[]}
 out=t.verbalize(result['records'],result['proof'],y);out['answer']=LABELS[y];out['family_certificate']=result['certificate'];out['retained_programs']=result['bank_size']
 out['proof_scope']='Executed proof of the first retained program; family answer is separately certified. Intended meaning may lie outside the retained beam.'
 return out
