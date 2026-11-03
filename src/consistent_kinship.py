"""NS023: reciprocal relation decoding and gender-typed learned compositions.
Training may use gender annotations to label inverse training examples.
Inference accepts ONLY story/query; no gold facts, genders, paths or target.
"""
import collections,itertools,re
import numpy as np
from sklearn.linear_model import LogisticRegression
import external_kinship as k
from local_kinship_repair import local_text
PAIRS={'parent':('father','mother'),'child':('son','daughter'),'sibling':('brother','sister'),'spouse':('husband','wife'),'grandparent':('grandfather','grandmother'),'grandchild':('grandson','granddaughter'),'avuncular':('uncle','aunt'),'nibling':('nephew','niece'),'parent_in_law':('father-in-law','mother-in-law'),'child_in_law':('son-in-law','daughter-in-law'),'sibling_in_law':('brother-in-law','sister-in-law')}
TYPE={r:t for t,rs in PAIRS.items() for r in rs}
SEX={r:i for rs in PAIRS.values() for i,r in enumerate(rs)}
INVERSE={'parent':'child','child':'parent','sibling':'sibling','spouse':'spouse','grandparent':'grandchild','grandchild':'grandparent','avuncular':'nibling','nibling':'avuncular','parent_in_law':'child_in_law','child_in_law':'parent_in_law','sibling_in_law':'sibling_in_law'}
def local_pairs(story):
 pairs=set()
 for sentence in re.split(r'[.!?]',story):pairs.update(itertools.permutations(k.names(sentence),2))
 return sorted(pairs)
def training_genders(row):
 return {n:0 if g=='male' else 1 for n,g in (x.rsplit(':',1) for x in row['genders'].split(','))}
def generalized_rules(rows):
 counts=collections.Counter()
 for row in rows:
  for a,b,c in k.supervision(row)[1]:
   if a in TYPE and b in TYPE and c in TYPE:counts[TYPE[a],TYPE[b],TYPE[c]]+=1
 chosen={}
 for (a,b,c),n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
  chosen.setdefault((a,b),c)
 rules=[]
 for (a,b),c in sorted(chosen.items()):
  for ar in PAIRS[a]:
   for sex,br in enumerate(PAIRS[b]):rules.append((ar,br,PAIRS[c][sex]))
 return rules,{'type_counts':[[*r,n] for r,n in sorted(counts.items())],'chosen':[[*ab,c] for ab,c in sorted(chosen.items())]}
class ConsistentGrounder:
 def __init__(self,seed=42):self.seed=seed
 def fit(self,rows):
  texts=[];labels=[];augmented=0
  for row in rows:
   facts,_=k.supervision(row);sex=training_genders(row)
   for (a,b),r in sorted(facts.items()):
    texts.append(local_text(row['story'],(a,b)));labels.append(r)
    if r in TYPE and a in sex:
     inv=PAIRS[INVERSE[TYPE[r]]][sex[a]]
     texts.append(local_text(row['story'],(b,a)));labels.append(inv);augmented+=1
  self.vec=k.features();X=self.vec.fit_transform(texts);self.clf=LogisticRegression(C=2,max_iter=300,class_weight='balanced',random_state=self.seed).fit(X,labels)
  self.train_counts={'stories':len(rows),'reference_positive_examples':len(labels)-augmented,'inverse_examples_using_training_gender':augmented,'negative_examples':0}
  return self
 def prepare(self,story,query,reciprocal=False):
  pairs=local_pairs(story)
  if not pairs:return {'story':story,'query':tuple(query),'pairs':[],'options':[]}
  probs=self.clf.predict_proba(self.vec.transform([local_text(story,p) for p in pairs]));labels=list(self.clf.classes_)
  if not reciprocal:
   options=[]
   for p in probs:
    order=np.argsort(-p);options.append([(str(labels[j]),float(p[j])) for j in order[:2]])
   return {'story':story,'query':tuple(query),'pairs':pairs,'options':options}
  idx={pair:i for i,pair in enumerate(pairs)};chosen={};sex_evidence=collections.defaultdict(lambda:np.zeros(2))
  for a,b in pairs:
   if a>b:continue
   i,j=idx[a,b],idx[b,a];scores={}
   for t in PAIRS:
    inv=INVERSE[t];p=sum(probs[i,x] for x,r in enumerate(labels) if TYPE[r]==t);q=sum(probs[j,x] for x,r in enumerate(labels) if TYPE[r]==inv)
    scores[t]=p*q
   t=max(scores,key=lambda x:(scores[x],x));chosen[a,b]=t;chosen[b,a]=INVERSE[t]
   for node,row,kind in [(b,i,t),(a,j,INVERSE[t])]:
    pp=np.array([sum(probs[row,x] for x,r in enumerate(labels) if TYPE[r]==kind and SEX[r]==s) for s in [0,1]])
    pp=(pp+1e-9)/(pp.sum()+2e-9);sex_evidence[node]+=np.log(pp)
  sexes={n:int(np.argmax(p)) for n,p in sex_evidence.items()};options=[]
  for (a,b),p in zip(pairs,probs):
   rel=PAIRS[chosen[a,b]][sexes[b]];prob=float(p[labels.index(rel)]) if rel in labels else 1e-9
   options.append([(rel,max(prob,1e-9)),(rel,max(prob,1e-9))])
  return {'story':story,'query':tuple(query),'pairs':pairs,'options':options}
