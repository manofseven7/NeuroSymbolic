"""Neural token-role semantic parser with a deterministic typed AST decoder.

Reference parser is used ONLY to create development supervision and audit gold
ASTs. predict() never invokes it. Decoder copies spans and resolves ellipsis;
it is not an unrestricted semantic parser or pretrained language model.
"""
import hashlib,itertools,json,re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import text_reasoner as t

def tokens(s):return re.findall(r'[a-z]+|,',s.lower())

def teacher(s):
    w=tokens(s);tags=['O']*len(w)
    def mark_subject(indices,part,kind):
        for i in indices:
            if w[i]=='the':continue
            tags[i]=part+('.VAR' if w[i] in ['someone','something','somebody','they','it','things','people','everything'] else '.'+kind)
    def span(lo,hi,part):
        words=w[lo:hi]
        cop=next((i for i in range(lo,hi) if w[i] in ['is','are']),None)
        if cop is not None:
            mark_subject(range(lo,cop),part,'SUB');tags[hi-1]=part+'.PRED'
            for i in range(cop+1,hi-1):
                if w[i]=='not':tags[i]=part+'.NEG'
        else:
            verb=next((i for i in range(lo,hi) if re.fullmatch(r'chases?|eats?|likes?|needs?|sees?|visits?',w[i])),None)
            if verb is not None:
                aux=next((i for i in range(lo,verb) if w[i] in ['does','do']),verb)
                mark_subject(range(lo,aux),part,'SUB');tags[verb]=part+'.PRED';mark_subject(range(verb+1,hi),part,'OBJ')
                for i in range(aux,verb):
                    if w[i]=='not':tags[i]=part+'.NEG'
            else:
                tags[hi-1]=part+'.PRED'
                for i in range(lo,hi-1):
                    if w[i]=='not':tags[i]=part+'.NEG'
    if w[0]=='if':
        then=w.index('then');cuts=[0]+[i for i in range(1,then) if w[i]=='and']+[then]
        for j,(left,right) in enumerate(zip(cuts,cuts[1:])):span(left+1,right,'B'+str(j))
        span(then+1,len(w),'H')
    elif 'things' in w or 'people' in w:
        vi=w.index('things') if 'things' in w else w.index('people');start=1 if w[0]=='all' else 0
        cuts=[start-1]+[i for i in range(start,vi) if w[i] in [',','and']]+[vi]
        for j,(left,right) in enumerate(zip(cuts,cuts[1:])):span(left+1,right,'B'+str(j))
        tags[vi]='B0.VAR';span(w.index('are')+1,len(w),'H')
    else:span(0,len(w),'F')
    ast=decode(w,tags)
    if ast!=t.sentence(s):raise ValueError(('teacher alignment',s,ast,t.sentence(s),tags))
    return w,tags

def features(w):
    n=len(w);then=w.index('then') if 'then' in w else -1
    result=[]
    for i,word in enumerate(w):
        d={'word':word,'suffix2':word[-2:],'suffix3':word[-3:],'prefix2':word[:2],'position':i/max(1,n-1),'length':n/30,'has_if':w[0]=='if','has_then':then>=0,'then_distance':(i-then)/30 if then>=0 else 0.,'commas_before':sum(v==',' for v in w[:i]),'and_before':sum(v=='and' for v in w[:i])}
        for offset in [-3,-2,-1,1,2,3]:d['offset'+str(offset)]=w[i+offset] if 0<=i+offset<n else '<edge>'
        result.append(d)
    return result

def decode(w,tags):
    slots={}
    for word,tag in zip(w,tags):
        if tag=='O':continue
        part,role=tag.split('.');slots.setdefault(part,{}).setdefault(role,[]).append(word)
    def make(part,default=None):
        z=slots.get(part,{})
        if len(z.get('PRED',[]))!=1:raise t.ParseError('neural predicate span')
        if 'VAR' in z:
            if 'SUB' in z:raise t.ParseError('conflicting subject slots')
            sub='?x'
        elif 'SUB' in z:sub=' '.join(z['SUB'])
        elif default is not None:sub=default
        else:raise t.ParseError('missing neural subject')
        obj=' '.join(z.get('OBJ',[]));pred=z['PRED'][0]
        if obj and pred.endswith('s'):pred=pred[:-1]
        return (pred,sub,obj,'NEG' not in z)
    if 'F' in slots:
        if set(slots)!={'F'}:raise t.ParseError('conflicting fact/rule slots')
        a=make('F')
        if '?x' in a:raise t.ParseError('variable neural fact')
        return ('fact',(),a)
    if 'H' not in slots or 'B0' not in slots:raise t.ParseError('missing neural rule slots')
    if set(slots)-{'H','B0','B1'}:raise t.ParseError('unsupported rule arity')
    body=[make('B0')]
    if 'B1' in slots:body.append(make('B1',body[0][1]))
    return ('rule',tuple(body),make('H',body[0][1]))

class Parser:
    def __init__(self,kind='mlp',hidden=64,seed=42):
        self.vectorizer=DictVectorizer();self.kind=kind
        self.model=MLPClassifier(hidden_layer_sizes=(hidden,),max_iter=120,alpha=.01,random_state=seed,early_stopping=True,validation_fraction=.1,n_iter_no_change=10) if kind=='mlp' else LogisticRegression(C=1,max_iter=300)
    def fit(self,sentences):
        xx=[];yy=[]
        for s in sentences:
            w,y=teacher(s);xx.extend(features(w));yy.extend(y)
        self.roles,encoded=np.unique(yy,return_inverse=True)
        self.model.fit(self.vectorizer.fit_transform(xx),encoded)
        return self
    @property
    def labels(self):return self.roles[self.model.classes_]
    def distributions(self,s):
        w=tokens(s);x=self.vectorizer.transform(features(w));return w,self.model.predict_proba(x)
    def predict(self,s):
        w,p=self.distributions(s);tags=self.labels[p.argmax(axis=1)]
        return decode(w,tags)
    def candidates(self,s,limit=3):
        w,p=self.distributions(s);best=p.argmax(axis=1);orders=np.argsort(-p,axis=1);uncertain=np.argsort(-p.max(axis=1))[-4:]
        variants=[best.copy()]
        for i in uncertain:
            z=best.copy();z[i]=orders[i,1];variants.append(z)
        # Small fixed beam of two uncertain token changes; never uses reference AST.
        for i,j in itertools.combinations(uncertain,2):
            z=best.copy();z[i]=orders[i,1];z[j]=orders[j,1];variants.append(z)
        out={}
        for indices in variants:
            try:a=decode(w,self.labels[indices])
            except t.ParseError:continue
            score=float(np.log(np.maximum(p[np.arange(len(w)),indices],1e-300)).sum());k=json.dumps(a)
            if k not in out or score>out[k][1]:out[k]=(a,score)
        return sorted(out.values(),key=lambda v:-v[1])[:limit]

def split_sentences(dev):
    sentences=sorted({s.strip() for r in dev for s in r['context'].split('.') if s.strip()}|{r['question'].split('?')[-1].strip().rstrip('.') for r in dev})
    train=[];valid=[]
    for s in sentences:
        (valid if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5==0 else train).append(s)
    return train,valid
