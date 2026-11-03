"""Controlled-English benchmark plus explicitly injected parser-fault diagnostic.

Not pretrained LLM evaluation; not a new end-to-end neural-symbolic method.
"""
import csv,hashlib,itertools,json,time,warnings
from pathlib import Path
import numpy as np
import text_reasoner as t
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.exceptions import ConvergenceWarning

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def entropy(p):return float(-(p*np.log2(np.maximum(p,1e-300))).sum())

def decision(p,pred,source):
    loss=(pred[:,None]!=pred[None,:])*(1+.5*(pred[None,:]==source))
    risk=loss@p;best=int(np.argmin(risk));return best,float(risk[best])

def select(p,pred,source,responses,used,policy,noise):
    base=decision(p,pred,source)[1];scores=[]
    for q,ans in enumerate(responses.T):
        if q in used:scores.append(-np.inf);continue
        k=int(ans.max())+1
        if k==1:scores.append(0.);continue
        if policy=='entropy':before=entropy(p)
        elif policy=='ec2':
            mass=np.bincount(pred,weights=p,minlength=4);before=.5*(p.sum()**2-(mass**2).sum())
        else:before=base
        after=0.
        for o in range(k):
            lk=np.where(ans==o,1-noise,noise/(k-1));joint=p*lk;mass=joint.sum()
            if not mass:continue
            post=joint/mass
            if policy=='entropy':v=entropy(post)
            elif policy=='ec2':
                # Original noiseless live-edge formula; noisy posterior surrogate.
                weights=joint if noise==0 else post
                cm=np.bincount(pred,weights=weights,minlength=4);v=.5*(weights.sum()**2-(cm**2).sum())
            else:v=decision(post,pred,source)[1]
            after+=mass*v
        scores.append(before-after)
    return int(np.argmax(scores))

def build(records,q):
    variants=list(itertools.product([0,1],repeat=3));programs=[t.transform(records,b) for b in variants]
    pred=np.array([t.run(rr,q)[0] for rr in programs]);resp=[]
    for j in range(len(records)):
        enc={};col=[]
        for rr in programs:
            key=json.dumps(rr[j][2]);enc.setdefault(key,len(enc));col.append(enc[key])
        resp.append(col)
    return programs,pred,np.array(resp,dtype=int).T,variants

def episode(pred,resp,variants,case,budget,policy,noise,seed):
    source=variants.index(tuple(case));prior=np.full(8,.4/7);prior[source]=.6;p=prior.copy()
    if policy=='static':return source,[]
    rng=np.random.default_rng(seed);observations=[]
    for q in range(resp.shape[1]):
        k=int(resp[:,q].max())+1;o=int(resp[0,q])
        if k>1 and rng.random()<noise:o=int(rng.choice([v for v in range(k) if v!=o]))
        observations.append(o)
    trace=[];used=[]
    for step in range(budget):
        if len(used)>=resp.shape[1]:break
        q=select(p,pred,int(pred[source]),resp,used,policy,noise);o=observations[q];used.append(q);trace.append([q,o])
        k=int(resp[:,q].max())+1
        p*=1 if k==1 else np.where(resp[:,q]==o,1-noise,noise/(k-1))
        if p.sum()==0:return source,trace
        p/=p.sum()
    return decision(p,pred,int(pred[source]))[0],trace

def main():
    cfg=json.loads(Path('configs/text_pilot_v1.json').read_text());out=Path('results/text_pilot_v1');out.mkdir(exist_ok=False);start=time.time()
    assert digest('src/text_reasoner.py')==cfg['parser_sha256']
    data=Path('third_party/LogicLM_ProofWriter');dev=json.loads((data/'dev.json').read_text());test=json.loads((data/'test.json').read_text())
    assert not {r['context'] for r in dev}&{r['context'] for r in test}
    labels=lambda rows:np.array(['ABC'.index(r['answer']) for r in rows])
    # Feature vocabulary and neural parameters see development data only.
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=10000,sublinear_tf=True)
    text=lambda rows:[r['context']+' QUESTION '+r['question'] for r in rows]
    X=vec.fit_transform(text(dev));Xt=vec.transform(text(test));yd=labels(dev);yt=labels(test);base={};predictions={}
    estimators=[('tfidf_logistic',LogisticRegression(C=1.,max_iter=500,random_state=42)),('tfidf_mlp',MLPClassifier(hidden_layer_sizes=(64,),activation='relu',solver='adam',alpha=.01,max_iter=200,random_state=42,early_stopping=False))]
    for name,model in estimators:
        st=time.time();model.fit(X,yd);z=model.predict(Xt);predictions[name]=z.tolist()
        base[name]=dict(accuracy=float(accuracy_score(yt,z)),macro_f1=float(f1_score(yt,z,labels=[0,1,2],average='macro')),seconds=time.time()-st,training_records=len(dev),iterations=int(np.max(model.n_iter_)))
        print(name,base[name],flush=True)
    rows=[];clean=[];parse_errors=[];proofcounts=dict(outputs_checked=0,proofs_checked=0,invalid=0,unknown=0,inconsistent=0);examples=[]
    cases={'clean':(0,0,0),'fact_negation_fault':(1,0,0),'rule_fault':(0,1,1),'joint_fault':(1,1,1)}
    with (out/'traces.jsonl').open('w') as f:
        for i,r in enumerate(test):
            group=hashlib.sha256(r['context'].encode()).hexdigest();gold=int(yt[i])
            try:records=t.parse(r['context']);q=t.query(r['question']);programs,pred,resp,variants=build(records,q)
            except t.ParseError as e:
                parse_errors.append({'id':r['id'],'error':str(e)});clean.append(2);continue
            clean.append(int(pred[0]));y,pf,_=t.run(programs[0],q)
            if len(examples)<8 and len(pf)>2:examples.append(dict(id=r['id'],question=r['question'],gold=gold,explanation=t.verbalize(records,pf,y)))
            for ci,(case,bits) in enumerate(cases.items()):
                for noise,budget,policy in itertools.product([0.,.1],[1,2],['static','entropy','ec2','risk']):
                    chosen,trace=episode(pred,resp,variants,bits,budget,policy,noise,i*100+ci)
                    source=variants.index(bits);z=int(pred[chosen]);row=dict(id=r['id'],group=group,case=case,noise=noise,budget=budget,policy=policy,prediction=z,gold=gold,correct=int(z==gold),source_correct=int(pred[source]==gold),changed=int(z!=pred[source]),selected=chosen,spent=len(trace))
                    rows.append(row);f.write(json.dumps({'metrics':row,'semantic_queries':trace})+'\n')
                    if policy=='risk' and budget==2:
                        yy,proof,_=t.run(programs[chosen],q);assert yy==z
                        proofcounts['outputs_checked']+=1
                        if z in (0,1):
                            proofcounts['proofs_checked']+=1;proofcounts['invalid']+=not t.verify(programs[chosen],q,z,proof)
                        elif z==2:proofcounts['unknown']+=1
                        else:proofcounts['inconsistent']+=1
            if (i+1)%100==0:print('text test processed',i+1,flush=True);f.flush()
    base['controlled_english_parser']=dict(accuracy=float(accuracy_score(yt,clean)),macro_f1=float(f1_score(yt,clean,labels=[0,1,2],average='macro')),parsed=len(test)-len(parse_errors),total=len(test),fallback='unknown on parse error')
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    predictions['controlled_english_parser']=clean
    (out/'clean_baselines.json').write_text(json.dumps(base,indent=2));(out/'predictions.json').write_text(json.dumps(predictions));(out/'parse_errors.json').write_text(json.dumps(parse_errors,indent=2));(out/'proof_checks.json').write_text(json.dumps(proofcounts,indent=2));(out/'explanation_examples.json').write_text(json.dumps(examples,indent=2))
    (out/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,episodes=len(rows),code_sha=digest(__file__),parser_sha=digest(t.__file__),numpy=np.__version__),indent=2))
    print(json.dumps({'clean':base,'proofs':proofcounts,'episodes':len(rows)},indent=2))

if __name__=='__main__':main()
