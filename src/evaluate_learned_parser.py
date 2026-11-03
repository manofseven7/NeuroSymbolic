"""Learned text-to-program evaluation, then actual-error semantic repair.

Candidate generation never reads reference ASTs. An oracle interpretation is
admitted only when its sentence is explicitly queried. Gold task labels are
used only after selection. Weights are normalized neural path scores, not
calibrated Bayesian probabilities.
"""
import csv,hashlib,itertools,json,time
from pathlib import Path
import joblib,numpy as np
import learned_parser as l
import text_reasoner as t
import text_pilot as utility

def beam(options,width):
    if any(not x for x in options):return []
    states=[((),0.)]
    for choices in options:
        states=sorted([(path+(j,),score+v[1]) for path,score in states for j,v in enumerate(choices)],key=lambda z:-z[1])[:width]
    return states

def execute(texts,options,path):
    rr=[(f's{i+1}',s,options[i][path[i]][0]) for i,s in enumerate(texts[:-1])]
    q=options[-1][path[-1]][0][2]
    label,pf,_=t.run(rr,q)
    return label,pf,rr,q

def distribution(texts,options,width):
    states=beam(options,width)
    if not states:return states,np.array([]),np.array([]),None
    predictions=np.array([execute(texts,options,h)[0] for h,score in states])
    scores=np.array([score for h,score in states]);weights=np.exp(scores-scores.max());weights/=weights.sum()
    responses=[]
    for q in range(len(options)):
        enc={};col=[]
        for h,_ in states:
            a=json.dumps(options[q][h[q]][0]);enc.setdefault(a,len(enc));col.append(enc[a])
        responses.append(col)
    return states,predictions,weights,np.array(responses).T

def adapt(texts,initial,confidence,policy,budget,width,annotator):
    options=[list(x) for x in initial];trace=[];used=[];novel=0
    states,pred,p,resp=distribution(texts,options,width);source=int(pred[0]) if states else 4
    for _ in range(budget):
        allowed=[q for q in range(len(options)) if q not in used]
        if not allowed:break
        missing=[q for q in allowed if not options[q]]
        if missing:q=min(missing,key=lambda v:(confidence[v],v))
        elif policy=='uncertainty' or not states:q=min(allowed,key=lambda v:(confidence[v],v))
        else:
            # Reuse audited loss, entropy and clean EC2 equations. Zero utility
            # falls back to confidence, identically for the three policies.
            q=utility.select(p,pred,source,resp,used,policy,0.)
            current=utility.decision(p,pred,source)[1]
            # If predictions are unanimous, task-based criteria have no signal.
            # Entropy may still query nuisance ambiguities; that is its policy.
            if len(set(pred))==1 and policy in ['risk','ec2']:q=min(allowed,key=lambda v:(confidence[v],v))
        gold=annotator(q);outside=all(v[0]!=gold for v in options[q]);novel+=outside
        trace.append(dict(sentence=q,annotation=gold,outside_sentence_bank=outside));used.append(q)
        # Query-specific correction is admitted even when missing from the beam.
        # No other sentence is replaced using the reference parser.
        options[q]=[(gold,0.)]
        states,pred,p,resp=distribution(texts,options,width)
    if not states:return dict(prediction=4,trace=trace,novel=novel,proof=[],records=[],query=None,complete=False,chosen=None,bank_size=0)
    chosen=utility.decision(p,pred,source)[0];path=states[chosen][0]
    y,pf,rr,q=execute(texts,options,path)
    return dict(prediction=y,trace=trace,novel=novel,proof=pf,records=rr,query=q,complete=True,chosen=path,bank_size=len(states))

def raw_answer(texts,asts):
    if any(v is None for v in asts):return 4
    rr=[(f's{i+1}',s,a) for i,(s,a) in enumerate(zip(texts[:-1],asts[:-1]))]
    if asts[-1][0]!='fact':return 4
    return t.run(rr,asts[-1][2])[0]

def main():
    cfg=json.loads(Path('configs/learned_parser_v1.json').read_text());out=Path('results/learned_parser_v1');sel=json.loads((out/'selection.json').read_text());target=out/'evaluation';target.mkdir(exist_ok=False);start=time.time()
    assert hashlib.sha256((out/(sel['selected']+'.joblib')).read_bytes()).hexdigest()==sel['model_sha256']
    model=joblib.load(out/(sel['selected']+'.joblib'));linear=joblib.load(out/'logistic0.joblib')
    test=json.loads(Path('third_party/LogicLM_ProofWriter/test.json').read_text());split=json.loads((out/'split.json').read_text());trainset=set(split['train']);sentences=sorted({s.strip() for r in test for s in r['context'].split('.') if s.strip()}|{r['question'].split('?')[-1].strip().rstrip('.') for r in test})
    raw={};lrraw={};options={};confidence={};parseaudit=[]
    for s in sentences:
        w,pp=model.distributions(s);confidence[s]=float(pp.max(axis=1).min())
        try:raw[s]=l.decode(w,model.labels[pp.argmax(axis=1)])
        except t.ParseError:raw[s]=None
        try:lrraw[s]=linear.predict(s)
        except t.ParseError:lrraw[s]=None
        opts=model.candidates(s)
        options[s]=opts
        # Reference ASTs are inspected only AFTER all predictions for this
        # sentence; they are not passed to candidate generation or selection.
        gold=t.sentence(s)
        parseaudit.append(dict(sentence=s,seen_training=s in trainset,raw_valid=raw[s] is not None,raw_exact=raw[s]==gold,linear_exact=lrraw[s]==gold,candidate_contains_reference=any(a==gold for a,score in opts),confidence=confidence[s]))
    rows=[];baseline=[];examples=[];oraclecalls=0;validproof=badproof=aligned=0
    with (target/'traces.jsonl').open('w') as f:
        for i,r in enumerate(test):
            texts=[s.strip() for s in r['context'].split('.') if s.strip()]+[r['question'].split('?')[-1].strip().rstrip('.')]
            oo=[options[s] if j<len(texts)-1 else [v for v in options[s] if v[0][0]=='fact' and '?x' not in v[0][2]] for j,s in enumerate(texts)]
            cc=[confidence[s] for s in texts];goldanswer='ABC'.index(r['answer']);group=hashlib.sha256(r['context'].encode()).hexdigest()
            base=dict(id=r['id'],group=group,gold=goldanswer,neural_raw=raw_answer(texts,[raw[s] for s in texts]),logistic_raw=raw_answer(texts,[lrraw[s] for s in texts]));baseline.append(base)
            goldasts=[t.sentence(s) for s in texts]
            initialbeam=beam(oo,cfg['beam']);covered=any(all(oo[j][h[j]][0]==goldasts[j] for j in range(len(texts))) for h,_ in initialbeam)
            def oracle(q):
                nonlocal oraclecalls
                oraclecalls+=1
                return t.sentence(texts[q])
            for budget in cfg['budgets']:
                policies=['none'] if budget==0 else cfg['policies']
                for policy in policies:
                    result=adapt(texts,oo,cc,policy,budget,cfg['beam'],oracle)
                    y=result['prediction'];pf=result['proof'];proofok=None;sourceok=None
                    if y in [0,1]:
                        proofok=t.verify(result['records'],result['query'],y,pf)
                        goldmap={f's{j+1}':goldasts[j] for j in range(len(texts)-1)}
                        chosenmap={sid:a for sid,txt,a in result['records']}
                        sourceok=result['query']==goldasts[-1][2] and all(chosenmap[node['id']]==goldmap[node['id']] for node in pf)
                        validproof+=1;badproof+=not proofok;aligned+=sourceok
                    row=dict(id=r['id'],group=group,budget=budget,policy=policy,prediction=y,gold=goldanswer,correct=int(y==goldanswer),abstain=int(y==4),inconsistent=int(y==3),spent=len(result['trace']),novel_annotations=result['novel'],program_bank_size=result['bank_size'],initial_reference_program_in_beam=int(covered),proof_relative_valid=proofok,proof_reference_aligned=sourceok)
                    rows.append(row);f.write(json.dumps({'metrics':row,'queries':result['trace'],'selected':result['chosen']})+'\n')
                    if policy=='risk' and budget==2 and len(examples)<8 and pf:
                        examples.append(dict(metrics=row,question=r['question'],explanation=t.verbalize(result['records'],pf,y),corrected_sentences=result['trace']))
            if (i+1)%100==0:print('learned eval',i+1,flush=True);f.flush()
    with (target/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (target/'sentence_audit.json').write_text(json.dumps(parseaudit,indent=2));(target/'baselines.json').write_text(json.dumps(baseline,indent=2));(target/'explanation_examples.json').write_text(json.dumps(examples,indent=2))
    result=dict(episodes=len(rows),oracle_calls=oraclecalls,positive_negative_proof_checks=validproof,invalid_program_relative_proofs=badproof,reference_aligned_proofs=aligned,seconds=time.time()-start,selection_sha256=hashlib.sha256((out/'selection.json').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Reference alignment is agreement with grammar teacher,not independent human semantic validation. Abstention4 distinct from logicalunknown2.')
    (target/'execution.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
