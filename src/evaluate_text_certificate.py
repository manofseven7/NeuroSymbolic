import csv,hashlib,json,time
from pathlib import Path
from collections import Counter
import joblib
import text_reasoner as t
from text_certificate import infer,audit_fol,explanation,clause_key
ROOT=Path(__file__).resolve().parents[1]
def main():
 cfg=json.loads((ROOT/'configs/text_certificate_v1.json').read_text());out=ROOT/'results/text_certificate_v1';out.mkdir(exist_ok=True);weights=ROOT/cfg['model'];assert hashlib.sha256(weights.read_bytes()).hexdigest()==cfg['model_sha256'];model=joblib.load(weights);data=json.loads((ROOT/'third_party/LogicLM_ProofWriter/test.json').read_text());cache={};rows=[];examples=[];start=time.monotonic()
 with (out/'traces.jsonl').open('w') as f:
  for i,r in enumerate(data):
   statement=r['question'].split('?')[-1].strip().rstrip('.');gold='ABC'.index(r['answer'])
   for budget in cfg['budgets']:
    got=infer(model,r['context'],statement,budget=budget,annotator=lambda i,s:t.sentence(s),cache=cache)
    # Audit only after prediction. Reference ASTs are never passed to the gate.
    refs=[t.sentence(s) for s in got['texts']];covered=any(all(got['options'][j][path[j]][0]==refs[j] for j in range(len(refs))) for path in got['states'])
    exact=[t.run([(f's{j+1}',s,got['options'][j][path[j]][0]) for j,s in enumerate(got['texts'][:-1])],got['options'][-1][path[-1]][0][2])[0] for path in got['states']]
    expected=exact[0] if exact and len(set(exact))==1 else 4;assert got['prediction']==expected
    audit=audit_fol(got['records'],got['query']) if budget==0 and got['bank_size'] else None
    if audit:assert audit['valid'],(r['id'],audit)
    proofok=t.verify(got['records'],got['query'],got['prediction'],got['proof']) if got['prediction'] in [0,1] else None
    if proofok is not None:assert proofok
    row={'id':r['id'],'group':hashlib.sha256(r['context'].encode()).hexdigest(),'budget':budget,'gold':gold,'prediction':got['prediction'],'top1':got.get('top1',4),'correct':int(got['prediction']==gold),'accepted':int(got['prediction']!=4),'bank_size':got['bank_size'],'annotations':len(got['annotations']),'reference_in_beam':covered,'certificate':got['certificate'],'proof_valid':proofok,'fol_bridge_valid':audit['valid'] if audit else None};rows.append(row)
    f.write(json.dumps({'metrics':row,'queried_sentences':got['annotations'],'retained_paths':got['states']})+'\n')
    if len(examples)<8 and (got['annotations'] or (budget==0 and i<4)):examples.append({'id':r['id'],'budget':budget,'context':r['context'],'question':statement,'output':explanation(got)})
   if (i+1)%50==0:print('text integration',i+1,'seconds',round(time.monotonic()-start,1),flush=True);f.flush()
 summary={'questions':len(data),'unique_contexts':len({r['group'] for r in rows}),'neural_model_sha256':cfg['model_sha256'],'budgets':{},'runtime_seconds':time.monotonic()-start,'fol_bridge_audits':sum(r['fol_bridge_valid'] is True for r in rows),'invalid_bridge_audits':sum(r['fol_bridge_valid'] is False for r in rows),'family_reference_checks':len(rows),'scope':cfg['scope'],'test_reuse':cfg['data']}
 for budget in cfg['budgets']:
  rr=[r for r in rows if r['budget']==budget];accepted=sum(r['accepted'] for r in rr);summary['budgets'][str(budget)]={'accepted':accepted,'coverage':accepted/len(rr),'correct':sum(r['correct'] for r in rr),'all_record_accuracy':sum(r['correct'] for r in rr)/len(rr),'accepted_accuracy':sum(r['correct'] for r in rr)/accepted if accepted else None,'top1_accuracy':sum(r['top1']==r['gold'] for r in rr)/len(rr),'semantic_annotations':sum(r['annotations'] for r in rr),'reference_program_covered':sum(r['reference_in_beam'] for r in rr),'certificate_types':dict(Counter(r['certificate'] for r in rr)),'proof_checks':sum(r['proof_valid'] is True for r in rr)}
 summary['code_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/text_certificate.py','src/evaluate_text_certificate.py','configs/text_certificate_v1.json']}
 with (out/'questions.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'examples.json').write_text(json.dumps(examples,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
