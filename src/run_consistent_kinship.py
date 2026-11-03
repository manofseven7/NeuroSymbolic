"""Fixed2x2validation study before a new held-out config; no test feedback."""
import json,time,joblib
from pathlib import Path
import external_kinship as k
from local_kinship_repair import ROOT,read,digest,write_rows,evaluate,summarize
from consistent_kinship import ConsistentGrounder,generalized_rules
OUT=ROOT/'results/consistent_kinship_v1'
VARIANTS=[('positive_local',False,False),('reciprocal',True,False),('typed',False,True),('reciprocal_typed',True,True)]
def infer(model,row,rules,reciprocal):
 p=model.prepare(row['story'],k.parse_query(row['query']),reciprocal);facts=k.selected_facts(p);ex=k.execute(facts,p['query'],rules);return p,facts,ex
def eval_rows(model,rows,rules,reciprocal,seed,method):
 out=[]
 for row in rows:
  p,f,ex=infer(model,row,rules,reciprocal);ref,_=k.supervision(row)
  out.append({'id':row['id'],'seed':seed,'method':method,'hops':int(row['task_name'].split('.')[-1]),'correct':int(ex['preferred']==row['target_text']),'prediction':ex['preferred'],'gold':row['target_text'],'answered':int(bool(ex['relations'])),'ref_edges':len(ref),'recovered':sum(pair in f and f[pair][0]==rel for pair,rel in ref.items()),'invalid_proof':int(bool(ex['proof']) and not k.verify_proof(ex['proof'],f,rules)),'limited':int(ex['limited'])})
 return out
def main():
 OUT.mkdir(exist_ok=True)
 (OUT/'protocol.json').write_text(json.dumps({'experiment':'NS-CONSISTENT-KINSHIP-023','variants':VARIANTS,'selection':'128oldgatevalidation, ties in listed order; audit separate128; no newtest tuning','budget':'same256trainingstories/seed; extra training gender used ONLY for inverse labels, explicitly an added annotation assumption','inference':'story/query only; cooccurring entity pairs, fixed C2LR; learned rule types; no oracle correction','fresh_config':'gen_train2345_test2to10/test.csv if exists; otherwise untouched subset of validation config, chosen before test results','stop':'one2x2study across3seeds; retain all outcomes; no test-driven search','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
 tr={r['id']:r for r in read('train.csv')};va={r['id']:r for r in read('validation.csv')};selections={};validation=[]
 for seed in [42,43,44]:
  fr=json.loads((ROOT/f'results/external_kinship_v1/frozen_{seed}.json').read_text());rows=[tr[i] for i in fr['training_ids']];model=ConsistentGrounder(seed).fit(rows);joblib.dump(model,OUT/f'model_{seed}.joblib');typed,rule_info=generalized_rules(rows);gate=[va[i] for i in fr['gate_ids']];audit=[va[i] for i in fr['audit_ids']];scores={}
  for name,rec,typ in VARIANTS:
   rr=eval_rows(model,gate,typed if typ else fr['bundle_rules'],rec,seed,name);scores[name]=sum(r['correct'] for r in rr);validation+=rr
  selected=max(scores,key=scores.get);rec,typ=next((r,t) for n,r,t in VARIANTS if n==selected);rr=eval_rows(model,audit,typed if typ else fr['bundle_rules'],rec,seed,selected)
  selections[str(seed)]={'selected':selected,'reciprocal':rec,'typed':typ,'rules':typed if typ else fr['bundle_rules'],'gate_correct':scores,'gate_n':len(gate),'audit_correct':sum(r['correct'] for r in rr),'audit_n':len(audit),'model_sha256':digest(OUT/f'model_{seed}.joblib'),'training_counts':model.train_counts,'rule_type_induction':rule_info}
  print(seed,{key:selections[str(seed)][key] for key in ['selected','gate_correct','audit_correct','training_counts']},flush=True)
 write_rows(OUT/'validation_predictions.csv',validation);(OUT/'FROZEN_SELECTION.json').write_text(json.dumps(selections,indent=2))
 print('FROZEN',flush=True)
if __name__=='__main__':main()
