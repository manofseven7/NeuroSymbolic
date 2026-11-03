import csv,hashlib,json,time
from pathlib import Path
from collections import defaultdict
from fol_certifier import parse
from fol_model_reuse import run
from fol_interval_certifier import run as interval_run
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'configs/folio_interval_v2.json').read_text());data=[json.loads(x) for x in (ROOT/'third_party/FOLIO_curated/dataset_rawpairs.jsonl').read_text().splitlines()];byid={x['id']:x for x in data}
    refs=json.loads((ROOT/'results/folio_certified_v1/records.json').read_text());out=ROOT/'results/folio_interval_v2';out.mkdir(exist_ok=True);rows=[];examples=[];start=time.monotonic()
    for rep in range(cfg['repetitions']):
        for i,r in enumerate(refs):
            k=parse(byid[r['story_id']]['FOL_sentence_new']);q=parse(byid[r['id']]['FOL_sentence_new']);methods=cfg['methods'];shift=(i+rep)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                got=interval_run(k,q,timeout=cfg['timeout_ms']) if method=='interval_gate' else run(k,q,method,timeout=cfg['timeout_ms']);ev=got.pop('reuse_evidence');got.pop('obligations',None);got.setdefault('certificate_kind','enumerated_or_selector')
                row={'id':r['id'],'story_id':r['story_id'],'rep':rep,'method':method,'reference_certificate':r['robust']['label'],**got};row['matches_reference_certificate']=row['label']==row['reference_certificate'];rows.append(row)
                if not row['matches_reference_certificate']:raise AssertionError('Certificate disagreement '+str(row))
                if rep==0 and method=='selector_model_reuse' and ev and len(examples)<4:examples.append({'id':r['id'],'evidence':ev})
            if (i+1)%25==0:print('rep',rep,'question',i+1,'elapsed',round(time.monotonic()-start,2),flush=True)
    with (out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for method in cfg['methods']:
        rr=[x for x in rows if x['method']==method];summary[method]={'runs':len(rr),'solver_checks':sum(x['solver_checks'] for x in rr),'mean_solver_checks':sum(x['solver_checks'] for x in rr)/len(rr),'total_runtime_seconds':sum(x['runtime_seconds'] for x in rr),'mean_runtime_ms':sum(x['runtime_seconds'] for x in rr)/len(rr)*1000,'certificate_disagreements':sum(not x['matches_reference_certificate'] for x in rr),'model_reuse_hits':sum(x['model_reuse_hits'] for x in rr),'finite_model_checks':sum(x['finite_model_checks'] for x in rr)}
    summary['scope']='Exploratory same bounded semantic-input benchmark, maximum four OR/XOR sites. CPU wall time includes model extraction and independent witness evaluation. Runtime depends on environment. No NLP accuracy advantage or novelty claim.'
    summary['code_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/fol_model_reuse.py','src/fol_interval_certifier.py','src/fast_model_export.py','src/benchmark_fol_interval.py','configs/folio_interval_v2.json']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'reuse_examples.json').write_text(json.dumps(examples,ensure_ascii=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
