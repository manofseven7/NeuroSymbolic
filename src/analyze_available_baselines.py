"""NS024: analyze complete, explicitly separated execution scopes.

This does not relax the final three-seed gate in analyze_published_baselines.py.
Seed42 is the first scheduled LLM seed, not an outcome-selected seed.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/published_baselines_v1'
DEST=OUT/'available_scopes'
LABELS=['father','mother','son','daughter','brother','sister','husband','wife',
        'grandfather','grandmother','grandson','granddaughter','uncle','aunt',
        'nephew','niece','father-in-law','mother-in-law','son-in-law',
        'daughter-in-law','brother-in-law','sister-in-law']

def csvrows(name):
    return list(csv.DictReader((OUT/name).open()))

def aligned(rows, reference):
    key=lambda r:(int(r['seed']),r['id'])
    rr={key(r):r for r in rows}
    oo={key(r):r for r in reference}
    assert len(rr)==len(rows), 'Duplicate prediction key'
    assert len(oo)==len(reference), 'Duplicate reference key'
    assert set(rr)==set(oo), 'Incomplete or mismatched prediction scope'
    for k,r in rr.items():
        assert r['gold']==oo[k]['gold'], ('Answer mismatch',k)
        assert int(r['hops'])==int(oo[k]['hops']), ('Depth mismatch',k)
        assert int(r['correct'])==int(r['prediction']==r['gold']), ('Scoring mismatch',k)
    return rr,oo

def paired_interval(rows, reference, repetitions=10000):
    rr,oo=aligned(rows,reference)
    ids=sorted({k[1] for k in rr})
    seeds=sorted({k[0] for k in rr})
    assert len(rr)==len(ids)*len(seeds), 'Unbalanced story/seed design'
    d=np.array([np.mean([int(oo[s,i]['correct'])-int(rr[s,i]['correct']) for s in seeds]) for i in ids])
    rng=np.random.default_rng(24024)
    bs=np.concatenate([rng.choice(d,(1000,len(d)),replace=True).mean(1) for _ in range(repetitions//1000)])*100
    lo,hi=np.quantile(bs,[.025,.975])
    return {'ours_minus_baseline_pp':float(100*d.mean()),'ci95_low':float(lo),
            'ci95_high':float(hi),'story_n':len(ids),'seeds':';'.join(map(str,seeds)),
            'replicates':repetitions,'interpretation':'exploratory; conditional on available fixed training seeds and seen test'}

def main():
    DEST.mkdir(exist_ok=True)
    ours=csvrows('ours_replayed.csv')
    inputs=json.loads((OUT/'payload/test_inputs.json').read_text())
    ids={r['id'] for r in inputs}
    assert len(inputs)==len(ids)==1026
    assert len(ours)==3078
    assert {(int(r['seed']),r['id']) for r in ours}=={(s,i) for s in [42,43,44] for i in ids}
    labels=sorted({r['target_text'] for s in [42,43,44] for r in json.loads((OUT/f'payload/train_{s}.json').read_text())})
    groups=[('graph_3seeds','Ours_NS023',ours,ours),
            ('graph_3seeds','EpiGNN_query',csvrows('epignn_predictions.csv'),ours),
            ('graph_3seeds','EpiGNN_program',csvrows('epignn_program_predictions.csv'),ours)]
    records=[json.loads(x) for x in (OUT/'payload/llm_results/predictions.jsonl').read_text().splitlines()]
    allkeys=[(int(r['seed']),r['method'],r['id']) for r in records]
    assert len(allkeys)==len(set(allkeys))
    assert all(r['id'] in ids for r in records)
    ours42=[r for r in ours if int(r['seed'])==42]
    groups.append(('llm_seed42','Ours_NS023',ours42,ours42))
    for method in ['cot','faithful','htt']:
        rows=[r for r in records if int(r['seed'])==42 and r['method']==method]
        assert len(rows)==1026
        groups.append(('llm_seed42',method,rows,ours42))
        if method=='faithful':
            matched=[{**r,'prediction':r['matched_prediction'],'correct':r['matched_correct']} for r in rows]
            groups.append(('llm_seed42','faithful_training_only',matched,ours42))
    metrics=[];paired=[];depth=[];resources=[]
    for scope,method,rows,reference in groups:
        aligned(rows,reference)
        for subset in ['all','short_2_3','long_4_10']:
            keep=lambda r:subset=='all' or (int(r['hops'])>=4)==(subset=='long_4_10')
            rr=[r for r in rows if keep(r)];oo=[r for r in reference if keep(r)]
            metrics.append({'scope':scope,'method':method,'subset':subset,'n_predictions':len(rr),
                'n_stories':len({r['id'] for r in rr}),'seeds':';'.join(map(str,sorted({int(r['seed']) for r in rr}))),
                'correct':sum(int(r['correct']) for r in rr),
                'accuracy_percent':float(100*np.mean([int(r['correct']) for r in rr])),
                'macro_f1_percent':float(100*f1_score([r['gold'] for r in rr],[r['prediction'] for r in rr],labels=labels,average='macro',zero_division=0)),
                'invalid_percent':float(100*np.mean([r['prediction'] not in LABELS for r in rr])),
                'truncated_percent':float(100*np.mean([r.get('truncated',False) for r in rr]))})
            if method!='Ours_NS023':paired.append({'scope':scope,'method':method,'subset':subset,**paired_interval(rr,oo)})
        for h in range(2,11):
            rr=[r for r in rows if int(r['hops'])==h]
            depth.append({'scope':scope,'method':method,'hops':h,'n_predictions':len(rr),
                          'accuracy_percent':100*np.mean([int(r['correct']) for r in rr])})
        if scope=='llm_seed42' and method in ['cot','faithful','htt']:
            resources.append({'scope':scope,'method':method,'n_predictions':len(rows),
                'mean_input_tokens':float(np.mean([r['input_tokens'] for r in rows])),
                'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rows])),
                'generation_seconds':sum(r['batch_seconds']/r['batch_n'] for r in rows),
                'truncated_percent':float(100*np.mean([r['truncated'] for r in rows]))})
    def write(name,rows):
        with (DEST/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('metrics.csv',metrics);write('paired_intervals.csv',paired)
    write('by_depth.csv',depth);write('llm_resources.csv',resources)
    counts={f'{s}/{m}':sum(int(r['seed'])==s and r['method']==m for r in records) for s in [42,43,44] for m in ['cot','faithful','htt']}
    rawfiles=['ours_replayed.csv','epignn_predictions.csv','epignn_program_predictions.csv','payload/llm_results/predictions.jsonl']
    summary={'status':'COMPLETE_WITHIN_SEPARATE_SCOPES; FULL_LLM_3SEED_COMPARISON_PENDING',
        'test_history':'previously seen regression test; no fresh long-depth claim',
        'llm_seed_reason':'fixed execution order; seed42 completed before interruption, not selected by accuracy',
        'saved_llm_records':len(records),'pending_llm_records':9234-len(records),'saved_counts':counts,
        'metrics':metrics,'paired':paired,'resources':resources,
        'raw_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in rawfiles}}
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2))
    for r in metrics:
        if r['subset']=='long_4_10':print(r['scope'],r['method'],round(r['accuracy_percent'],3))
    for r in paired:
        if r['subset']=='long_4_10':print('DELTA',r['scope'],r['method'],*[round(r[k],3) for k in ['ours_minus_baseline_pp','ci95_low','ci95_high']])

if __name__=='__main__':main()
