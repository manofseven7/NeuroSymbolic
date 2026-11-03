"""Analyze saved NS024 predictions; no training or prompt selection here."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/published_baselines_v1'

def read_csv(name):
    return list(csv.DictReader((OUT/name).open()))

def analyze(require_complete=True):
    ours=read_csv('ours_replayed.csv')
    groups={'Ours_NS023':ours,
            'EpiGNN2025_shared_grounder':read_csv('epignn_predictions.csv'),
            'EpiGNN2025_shared_grounder_program':read_csv('epignn_program_predictions.csv')}
    llm=OUT/'payload/llm_results/predictions.jsonl'
    records=[json.loads(x) for x in llm.read_text().splitlines()] if llm.exists() else []
    for method in ['cot','faithful','htt']:
        rows=[r for r in records if r['method']==method]
        if rows:
            groups[method]=rows
            if method=='faithful':
                groups['faithful_training_only']=[{**r,'prediction':r['matched_prediction'],
                    'correct':r['matched_correct']} for r in rows]
    if require_complete:
        assert all(m in groups for m in ['cot','faithful','htt']), 'LLM runs incomplete'
    expected={(int(r['seed']),r['id']) for r in ours}
    assert len(expected)==3078
    labels=sorted({r['target_text'] for seed in [42,43,44] for r in
        json.loads((OUT/f'payload/train_{seed}.json').read_text())})
    prediction_labels=set(labels)|{'husband','wife','brother-in-law','sister-in-law'}
    metrics=[]
    paired=[]
    by_hop=[]
    ours_map={(int(r['seed']),r['id']):r for r in ours}
    for method,rows in groups.items():
        keys=[(int(r['seed']),r['id']) for r in rows]
        if set(keys)!=expected or len(keys)!=len(expected):
            if require_complete:raise AssertionError((method,len(keys),'missing or duplicate predictions'))
            continue
        for r in rows:
            assert r['gold']==ours_map[int(r['seed']),r['id']]['gold']
            assert int(r['correct'])==int(r['prediction']==r['gold'])
        for subset in ['all','short_2_3','long_4_10']:
            rr=[r for r in rows if subset=='all' or (int(r['hops'])>=4)==(subset=='long_4_10')]
            metrics.append({'method':method,'subset':subset,'n_predictions':len(rr),
                'n_questions':len({r['id'] for r in rr}),'correct':sum(int(r['correct']) for r in rr),
                'accuracy_percent':100*np.mean([int(r['correct']) for r in rr]),
                'macro_f1_percent':100*f1_score([r['gold'] for r in rr],
                    [r['prediction'] for r in rr],labels=labels,average='macro',zero_division=0),
                'invalid_percent':100*np.mean([r['prediction'] not in prediction_labels for r in rr]),
                'truncated_percent':100*np.mean([r.get('truncated',False) for r in rr])})
            if method!='Ours_NS023':
                # Resample stories, keeping all three seeds paired within each story.
                ids=sorted({r['id'] for r in rr})
                delta={i:[] for i in ids}
                for r in rr:
                    o=ours_map[int(r['seed']),r['id']]
                    delta[r['id']].append(int(o['correct'])-int(r['correct']))
                assert all(len(v)==3 for v in delta.values())
                d=np.array([np.mean(delta[i]) for i in ids])
                rng=np.random.default_rng(24024)
                bs=np.concatenate([rng.choice(d,(1000,len(d)),replace=True).mean(1) for _ in range(10)])*100
                lo,hi=np.quantile(bs,[.025,.975])
                paired.append({'baseline':method,'subset':subset,'ours_minus_baseline_pp':float(100*d.mean()),
                    'ci95_low':float(lo),'ci95_high':float(hi),'bootstrap_story_n':len(ids),'replicates':10000})
        for seed in [42,43,44]:
            for hop in sorted({int(r['hops']) for r in rows}):
                rr=[r for r in rows if int(r['seed'])==seed and int(r['hops'])==hop]
                by_hop.append({'method':method,'seed':seed,'hops':hop,'n':len(rr),
                    'correct':sum(int(r['correct']) for r in rr),'accuracy_percent':100*np.mean([int(r['correct']) for r in rr])})
    def write(name,rows):
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('comparison_metrics.csv',metrics)
    write('paired_story_bootstrap.csv',paired)
    write('comparison_by_hop_seed.csv',by_hop)
    resources=[]
    for method in ['cot','faithful','htt']:
        rr=[r for r in records if r['method']==method]
        if len(rr)==3078:
            resources.append({'method':method,'n_predictions':len(rr),
                'mean_input_tokens':float(np.mean([r['input_tokens'] for r in rr])),
                'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rr])),
                'generation_seconds':sum(r['batch_seconds']/r['batch_n'] for r in rr),
                'truncated_percent':100*np.mean([r['truncated'] for r in rr]),
                'hardware':'RTX PRO6000 Blackwell; BF16; transformers4.51.3; batch128'})
    if resources:write('llm_generation_resources.csv',resources)
    (OUT/'comparison_summary.json').write_text(json.dumps({'complete':require_complete,
        'questions_previously_seen':True,'metrics':metrics,'paired':paired},indent=2))
    for r in metrics:
        if r['subset']=='long_4_10':print(r['method'],round(r['accuracy_percent'],3))
    return metrics,paired

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    analyze(not a.partial)
