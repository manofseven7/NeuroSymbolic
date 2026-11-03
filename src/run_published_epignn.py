"""NS024: official EpiGNN2025 with NS023 shared frozen text grounder.
The pinned author's model is unmodified. Missing unused imports get fail-fast
placeholders; nodes are relabelled to insertion order required by its path helper.
"""
import sys,json,time,random,copy,csv,argparse,ast,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/published_baselines_v1';DATA=OUT/'payload';UP=DATA/'upstream/epignn';sys.path.insert(0,str(UP))
import numpy as np
import torch
from torch_geometric.data import HeteroData,Batch
import src.utils as _author_utils
def _unused(*args,**kwargs):raise RuntimeError('Unimplemented upstream helper unexpectedly invoked')
for name in ['get_all_source_sink_paths_from_edge_index','get_sizes_to_unbatch_edge_index']:
 if not hasattr(_author_utils,name):setattr(_author_utils,name,_unused)
from src.model_nbf_fb import NBFdistR
LABELS=['father','mother','son','daughter','brother','sister','husband','wife','grandfather','grandmother','grandson','granddaughter','uncle','aunt','nephew','niece','father-in-law','mother-in-law','son-in-law','daughter-in-law','brother-in-law','sister-in-law'];INDEX={r:i for i,r in enumerate(LABELS)}
def load(n):return json.loads((DATA/n).read_text())
def connected(g):
 known={g['query'][0]}
 for _ in range(g['n_nodes']):
  nxt=known|{b for a,b,r in g['edges'] if a in known}|{a for a,b,r in g['edges'] if b in known}
  if nxt==known:break
  known=nxt
 return bool(g['edges']) and g['query'][1] in known
def make_data(g,training=False):
 ns=list(dict.fromkeys(v for a,b,r in g['edges'] for v in [a,b]));ni={v:i for i,v in enumerate(ns)}
 ei=torch.tensor([[ni[a],ni[b]] for a,b,r in g['edges']],dtype=torch.long).T;et=torch.tensor([INDEX[r] for a,b,r in g['edges']],dtype=torch.long);q=torch.tensor([[ni[g['query'][0]]],[ni[g['query'][1]]]],dtype=torch.long)
 return HeteroData(fw={'x':torch.arange(len(ns)).unsqueeze(1)},bw={'x':torch.arange(len(ns)).unsqueeze(1)},fw__rel__fw={'edge_index':ei,'edge_type':et,'target_edge_index':q,'target_edge_type':torch.tensor([INDEX[g['target']] if training else 0])},bw__rel__bw={'edge_index':ei.flip(0),'edge_type':et})
def predict(model,graphs):
 model.eval();out=['__ABSTAIN__']*len(graphs);valid=[i for i,g in enumerate(graphs) if connected(g)]
 with torch.inference_mode():
  for start in range(0,len(valid),128):
   ids=valid[start:start+128];batch=Batch.from_data_list([make_data(graphs[i]) for i in ids]);logits,_=model(batch)
   for i,p in zip(ids,logits.argmax(dim=-1).tolist()):out[i]=LABELS[p]
 return out
def expand_program_queries(seed,graphs):
 # Give the baseline the intermediate/leaf supervision already available to ours.
 from consistent_kinship import PAIRS,TYPE,INVERSE,training_genders
 raw={r['id']:r for r in load(f'train_{seed}.json')};expanded=[]
 for g in graphs:
  r=raw[g['id']];ns=list(dict.fromkeys(re.findall(r'\[([^\]]+)\]',r['story'])));ni={n:i for i,n in enumerate(ns)};sex=training_genders(r);steps=ast.literal_eval(r['proof_state']);atoms=set()
  for step in steps:
   for h,body in step.items():atoms.add(tuple(h));atoms.update(tuple(a) for a in body)
  targets={(a,rel,b) for a,rel,b in atoms}
  for a,rel,b in atoms:
   if rel in TYPE and a in sex:targets.add((b,PAIRS[INVERSE[TYPE[rel]]][sex[a]],a))
  for a,rel,b in sorted(targets):
   if a!=b and rel in INDEX:expanded.append({**g,'query':[ni[a],ni[b]],'target':rel})
 return expanded

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--program-supervision',action='store_true');args=parser.parse_args();suffix='_program' if args.program_supervision else ''
 torch.set_num_threads(1);results=[];start=time.monotonic()
 for seed in [42,43,44]:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);train=[g for g in load(f'graphs_train_{seed}.json') if connected(g)];gate=load(f'graphs_gate_{seed}.json');test=load(f'graphs_test_{seed}.json')
  if args.program_supervision:train=expand_program_queries(seed,train)
  print('TRAIN_QUERIES',seed,len(train),flush=True)
  model=NBFdistR(hidden_dim=64,num_relations=len(LABELS),shared=True,use_mlp_classifier=False,dist='euclidean',num_layers=9,residual=False,eval_mode=True,facets=8,aggr_type='mul',ablate_compose=False,ablate_probas=False)
  opt=torch.optim.Adam(model.parameters(),lr=.01);best=-1;history=[]
  for epoch in range(100):
   model.train();order=torch.randperm(len(train)).tolist();losses=[]
   for offset in range(0,len(order),128):
    batch=Batch.from_data_list([make_data(train[i],True) for i in order[offset:offset+128]]);opt.zero_grad();logits,_=model(batch);loss=torch.nn.functional.cross_entropy(logits,batch['fw','rel','fw'].target_edge_type);loss.backward();opt.step();losses.append(float(loss.detach()))
   pred=predict(model,gate);correct=sum(p==g['target'] for p,g in zip(pred,gate));history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'gate_correct':correct})
   if correct>best:best=correct;state=copy.deepcopy(model.state_dict());bestepoch=epoch+1
   if epoch%10==0:print('EPIGNN',seed,'epoch',epoch+1,'gate',correct,'best',best,flush=True)
  model.load_state_dict(state);torch.save(state,OUT/f'epignn{suffix}_{seed}.pt');(OUT/f'epignn{suffix}_training_{seed}.json').write_text(json.dumps({'selected_epoch':bestepoch,'best_gate_correct':best,'history':history,'train_n':len(train),'configuration':'author64hidden8facets9rounds100epochsAdam.01batch128; gate selection only'},indent=2));pred=predict(model,test)
  for p,g in zip(pred,test):results.append({'id':g['id'],'seed':seed,'method':'EpiGNN2025_shared_grounder'+suffix,'hops':g['hops'],'prediction':p,'gold':g['target'],'correct':int(p==g['target'])})
  with (OUT/f'epignn{suffix}_predictions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
  print('EPIGNN_TEST',seed,sum(p==g['target'] for p,g in zip(pred,test)),len(test),'seconds',time.monotonic()-start,flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
