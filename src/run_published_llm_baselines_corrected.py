"""Executed matched adaptations of CoT, Faithful CoT and HtT on one local LM.
Reuse author Faithful solver/table and HtT RuleLibrary + knowledge renderer.
No test annotations enter prompt construction. All outputs/checkpoints retained.
"""
import argparse,ast,hashlib,importlib.util,json,os,pickle,re,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch,yaml,jinja2
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM,AutoTokenizer
BASE=Path(__file__).resolve().parent
DATA=BASE
OUT=BASE/'llm_results';OUT.mkdir(exist_ok=True)
MODEL='Qwen/Qwen2.5-7B-Instruct'
LABELS=['father','mother','son','daughter','brother','sister','husband','wife','grandfather','grandmother','grandson','granddaughter','uncle','aunt','nephew','niece','father-in-law','mother-in-law','son-in-law','daughter-in-law','brother-in-law','sister-in-law']
LABEL_PATTERN=r'(?<![\w-])('+'|'.join(sorted(LABELS,key=len,reverse=True))+r')(?![\w-])'
def load(name):return json.loads((DATA/name).read_text())
def norm_answer(text):
 text=text.lower().replace('**','').strip();lines=[x for x in text.splitlines() if x.strip()]
 # Common final-line parser, independent of the gold answer.
 for line in reversed(lines):
  if re.search(r'final answer|therefore|answer is',line):
   hits=re.findall(LABEL_PATTERN,line)
   if hits:return hits[-1]
 hits=re.findall(LABEL_PATTERN,lines[-1] if lines else '')
 return hits[-1] if hits else '__INVALID__'
def chain(row):
 steps=ast.literal_eval(row['proof_state']);mapping={tuple(h):[tuple(x) for x in body] for step in steps for h,body in step.items()};q=ast.literal_eval(row['query']) if isinstance(row['query'],str) else row['query'];root=(q[0],row['target_text'],q[1])
 def expand(h):
  if h not in mapping:return [h]
  b=mapping[h]
  if len(b)==2 and not (b[0][0]==h[0] and b[0][2]==b[1][0] and b[1][2]==h[2]):b=b[::-1]
  return [x for child in b for x in expand(child)]
 return q,expand(root),mapping

def cot_answer(row):
 q,facts,mapping=chain(row);lines=[f'[{b}] is [{a}]\'s {r}.' for a,r,b in facts]
 lines.append(f'Final Answer: [{q[1]}] is [{q[0]}]\'s {row["target_text"]}.')
 return '\n'.join(lines)
def faithful_answer(row):
 q,facts,_=chain(row);lines=['# To answer this question, we answer the following subquestions:']
 for i,(a,r,b) in enumerate(reversed(facts),1):
  lines.append(f'# {i}. How is [{b}] related to [{a}]? (independent)')
  lines.append(f'relation({b}, {a}) = {r}')
 lines.append(f'# {len(facts)+1}. Final answer: How is [{q[1]}] related to [{q[0]}]? (depends on the subquestions)')
 lines.append(f'relation({q[1]}, {q[0]}) = '+ ' @ '.join(f'relation({b}, {a})' for a,r,b in reversed(facts)))
 return '\n'.join(lines)
def htt_answer(row):
 q,facts,mapping=chain(row);lines=[]
 for h,body in mapping.items():
  if len(body)!=2:continue
  b=body if body[0][0]==h[0] else body[::-1]
  lines.append(f"For {b[0][1]}'s {b[1][1]}, we have {b[0][1]}'s {b[1][1]} is {h[1]}.")
 lines.append('Therefore, the answer is '+row['target_text']+'.');return '\n'.join(lines)
def htt_text_answer(row):
 q,facts,_=chain(row)
 triplets=', '.join(f'({a}, {r}, {b})' for a,r,b in facts)
 lines=['We first extract triplets from the document, find the path, then reduce its relations.', 'The triplets include '+triplets+'.', 'The path from '+q[0]+' to '+q[1]+' is '+triplets+'.', 'The relations on the path are '+', '.join(r for a,r,b in facts)+'.', htt_answer(row)]
 return '\n'.join(lines)

def query(row):
 q=row['query'];return ast.literal_eval(q) if isinstance(q,str) else q

def build_prompt(method,story,q,demos,rules=None,training_path=None):
 examples=[]
 if method=='htt_train':
  for r in demos:
   qq,facts,_=chain(r);examples.append('Context: The relations on the path from '+qq[0]+' to '+qq[1]+' are '+', '.join(x[1] for x in facts)+'.\nQuestion: '+qq[1]+" is "+qq[0]+"'s what?\nAnswer:\n"+htt_answer(r))
  examples.append('Context: The relations on the path from '+q[0]+' to '+q[1]+' are '+', '.join(training_path)+'.\nQuestion: '+q[1]+" is "+q[0]+"'s what?\nAnswer:\n")
  return None,'\n\n'.join(examples)
 for r in demos:
  qq=query(r)
  if method=='faithful':
   examples.append('# Context: '+r['story']+'\n# Question: How is ['+qq[1]+'] related to ['+qq[0]+']?\n'+faithful_answer(r))
  elif method=='htt':
   examples.append('Document: '+r['story']+'\nQuestion: '+qq[1]+" is "+qq[0]+"'s what?\nAnswer:\n"+htt_text_answer(r))
  else:examples.append('Context: '+r['story']+'\nQuestion: How is ['+qq[1]+'] related to ['+qq[0]+']?\nAnswer:\n'+cot_answer(r))
 if method=='faithful':
  instruction='Follow the demonstrated relation(X, Y) = label format, where label describes X relative to Y. Write brief numbered natural-language subquestions and a relation assignment for each, following the examples. Order the path from the asked person to the reference person. Use no quotation marks around labels. An external solver composes them.\n'
  examples.append('# Context: '+story+'\n# Question: How is ['+q[1]+'] related to ['+q[0]+']?\n# To answer this question, write a program to answer the following subquestions:\n')
 elif method=='htt':
  instruction='Extract the kinship relations, find the path, and apply the provided rules. End with "Therefore, the answer is LABEL."\n';examples.append('Document: '+story+'\nQuestion: '+q[1]+" is "+q[0]+"'s what?\nAnswer:\n")
 else:
  instruction='Reason step by step about kinship. End with "Final Answer: [person] is [person]\\\'s LABEL."\n';examples.append('Context: '+story+'\nQuestion: How is ['+q[1]+'] related to ['+q[0]+']?\nAnswer:\n')
 system=None
 if method=='htt':system=jinja2.Template(yaml.safe_load((BASE/'upstream/htt/htt_test.yaml').read_text())['system']).render(rules=rules or [])
 return system,instruction+'\n\n'.join(examples)

class RestrictedPickle(pickle.Unpickler):
 def find_class(self,*args):raise pickle.UnpicklingError('globals prohibited')
def native_table():
 with (BASE/'upstream/faithful_cot/source/model/solver/CLUTRR/trans_rules.pkl').open('rb') as f:return RestrictedPickle(f).load()
def solve_faithful(text,table):
 # Author solver's relation extraction and reverse reduction retained exactly.
 relations=[]
 for line in text.split('\n')[1:]:
  if ' = ' in line and '@' not in line:relations.append(line.split(' = ')[1])
 if not relations:return '__INVALID__'
 final=''
 for relation in reversed(relations):
  if not final:final=relation
  elif (final,relation) in table:final=table[final,relation]
  else:return '__INVALID__'
 return final

def solve_author_faithful(text):
 root=BASE/'upstream/faithful_cot'
 spec=importlib.util.spec_from_file_location('author_faithful_solver',root/'source/model/solver/CLUTRR/CLUTRR_solver.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 previous=os.getcwd()
 try:
  os.chdir(root);answer=module.solve(text)
 finally:os.chdir(previous)
 return '__INVALID__' if answer=='[invalid]' else answer

def get_htt_library_class():
 path=BASE/'upstream/htt/source/prompts.py';spec=importlib.util.spec_from_file_location('htt_official_prompts',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.RuleLibrary

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 info=model_info(MODEL);revision=info.sha
 (OUT/'model_manifest.json').write_text(json.dumps({'model':MODEL,'revision':revision,'torch':torch.__version__,'dtype':'bfloat16','device':torch.cuda.get_device_name(0),'max_new_tokens':768,'greedy':True,'batch_size':128,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
 tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa');model.eval()
 def generate(prompts):
  formatted=[]
  for sys,p in prompts:
   messages=([{'role':'system','content':sys}] if sys else [])+[{'role':'user','content':p}];formatted.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
  tokenized=tokenizer(formatted,padding=True,return_tensors='pt',truncation=False)
  if tokenized.input_ids.shape[1]>16384:raise RuntimeError('Input exceeds fixed budget; do not silently truncate')
  tokenized={key:value.to('cuda') for key,value in tokenized.items()};start=time.monotonic()
  with torch.inference_mode():generated=model.generate(**tokenized,max_new_tokens=768,do_sample=False,pad_token_id=tokenizer.pad_token_id)
  torch.cuda.synchronize();elapsed=time.monotonic()-start;outputs=generated[:,tokenized['input_ids'].shape[1]:]
  decoded=tokenizer.batch_decode(outputs,skip_special_tokens=True)
  return [{'text':t,'input_tokens':int(tokenized['attention_mask'][i].sum()),'output_tokens':int((outputs[i]!=tokenizer.pad_token_id).sum()),'truncated':bool(outputs.shape[1]==768 and int(outputs[i,-1])!=tokenizer.eos_token_id),'batch_seconds':elapsed,'batch_n':len(prompts)} for i,t in enumerate(decoded)]
 tests=load('test_inputs.json');gold=load('evaluation_labels.json');nativetable=native_table();RuleLibrary=get_htt_library_class()
 path=OUT/('smoke_predictions.jsonl' if args.smoke else 'predictions.jsonl');done=set()
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);done.add((r['seed'],r['method'],r['id']))
 log=path.open('a',buffering=1)
 for seed in ([42] if args.smoke else [42,43,44]):
  train=load(f'train_{seed}.json');demos=train[:5];torch.manual_seed(seed)
  (OUT/f'demonstrations_{seed}.json').write_text(json.dumps([r['id'] for r in demos]));table={(a,b):c for a,b,c in load(f'rules_{seed}.json')};library=RuleLibrary();trainpath=OUT/f'htt_training_{seed}.jsonl';training_done={}
  if trainpath.exists():
   for line in trainpath.read_text().splitlines():
    r=json.loads(line);training_done[r['id']]=r;library.update(r['rules'],r['correct'])
  remaining=[r for r in train if r['id'] not in training_done]
  if args.smoke:remaining=remaining[:8]
  with trainpath.open('a',buffering=1) as f:
   for start in range(0,len(remaining),128):
    batch=remaining[start:start+128];prompts=[]
    for r in batch:
     q,facts,_=chain(r);prompts.append(build_prompt('htt_train','',q,demos,training_path=[x[1] for x in facts]))
    for r,result in zip(batch,generate(prompts)):
     rules=re.findall(r"[a-z\-]+'s [a-z\-]+ is [a-z\-]+\.",result['text']);correct=int(norm_answer(result['text'])==r['target_text']);library.update(rules,correct);f.write(json.dumps({'id':r['id'],'rules':rules,'correct':correct,**result})+'\n')
    print('HTT_TRAIN',seed,start+len(batch),'/',len(remaining),flush=True)
  library.save(str(OUT/f'htt_library_{seed}.yaml'));rules=library.to_prompt(min_coverage=2,min_confidence=.7)
  (OUT/f'htt_retained_rules_{seed}.json').write_text(json.dumps(rules));print('HTT_RETAINED',seed,len(rules),flush=True)
  cases=load(f'gate_{seed}.json')[:8] if args.smoke else tests
  for method in ['cot','faithful','htt']:
   remaining=[r for r in cases if (seed,method,r['id']) not in done]
   for start in range(0,len(remaining),128):
    batch=remaining[start:start+128];prompts=[build_prompt(method,r['story'],query(r),demos,rules) for r in batch]
    for r,result in zip(batch,generate(prompts)):
     normalized='# Chat completion header\n'+'\n'.join(line.strip() for line in result['text'].splitlines())
     pred=solve_author_faithful(normalized) if method=='faithful' else norm_answer(result['text']);matched=solve_faithful(normalized,table) if method=='faithful' else None
     truth=r['target'] if args.smoke else gold[r['id']]
     record={'id':r['id'],'seed':seed,'method':method,'hops':r['hops'],'prediction':pred,'correct':int(pred==truth),'gold':truth,**result}
     if matched is not None:record.update({'matched_prediction':matched,'matched_correct':int(matched==truth)})
     log.write(json.dumps(record)+'\n')
    print('TEST',seed,method,start+len(batch),'/',len(remaining),'last_batch_seconds',round(result['batch_seconds'],2),flush=True)
 log.close();print('COMPLETE_SMOKE' if args.smoke else 'COMPLETE_FULL',flush=True)
if __name__=='__main__':main()
