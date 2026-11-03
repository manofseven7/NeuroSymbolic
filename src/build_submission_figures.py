"""Build publication vectors from saved results; never performs inference or training."""
from pathlib import Path
import json, csv, hashlib, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/review_version/figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';blue='#287fa3';teal='#128477';orange='#b86a25';grey='#5e6873'
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off');return f,a
def box(a,x,y,w,h,title,body,color=blue,fs=9):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',facecolor=color+'12',edgecolor=color,lw=1.25))
 a.text(x+w/2,y+h*.72,title,ha='center',va='center',weight='bold',color=navy,fontsize=fs+1)
 a.text(x+w/2,y+h*.34,body,ha='center',va='center',fontsize=fs,color=navy,linespacing=1.45)
def arrow(a,p,q,color=grey,dashed=False):
 a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,color=color,lw=1.2,linestyle='--' if dashed else '-'))
def save(f,name):
 for ext in ['pdf','svg','png']:
  stream=io.BytesIO()
  f.savefig(stream,format=ext,bbox_inches='tight',pad_inches=.09,dpi=240)
  raw=stream.getvalue()
  if ext=='pdf':assert raw.rstrip().endswith(b'%%EOF'),name
  (OUT/f'{name}.{ext}').write_bytes(raw)
 plt.close(f)
f,a=canvas(10.8,5.7)
a.text(.015,.975,'INPUT-SPECIFIC FRONT ENDS',weight='bold',color=grey,fontsize=10)
a.text(.565,.975,'AUDITABLE REASONING INTERFACE',weight='bold',color=grey,fontsize=10)
for y,t,b in [(.70,'Supplied formulas','Curated FOLIO\nscoped OR / XOR choices'),(.40,'Controlled English','Frozen token-role parser\nranked program beam'),(.10,'Relational stories','Frozen kinship grounder\nordered-pair alternatives')]:
 box(a,.02,y,.205,.19,t,b,fs=8.7)
arrow(a,(.234,.795),(.28,.795));arrow(a,(.234,.495),(.28,.495));arrow(a,(.234,.195),(.28,.195))
box(a,.29,.70,.215,.19,'FOL adapter','Must / may formulas\nZ3 + exact fallback',blue,8.7)
box(a,.29,.40,.215,.19,'Signed-Horn adapter','Intersection / union\nclosure + exact fallback',blue,8.7)
box(a,.29,.10,.215,.19,'Kinship adapter','Bounded path composition\ncommon / possible edges',blue,8.7)
box(a,.58,.53,.20,.26,'Family decision','Uniform answer or abstain\nexplicit unresolved state',teal,8.7)
box(a,.58,.13,.20,.26,'Selected execution','Preferred answer\nprogram-relative evidence',orange,8.7)
for y in [.795,.495,.195]:
 arrow(a,(.515,y),(.57,.65),teal)
 arrow(a,(.515,y),(.57,.26),orange,True)
box(a,.84,.32,.14,.31,'Report','Prediction\nexecuted evidence\ncertificate status',navy,8.5)
arrow(a,(.79,.65),(.83,.55),teal);arrow(a,(.79,.26),(.83,.40),orange)
a.text(.01,.015,'A family certificate concerns retained interpretations.  A displayed proof concerns the selected program.',fontsize=9,color=grey)
save(f,'system_architecture')
f,a=canvas(10.8,5.6)
a.axhspan(.53,1,color='#eef4f8');a.axhspan(0,.46,color='#f0f7f4')
a.text(.02,.94,'TRAINING AND ADMISSION ONLY',weight='bold',color=navy,fontsize=11)
box(a,.025,.63,.21,.22,'Training programs','256 stories / seed\npositive + inverse examples',blue,8.7)
box(a,.285,.63,.21,.22,'Learn local structure','Pair-relation grounder\nrelation types + chain rules',blue,8.7)
box(a,.545,.63,.21,.22,'Admission validation','128 separate stories\nselect a coherent variant',blue,8.7)
box(a,.805,.63,.17,.22,'Freeze','Model + rules\nseeds 42, 43, 44',navy,8.7)
for x,y in [(.244,.276),(.504,.536),(.765,.796)]:arrow(a,(x,.74),(y,.74))
a.text(.02,.515,'NO TEST LABELS, PROOF PROGRAMS OR GENDERS ENTER INFERENCE',weight='bold',color=grey,fontsize=9.5)
a.text(.02,.405,'INFERENCE ON FROZEN INPUTS',weight='bold',color=teal,fontsize=11)
box(a,.025,.10,.21,.23,'Text + query pair','Entity brackets supplied\nlocal ordered-pair features',teal,8.7)
box(a,.285,.10,.21,.23,'Ground and compose','Directed / reciprocal edges\n64-path cap; binary bracketings',teal,8.4)
box(a,.545,.10,.21,.23,'Retain alternatives','Up to 16 programs\nbounds + exact fallback',teal,8.7)
box(a,.805,.10,.17,.23,'Separate outputs','Preferred relation\nproof + family status',teal,8.7)
for x,y in [(.244,.276),(.504,.536),(.765,.796)]:arrow(a,(x,.21),(y,.21),teal)
a.plot([.89,.89,.39],[.62,.455,.455],color=navy,lw=1.2,ls='--')
arrow(a,(.39,.455),(.39,.34),navy,True)
save(f,'kinship_pipeline')
# Values are recomputed from the archived decisions, with all seeds retained per story.
p=ROOT/'results/unseen_robustness_v1'
rows=[json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()]
cot=[json.loads(s) for s in (p/'cot_predictions.jsonl').read_text().splitlines()]

held=json.loads((p/'heldout.json').read_text());tasks={r['id']:r['task_name'] for r in held}
methods=['NS023','NS022','direct','cot'];names=['Proposed','Previous adapter','Direct classifier','CoT / Qwen-7B'];colors=[teal,blue,grey,orange]
allrows=rows+cot
stats=[]
for m in methods:
 for task in ['all','task_1.2','task_1.3','task_2.3','task_3.3','task_4.3']:
  rr=[r for r in allrows if r['method']==m and (task=='all' or tasks[r['id']]==task)]
  assert len(rr)==(558 if task=='all' else (78 if task=='task_1.2' else 120))
  assert len({(r['seed'],r['id']) for r in rr})==len(rr)
  assert all(int(r['correct'])==int(r['prediction']==r['gold']) for r in rr)
  stats.append({'method':m,'task':task,'n':len(rr),'accuracy':100*np.mean([int(r['correct']) for r in rr])})
f,ax=plt.subplots(1,2,figsize=(11.4,4.5),gridspec_kw={'width_ratios':[1.18,1]})
x=np.arange(5);width=.18
for i,(m,n,c) in enumerate(zip(methods,names,colors)):
 yy=[r['accuracy'] for r in stats if r['method']==m and r['task']!='all']
 ax[0].bar(x+(i-1.5)*width,yy,width,color=c,label=n)
ax[0].set_xticks(x,['Clean 2\n$n=26$','Clean 3\n$n=40$','Support 3\n$n=40$','Irrelevant 3\n$n=40$','Disconnect 3\n$n=40$'],fontsize=8.4)
ax[0].set_ylabel('Preferred-answer accuracy (%)');ax[0].set_ylim(0,70);ax[0].set_title('(a) Untouched short-depth robustness',fontsize=11,loc='left');ax[0].legend(ncol=2,fontsize=8.2,frameon=False,loc='upper left');ax[0].grid(axis='y',alpha=.15);ax[0].set_axisbelow(True)
paired=list(csv.DictReader((ROOT/'results/published_baselines_v1/paired_story_bootstrap.csv').open()))
reg=next(r for r in paired if r['baseline']=='cot' and r['subset']=='long_4_10')
summ=json.loads((p/'summary.json').read_text());fresh=next(r for r in summ['paired'] if r['baseline']=='cot')
vals=[float(reg['ours_minus_baseline_pp']),fresh['delta_pp']];low=[float(reg['ci95_low']),fresh['ci95'][0]];high=[float(reg['ci95_high']),fresh['ci95'][1]]
for y,v,l,h,c in zip([1,0],vals,low,high,[grey,teal]):
 ax[1].errorbar(v,y,xerr=[[v-l],[h-v]],fmt='o',capsize=5,ms=7,color=c)
 ax[1].text(v,y+.13,f'{v:.2f} [{l:.2f}, {h:.2f}]',ha='center',fontsize=9,color=c)
ax[1].axvline(0,color=grey,lw=1,ls=':');ax[1].set_xlim(-1,38);ax[1].set_ylim(-.5,1.5)
ax[1].set_yticks([1,0],['Seen regression\n903 stories; 4–10 hops','Untouched robustness\n186 stories; 2–3 hops'],fontsize=9)
ax[1].set_xlabel('Proposed minus CoT (percentage points)');ax[1].set_title('(b) Paired story-bootstrap 95% intervals',fontsize=11,loc='left');ax[1].grid(axis='x',alpha=.15)
f.tight_layout(w_pad=2)
save(f,'robustness_confirmation')
inputs=[p/'predictions.jsonl',p/'cot_predictions.jsonl',p/'heldout.json',p/'summary.json',ROOT/'results/published_baselines_v1/paired_story_bootstrap.csv']
report={'analysis':'Descriptive publication figures only; no new experimental decisions','input_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in inputs},'robustness_metrics':stats,'figures':['system_architecture','kinship_pipeline','robustness_confirmation']}
(ROOT/'reports/submission_v18/figure_provenance.json').write_text(json.dumps(report,indent=2))
