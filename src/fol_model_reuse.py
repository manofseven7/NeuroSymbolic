"""Sound finite-model reuse for certification over interpretation families.

Two polarities: K & not Q, K & Q. A concrete model can establish SAT for
many interpretations; model failure never establishes UNSAT. Independent pure
Python AST evaluation checks reuse. UNSAT remains solver-relative.
"""
import itertools,time
import z3
from fol_certifier import Compiler,ambiguity_paths,replace_ops,certify
from fast_model_export import export_model
from finite_fol_model import verify_model

def family(k,q,max_sites=4):
    sites=[('k',p) for p in ambiguity_paths(k)]+[('q',p) for p in ambiguity_paths(q)];sites=sites[:max_sites];variants=[]
    for bits in itertools.product((0,1),repeat=len(sites)):
        changes={'k':{},'q':{}}
        for (part,path),bit in zip(sites,bits):
            a=k if part=='k' else q
            for j in path:a=a[j]
            changes[part][path]=('∨' if a[0]=='⊕' else '⊕') if bit else a[0]
        variants.append((bits,replace_ops(k,changes['k']),replace_ops(q,changes['q'])))
    return sites,variants

def parameterize(c,a,site_map,selectors,path=(),env=None):
    env={} if env is None else env;op=a[0]
    if op in ('atom','term','=','≠'):return c.compile(a,env)
    if op in ('∀','∃'):
        v=z3.Const('v_'+a[1],c.domain);body=parameterize(c,a[2],site_map,selectors,path+(2,),{**env,a[1]:v})
        return (z3.ForAll if op=='∀' else z3.Exists)([v],body)
    x=parameterize(c,a[1],site_map,selectors,path+(1,),env)
    if op=='¬':return z3.Not(x)
    y=parameterize(c,a[2],site_map,selectors,path+(2,),env)
    functions={'∧':z3.And,'∨':z3.Or,'⊕':z3.Xor,'→':z3.Implies,'↔':lambda x,y:x==y}
    original=functions[op](x,y)
    if path in site_map:
        flipped=functions['∨' if op=='⊕' else '⊕'](x,y)
        return z3.If(selectors[site_map[path]],flipped,original)
    return original

class CachedCompiler(Compiler):
    def __init__(self):super().__init__();self.cache={}
    def compile(self,a,bound=None):
        env={} if bound is None else bound;key=(a,tuple(sorted((n,v.sexpr()) for n,v in env.items())))
        if key not in self.cache:self.cache[key]=super().compile(a,env)
        return self.cache[key]

def run(k,q,method,max_sites=4,timeout=1500):
    started=time.perf_counter();sites,variants=family(k,q,max_sites);c=CachedCompiler();checks=0;reuse_hits=0;model_checks=0;models=[];evidence=[]
    shared=z3.Solver();shared.set(timeout=timeout,random_seed=0)
    def check(formula):
        nonlocal checks
        s=shared if method.startswith('incremental') else z3.Solver();s.set(timeout=timeout,random_seed=0);s.push();s.add(formula);status=s.check();checks+=1
        model=export_model(c,s.model()) if status==z3.sat and method.endswith('model_reuse') else None
        s.pop();return str(status),model
    states=[[None,None] for _ in variants]
    if method.startswith('selector') or method=='incremental_model_reuse':
        selectors=[z3.Bool('interpretation_'+str(j)) for j in range(len(sites))]
        kk=parameterize(c,k,{p:j for j,(part,p) in enumerate(sites) if part=='k'},selectors)
        qq=parameterize(c,q,{p:j for j,(part,p) in enumerate(sites) if part=='q'},selectors)
        global_status=[]
        for polarity in (0,1):
            status,model=check(z3.And(kk,qq if polarity else z3.Not(qq)));global_status.append(status)
            if status=='unsat':
                for st in states:st[polarity]='unsat'
            elif status=='unknown':
                # Definitive per-interpretation checks may still settle the family.
                pass
            elif model is not None:models.append((polarity,model))
        if global_status in (['unsat','sat'],['sat','unsat'],['unsat','unsat']):
            result='True' if global_status==['unsat','sat'] else 'False' if global_status==['sat','unsat'] else 'Abstain'
            return {'label':result,'certified':result!='Abstain','solver_checks':checks,'model_reuse_hits':0,'finite_model_checks':0,'branches':len(variants),'resolved_branches':0,'runtime_seconds':time.perf_counter()-started,'reuse_evidence':[]}
    def apply_models():
        nonlocal reuse_hits,model_checks
        for polarity,model in models:
            for i,(_,a,b) in enumerate(variants):
                if states[i][polarity] is None:
                    model_checks+=1
                    if verify_model(a,b,model,bool(polarity)):
                        states[i][polarity]='sat';reuse_hits+=1;evidence.append({'branch':i,'polarity':polarity,'model':model})
        models.clear()
    apply_models()
    def label(st):
        a,b=st
        if a==b=='unsat':return 'Inconsistent'
        if a=='unsat' and b=='sat':return 'True'
        if a=='sat' and b=='unsat':return 'False'
        if a==b=='sat':return 'Uncertain'
        return None
    early=method!='exhaustive'
    for i,(_,a,b) in enumerate(variants):
        kk,qq=c.compile(a),c.compile(b)
        for polarity in (0,1):
            if states[i][polarity] is None:
                status,model=check(z3.And(kk,qq if polarity else z3.Not(qq)));states[i][polarity]=status
                if model is not None:models.append((polarity,model));apply_models()
        known={label(st) for st in states}-{None,'Inconsistent'}
        if early and len(known)>1:break
    labels=[label(st) for st in states];known=set(labels)-{None,'Inconsistent'}
    certified=len(known)==1 and None not in labels
    result=next(iter(known)) if certified else 'Abstain'
    return {'label':result,'certified':certified,'solver_checks':checks,'model_reuse_hits':reuse_hits,'finite_model_checks':model_checks,'branches':len(variants),'resolved_branches':sum(x is not None for x in labels),'runtime_seconds':time.perf_counter()-started,'reuse_evidence':evidence}
