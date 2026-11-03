"""Sound must/may bounds over logical interpretations, with exact fallback.

For every enumerated interpretation theta: lower(F) => F_theta => upper(F).
Quantification preserves sound bounds, but not necessarily tightness: selector
choices are shared across all entities. No finite-domain approximation is used.
This implements a sufficient certificate, not a complete replacement for FOL.
"""
import time
import z3
from fol_certifier import ambiguity_paths,Compiler
from fol_model_reuse import run as exact_run,CachedCompiler

def bounds(c,a,sites,path=(),env=None):
    env={} if env is None else env;op=a[0]
    if not any(p[:len(path)]==path for p in sites):
        f=c.compile(a,env);return f,f
    if op in ('atom','term','=','≠'):
        f=c.compile(a,env);return f,f
    if op in ('∀','∃'):
        v=z3.Const('v_'+a[1],c.domain);lo,hi=bounds(c,a[2],sites,path+(2,),{**env,a[1]:v});quant=z3.ForAll if op=='∀' else z3.Exists
        return quant([v],lo),quant([v],hi)
    al,ah=bounds(c,a[1],sites,path+(1,),env)
    if op=='¬':return z3.Not(ah),z3.Not(al)
    bl,bh=bounds(c,a[2],sites,path+(2,),env)
    if op in ('⊕','↔') or path in sites:
        if al.eq(ah) and bl.eq(bh):xorlo=xorhi=z3.Xor(al,bl)
        else:
            xorlo=z3.Or(z3.And(al,z3.Not(bh)),z3.And(z3.Not(ah),bl))
            xorhi=z3.Or(z3.And(ah,z3.Not(bl)),z3.And(z3.Not(al),bh))
    if path in sites:return xorlo,z3.Or(ah,bh)
    if op=='∧':return z3.And(al,bl),z3.And(ah,bh)
    if op=='∨':return z3.Or(al,bl),z3.Or(ah,bh)
    if op=='⊕':return xorlo,xorhi
    if op=='→':return z3.Implies(ah,bl),z3.Implies(al,bh)
    if op=='↔':return z3.Not(xorhi),z3.Not(xorlo)
    raise ValueError(op)

def run(k,q,max_sites=4,timeout=1500):
    start=time.perf_counter();sites=[('k',p) for p in ambiguity_paths(k)]+[('q',p) for p in ambiguity_paths(q)];sites=sites[:max_sites]
    if not sites:
        got=exact_run(k,q,'incremental_early_stop',max_sites=max_sites,timeout=timeout);got['runtime_seconds']=time.perf_counter()-start;got['certificate_kind']='no_ambiguity_exact';got['obligations']=[];return got
    c=CachedCompiler()
    kl,kh=bounds(c,k,{p for part,p in sites if part=='k'});ql,qh=bounds(c,q,{p for part,p in sites if part=='q'})
    solver=z3.Solver();solver.set(timeout=timeout,random_seed=0);cache={};checks=0;trace=[]
    def check(formula,name):
        nonlocal checks
        formula=z3.simplify(formula);key=formula.sexpr()
        if key in cache:status=cache[key];cached=True
        else:
            solver.push();solver.add(formula);status=str(solver.check());solver.pop();cache[key]=status;checks+=1;cached=False
        trace.append({'obligation':name,'status':status,'cached':cached});return status
    def done(label,kind,extra=None):
        extra={} if extra is None else extra
        return {'label':label,'certified':label!='Abstain','solver_checks':checks+extra.get('solver_checks',0),'model_reuse_hits':0,'finite_model_checks':0,'branches':2**len(sites),'resolved_branches':0,'runtime_seconds':time.perf_counter()-start,'reuse_evidence':[],'certificate_kind':kind,'obligations':trace}
    pos=check(z3.And(kl,ql),'common_positive_witness');neg=check(z3.And(kl,z3.Not(qh)),'common_negative_witness')
    if pos==neg=='sat':return done('Uncertain','two_common_models')
    if pos=='sat' and check(z3.And(kh,z3.Not(ql)),'possible_countermodel')=='unsat':return done('True','positive_model_and_upper_refutation')
    if neg=='sat' and check(z3.And(kh,qh),'possible_positive_witness')=='unsat':return done('False','negative_model_and_upper_refutation')
    fallback=exact_run(k,q,'incremental_early_stop',max_sites=max_sites,timeout=timeout)
    return done(fallback['label'],'exact_fallback',fallback)
