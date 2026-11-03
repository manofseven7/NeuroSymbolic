"""Dependency-free verification of finite relational first-order structures.

The producer may be Z3; this checker neither imports nor calls a solver.
Input AST follows fol_certifier's declarative grammar, all predicates extensional.
"""

def evaluate(a,model,env=None):
    env={} if env is None else env;op=a[0]
    def term(name):return env[name] if name in env else model['constants'][name]
    if op=='atom':return [term(x) for x in a[2]] in model['relations'][a[1]]['true_tuples']
    if op in ('=','≠'):
        eq=term(a[1][1])==term(a[2][1]);return eq if op=='=' else not eq
    if op=='¬':return not evaluate(a[1],model,env)
    if op in ('∀','∃'):
        values=(evaluate(a[2],model,{**env,a[1]:v}) for v in range(model['domain_size']))
        return all(values) if op=='∀' else any(values)
    x=evaluate(a[1],model,env);y=evaluate(a[2],model,env)
    if op=='∧':return x and y
    if op=='∨':return x or y
    if op=='⊕':return x!=y
    if op=='→':return not x or y
    if op=='↔':return x==y
    raise ValueError('Invalid AST '+str(op))

def verify_model(theory,question,model,question_value):
    n=model['domain_size']
    if not isinstance(n,int) or n<1:return False
    if any(not isinstance(v,int) or not 0<=v<n for v in model['constants'].values()):return False
    for r in model['relations'].values():
        if any(len(t)!=r['arity'] or any(not isinstance(v,int) or not 0<=v<n for v in t) for t in r['true_tuples']):return False
    try:return evaluate(theory,model) and evaluate(question,model)==question_value
    except (KeyError,IndexError,TypeError,ValueError):return False
