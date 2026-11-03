"""Compile solver model functions to Python, with exact per-atom fallback.

Avoid repeated solver API calls when enumerating finite relation tables. The
result is still independently checked by finite_fol_model against source ASTs.
"""
import itertools
import z3

def export_model(c,m):
    for v in c.constants.values():m.eval(v,model_completion=True)
    universe=m.get_universe(c.domain)
    if universe is None:universe=[m.eval(z3.Const('export_dummy',c.domain),model_completion=True)]
    universe=list(universe);index={str(v):i for i,v in enumerate(universe)};compiled={};active=set()
    def expr(a):
        if z3.is_true(a):return lambda args:True
        if z3.is_false(a):return lambda args:False
        if z3.is_var(a):
            j=z3.get_var_index(a);return lambda args:args[j]
        if str(a) in index:
            value=index[str(a)];return lambda args:value
        parts=[expr(x) for x in a.children()];kind=a.decl().kind()
        if kind==z3.Z3_OP_AND:return lambda args:all(f(args) for f in parts)
        if kind==z3.Z3_OP_OR:return lambda args:any(f(args) for f in parts)
        if kind==z3.Z3_OP_NOT:return lambda args:not parts[0](args)
        if kind==z3.Z3_OP_EQ:return lambda args:parts[0](args)==parts[1](args)
        if kind==z3.Z3_OP_DISTINCT:return lambda args:len(set(f(args) for f in parts))==len(parts)
        if kind==z3.Z3_OP_ITE:return lambda args:parts[1](args) if parts[0](args) else parts[2](args)
        if kind==z3.Z3_OP_IMPLIES:return lambda args:not parts[0](args) or parts[1](args)
        if kind==z3.Z3_OP_XOR:return lambda args:parts[0](args)!=parts[1](args)
        if kind==z3.Z3_OP_UNINTERPRETED:
            f=function(a.decl());return lambda args:f(tuple(p(args) for p in parts))
        raise ValueError('Unsupported model expression '+str(a))
    def value(a):
        if z3.is_true(a):return True
        if z3.is_false(a):return False
        return index[str(a)]
    def function(decl):
        name=decl.name()
        if name in compiled:return compiled[name]
        if name in active:raise ValueError('Recursive model definition')
        active.add(name);interp=m[decl]
        if interp is None:
            m.eval(decl(*([universe[0]]*decl.arity())),model_completion=True);interp=m[decl]
        entries={tuple(value(interp.entry(i).arg_value(j)) for j in range(decl.arity())):value(interp.entry(i).value()) for i in range(interp.num_entries())}
        fallback=expr(interp.else_value());f=lambda args:entries[args] if args in entries else fallback(args)
        compiled[name]=f;active.remove(name);return f
    constants={name:index[str(m.eval(v,model_completion=True))] for name,v in c.constants.items()};relations={}
    for name,(arity,f) in c.pred.items():
        try:ff=function(f)
        except (ValueError,KeyError,AttributeError,z3.Z3Exception):
            ff=lambda args,f=f:z3.is_true(m.eval(f(*[universe[v] for v in args]),model_completion=True))
        true=[list(values) for values in itertools.product(range(len(universe)),repeat=arity) if ff(values)]
        relations[name]={'arity':arity,'true_tuples':true}
    return {'domain_size':len(universe),'constants':constants,'relations':relations}
