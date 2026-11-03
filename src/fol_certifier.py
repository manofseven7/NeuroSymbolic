"""Strict FOL front end and solver-relative evidence. No natural-language parser.

One nonempty uninterpreted domain, no unique names or closed-world assumption.
Unknown and inconsistency are never mapped to an NLI answer. Unsat cores are
premise support sets, NOT natural-language or independently checked proof traces.
"""
import re
import z3

class FOLParseError(ValueError): pass
TOKEN=re.compile(r'\s*([∀∃¬∧∨⊕→↔=≠(),]|\w+)')
PREC={'↔':1,'→':2,'⊕':3,'∨':4,'∧':5,'=':6,'≠':6}

class Parser:
    def __init__(self,text):
        self.tokens=[];pos=0
        while pos<len(text):
            if not text[pos:].strip():break
            m=TOKEN.match(text,pos)
            if not m:raise FOLParseError(f'Unexpected character at {pos}: {text[pos:pos+25]}')
            self.tokens.append(m[1]);pos=m.end()
        self.i=0
    def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else None
    def take(self,expected=None):
        t=self.peek()
        if t is None or (expected is not None and t!=expected):raise FOLParseError(f'Expected {expected}; found {t}')
        self.i+=1;return t
    def expr(self,level=0):
        left=self.unary()
        while self.peek() in PREC and PREC[self.peek()]>=level:
            op=self.take();p=PREC[op]
            right=self.expr(p if op=='→' else p+1)
            left=(op,left,right)
        return left
    def unary(self):
        t=self.take()
        if t=='¬':return ('¬',self.unary())
        if t in ('∀','∃'):
            v=self.take()
            if not (v.isidentifier() and v[0].islower()):raise FOLParseError('Invalid variable '+v)
            return (t,v,self.unary())
        if t=='(':
            a=self.expr();self.take(')');return a
        if not re.fullmatch(r'\w+',t):raise FOLParseError('Expected symbol '+t)
        if self.peek()!='(':return ('term',t)
        self.take('(');args=[]
        if self.peek()!=')':
            while True:
                a=self.take()
                if not re.fullmatch(r'\w+',a):raise FOLParseError('Invalid argument '+a)
                args.append(a)
                if self.peek()!=',':break
                self.take(',')
        self.take(')');return ('atom',t,tuple(args))

def parse(text):
    p=Parser(text);a=p.expr()
    if p.peek() is not None:raise FOLParseError('Trailing input '+str(p.tokens[p.i:]))
    return a

def conjuncts(a):
    return conjuncts(a[1])+conjuncts(a[2]) if a[0]=='∧' else [a]

class Compiler:
    def __init__(self):
        self.domain=z3.DeclareSort('Entity');self.pred={};self.constants={}
    def term(self,t,bound):
        if t in bound:return bound[t]
        if t in {'x','y','z','u','v','w','x1','x2','y1','y2','z1','z2'}:raise FOLParseError('Free variable '+t)
        if t not in self.constants:self.constants[t]=z3.Const('c_'+t,self.domain)
        return self.constants[t]
    def compile(self,a,bound=None):
        bound={} if bound is None else bound;op=a[0]
        if op=='atom':
            _,name,args=a
            if name in self.pred and self.pred[name][0]!=len(args):raise FOLParseError('Arity conflict '+name)
            if name not in self.pred:self.pred[name]=(len(args),z3.Function('p_'+name,*([self.domain]*len(args)),z3.BoolSort()))
            return self.pred[name][1](*[self.term(t,bound) for t in args])
        if op=='term':raise FOLParseError('Bare term is not a formula '+a[1])
        if op in ('∀','∃'):
            v=z3.Const('v_'+a[1],self.domain);env={**bound,a[1]:v};body=self.compile(a[2],env)
            return (z3.ForAll if op=='∀' else z3.Exists)([v],body)
        if op=='¬':return z3.Not(self.compile(a[1],bound))
        if op in ('=','≠'):
            if a[1][0]!='term' or a[2][0]!='term':raise FOLParseError('Equality requires terms')
            eq=self.term(a[1][1],bound)==self.term(a[2][1],bound);return eq if op=='=' else z3.Not(eq)
        x,y=self.compile(a[1],bound),self.compile(a[2],bound)
        return {'∧':z3.And,'∨':z3.Or,'⊕':z3.Xor,'→':z3.Implies,'↔':lambda x,y:x==y}[op](x,y)

def solve(theory,question,timeout_ms=1500,evidence=False):
    c=Compiler();fs=[c.compile(a) for a in conjuncts(theory)];q=c.compile(question)
    s=z3.Solver();s.set(timeout=timeout_ms,random_seed=0)
    for i,f in enumerate(fs):s.assert_and_track(f,z3.Bool('premise_'+str(i)))
    base=s.check();out={'label':None,'base_status':str(base),'checks':{}}
    if base==z3.unsat:
        out.update(label='Inconsistent',inconsistent_core=[str(x) for x in s.unsat_core()]);return out
    if base==z3.unknown:out.update(label='SolverUnknown',reason=s.reason_unknown());return out
    statuses=[]
    for name,extra in [('countermodel',z3.Not(q)),('witness',q)]:
        s.push();s.add(extra);r=s.check();d={'status':str(r)}
        if evidence and r==z3.sat:d['model']=s.model().sexpr()
        if r==z3.unsat:d['support']=[str(x) for x in s.unsat_core()]
        if r==z3.unknown:d['reason']=s.reason_unknown()
        out['checks'][name]=d;statuses.append(r);s.pop()
    a,b=statuses
    out['label']='True' if a==z3.unsat else 'False' if b==z3.unsat else 'Uncertain' if a==b==z3.sat else 'SolverUnknown'
    return out

def ambiguity_paths(a,path=()):
    out=[path] if a[0] in ('∨','⊕') else []
    if a[0] in ('atom','term'):return out
    indices=[1] if a[0]=='¬' else [2] if a[0] in ('∀','∃') else [1,2]
    for i in indices:out.extend(ambiguity_paths(a[i],path+(i,)))
    return out

def replace_ops(a,changes,path=()):
    if a[0] in ('atom','term'):return a
    out=list(a)
    if path in changes:out[0]=changes[path]
    indices=[1] if a[0]=='¬' else [2] if a[0] in ('∀','∃') else [1,2]
    for i in indices:out[i]=replace_ops(a[i],changes,path+(i,))
    return tuple(out)

def certify(results):
    """Conditional guarantee: true interpretation must be in the enumerated set."""
    labels=[r['label'] for r in results];valid=[l for l in labels if l not in ('Inconsistent',)]
    # Parse failures and solver unknowns block certification, not silently removed.
    cert=bool(valid) and all(l in ('True','False','Uncertain') for l in valid) and len(set(valid))==1
    return {'certified':cert,'label':valid[0] if cert else 'Abstain','candidate_labels':labels,'inconsistent_excluded':labels.count('Inconsistent')}
