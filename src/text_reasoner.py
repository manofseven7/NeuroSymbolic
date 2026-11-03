"""Controlled-English parser and proof-tracing Horn executor.

No pretrained language model. Grammar developed on upstream dev only.
Explicit negation, open world, single shared universal variable per rule.
"""
import itertools,json,re
from dataclasses import dataclass

class ParseError(ValueError):pass

def entity(x):
    x=re.sub(r'^the ','',x.strip().lower())
    return '?x' if x in ['something','someone','somebody','it','they','everything'] else x

def atom(text,subject=None):
    t=text.strip().lower().rstrip('.').strip()
    t=re.sub(r'\bthat is\b','is',t)
    if subject and not re.search(r'\b(is|are|does)\b',t) and len(t.split())<=2:
        t=subject+' is '+t
    m=re.fullmatch(r'(.+?) (?:is|are) (not )?([a-z]+)',t)
    if m:return (m[3],entity(m[1]),'',not bool(m[2]))
    m=re.fullmatch(r'(.+?) ((?:does|do) not )?(chases?|eats?|likes?|needs?|sees?|visits?) (.+)',t)
    if m:
        verb=m[3]
        if verb.endswith('s'):verb=verb[:-1]
        return (verb,entity(m[1]),entity(m[4]),not bool(m[2]))
    raise ParseError('Unsupported atom: '+text)

def sentence(text):
    t=text.strip().rstrip('.').lower()
    if t.startswith('if '):
        parts=t[3:].split(' then ')
        if len(parts)!=2:raise ParseError(text)
        body=[];subject=None
        for part in parts[0].split(' and '):
            a=atom(part,subject);body.append(a)
            if subject is None:subject=a[1]
        return ('rule',tuple(body),atom(parts[1]))
    m=re.fullmatch(r'(?:all )?(.+?) (?:things|people) are (not )?([a-z]+)',t)
    if m:
        words=re.split(r', | and ',m[1]);body=[]
        for word in words:
            word=word.strip();neg=word.startswith('not ');pred=word[4:] if neg else word
            if not re.fullmatch('[a-z]+',pred):raise ParseError(text)
            body.append((pred,'?x','',not neg))
        return ('rule',tuple(body),(m[3],'?x','',not bool(m[2])))
    return ('fact',(),atom(t))

def parse(context):
    return [(f's{i+1}',s.strip(),sentence(s)) for i,s in enumerate(context.split('.')) if s.strip()]

def query(text):return atom(text.split('?')[-1])

def key(a):return json.dumps(a,separators=(',',':'))
def opposite(a):return a[:3]+(not a[3],)
def ground(a,x):return tuple(x if v=='?x' else v for v in a)

def transform(records,bits):
    """Injected diagnostic alternatives: fact negation, conjunction, rule polarity.

    bit=1 means fault. No claim these are naturally observed model errors.
    """
    ff,conj,head=bits;out=[]
    for sid,text,(kind,body,a) in records:
        if kind=='fact' and ff and not a[3]:a=opposite(a)
        if kind=='rule':
            if conj and len(body)>1:body=body[:1]
            if head and not a[3]:a=opposite(a)
        out.append((sid,text,(kind,body,a)))
    return out

def run(records,q):
    entities={v for _,_,(_,body,a) in records for b in (*body,a) for v in b[1:3] if v and v!='?x'}
    entities|={v for v in q[1:3] if v and v!='?x'}
    if '?x' in q:raise ParseError('Non-ground question')
    known={};rules=[];origins={}
    for sid,text,(kind,body,a) in records:
        origins[sid]=text
        if kind=='fact':
            if '?x' in a:raise ParseError('Variable fact')
            known.setdefault(a,{'id':sid,'atom':a,'parents':[],'kind':'fact'})
        else:
            var=any('?x' in b for b in (*body,a))
            for x in sorted(entities) if var else ['']:
                rules.append((sid,tuple(ground(b,x) for b in body),ground(a,x),x))
    while True:
        changed=False
        for sid,body,a,x in rules:
            if a not in known and all(b in known for b in body):
                known[a]={'id':sid,'atom':a,'parents':list(body),'kind':'rule','binding':x};changed=True
        if not changed:break
    pos=q in known;neg=opposite(q) in known
    label=3 if pos and neg else 0 if pos else 1 if neg else 2
    target=q if pos else opposite(q) if neg else None;proof=[];visited=set()
    def visit(a):
        if a in visited:return
        node=known[a]
        for b in node['parents']:visit(b)
        visited.add(a);proof.append(node)
    if target:visit(target)
    return label,proof,known

def verify(records,q,label,proof):
    """Independently check each proof step against source and prior proof nodes."""
    byid={sid:r for sid,_,r in records};available=set()
    for node in proof:
        kind,body,a=byid[node['id']];got=tuple(node['atom'])
        if kind=='fact':
            if got!=a:return False
        else:
            x=node.get('binding','');bb=tuple(ground(b,x) for b in body)
            if got!=ground(a,x) or not all(v in available for v in bb):return False
        available.add(got)
    if label in (0,1):return (q if label==0 else opposite(q)) in available
    if label==2:return not proof
    return False

def verbalize(records,proof,label):
    text={sid:txt for sid,txt,_ in records};lines=[]
    for step in proof:
        a=step['atom'];subject=a[1];pred=a[0];neg='' if a[3] else 'not '
        fact=f'{subject} is {neg}{pred}' if not a[2] else f'{subject} {"does not " if not a[3] else ""}{pred} {a[2]}'
        lines.append({'source_id':step['id'],'source_text':text[step['id']],'derived':fact,'kind':step['kind'],'binding':step.get('binding')})
    return {'answer':['true','false','unknown','inconsistent'][label],'steps':lines,'unknown_note':'Neither statement nor explicit negation was derived in the finite parsed program.' if label==2 else None}
