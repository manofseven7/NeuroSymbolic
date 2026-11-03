"""Download selected pinned dataset files only when explicitly requested."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile,urllib.request
R=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['FOLIO_curated','LogicLM_ProofWriter','CLUTRR_external','all']);a=p.parse_args()
 rows=json.loads((R/'data_sources.json').read_text())
 if not a.dataset:
  for r in rows:print(r['dataset'],r['path'],r['url'])
  print('No files downloaded. Specify --dataset to opt in.');return
 for r in rows:
  if a.dataset not in ['all',r['dataset']]:continue
  target=R/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():
   if hashlib.sha256(target.read_bytes()).hexdigest()==r['sha256']:print('Verified existing:',r['path']);continue
   raise ValueError('Existing file has unexpected hash: '+r['path'])
  fd,tmp=tempfile.mkstemp(prefix='.download-',dir=target.parent)
  try:
   h=hashlib.sha256()
   with os.fdopen(fd,'wb') as out,urllib.request.urlopen(r['url'],timeout=60) as src:
    while True:
     b=src.read(1024*1024)
     if not b:break
     h.update(b);out.write(b)
   if h.hexdigest()!=r['sha256']:raise ValueError('Download checksum mismatch: '+r['path'])
   os.replace(tmp,target);print('Downloaded and verified:',r['path'])
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
if __name__=='__main__':main()
