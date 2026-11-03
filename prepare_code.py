"""Restore supporting source/tests/configuration only; no datasets or model files."""
from pathlib import Path
import hashlib,json,stat,zipfile
R=Path(__file__).resolve().parent
with zipfile.ZipFile(R/'supporting_code.zip') as z:
 assert z.testzip() is None
 for info in z.infolist():
  dest=(R/info.filename).resolve()
  if R not in dest.parents or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive path')
  if dest.exists() and dest.read_bytes()!=z.read(info):raise FileExistsError('Refusing to overwrite modified file: '+info.filename)
 for info in z.infolist():
  dest=R/info.filename
  if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
hashes=json.loads((R/'SOURCE_SHA256.json').read_text())
for name,h in hashes.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
print('PASS: all',len(hashes),'original source modules restored byte-for-byte. No data, models or predictions extracted.')
