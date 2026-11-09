from pathlib import Path
import tarfile,hashlib,json
root=Path.cwd();out=root/'work/github-release';out.mkdir(exist_ok=True)
exclude={'work/model-env','work/module-cache','work/matplotlib','work/github-release'}
def filt(info):
 p=info.name
 if any(p==x or p.startswith(x+'/') for x in exclude) or '/__pycache__' in p or p.endswith('/.DS_Store'):return None
 return info
archive=out/'project-snapshot.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for name in ['outputs','work','README.md','requirements.txt','.gitignore']:
  tar.add(root/name,arcname=name,filter=filt)
h=hashlib.sha256();parts=[]
with archive.open('rb') as f:
 n=0
 while True:
  chunk=f.read(1024*1024)
  if not chunk:break
  if n%(900*1024*1024)==0:
   if n:dest.close()
   path=out/f'project-snapshot.tar.gz.part{len(parts)+1:03d}';dest=path.open('wb');parts.append(path)
  h.update(chunk);dest.write(chunk);n+=len(chunk)
 dest.close()
(out/'SHA256SUMS').write_text(h.hexdigest()+'  project-snapshot.tar.gz\n')
print(json.dumps({'bytes':n,'parts':[p.name for p in parts]}),flush=True)
