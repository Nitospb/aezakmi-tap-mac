from pathlib import Path
import sys,importlib.util,json,subprocess,numpy as np,joblib
root=Path('outputs/tap-model').resolve();spec=importlib.util.spec_from_file_location('als_dynamics_features',root/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import extract,SESSION
rows=[json.loads(s) for s in (SESSION/'trials.jsonl').read_text().splitlines()];rows=[r for r in rows if r['capture_source']=='automatic'][:3]
raw=np.memmap(SESSION/'samples.bin',dtype=d.DT,mode='r');light,angle=d.aligned_raw();model=joblib.load(root/'light-dynamics/model.joblib')
requests=[];expected=[]
for r in rows:
 t=r['trigger_t'];a=raw[(raw['t']>=t-.9)&(raw['t']<=t+.4)];a=np.sort(a,order='t')
 requests.append(json.dumps(dict(trigger=t,lid=r['lid_angle'],debug=True,samples=[[int(s['kind']),s['t'],s['x'],s['y'],s['z']] for s in a])))
 M=extract(r)[0]['motion'];L=d.get_features(r,light,angle)[0];p=d.prediction(d.mix(d.probs(model['motion'],M[None,:]),d.probs(model['light'],L[None,:]),model['weight']))[0];expected.append((M,L,p))
proc=subprocess.run([sys.executable,str(root/'live/worker.py'),str(root)],input='\n'.join(requests)+'\n',text=True,capture_output=True,check=True)
lines=[json.loads(x) for x in proc.stdout.splitlines()];assert lines[0]['ready'];assert len(lines)==4
for response,(M,L,p) in zip(lines[1:],expected):
 assert 'error' not in response,response
 np.testing.assert_allclose(response['motion_features'],M,rtol=1e-8,atol=1e-9);np.testing.assert_allclose(response['light_features'],L,rtol=1e-8,atol=1e-9)
 assert [response['col'],response['row']]==p.tolist()
print('Three recorded taps: all 652 motion + 153 light features and final predictions match training exactly.')
