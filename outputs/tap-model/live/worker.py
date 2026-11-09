"""Persistent local inference worker. Uses the exact saved training feature functions."""
import sys,json,importlib.util,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from feedback_training import retrain,atomic_json
from pathlib import Path
import numpy as np,joblib
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('als_dynamics_features',root/'light-dynamics/train.py');dynamics=importlib.util.module_from_spec(spec);spec.loader.exec_module(dynamics)
from train import extract
state_root=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root.parent/'tap-play-data'
state_root.mkdir(parents=True,exist_ok=True)
base_path=root/'light-dynamics/model.joblib';active_path=base_path;version=0
manifest_path=state_root/'active-model.json'
if manifest_path.exists():
 manifest=json.loads(manifest_path.read_text());candidate=Path(manifest['path']).resolve()
 if candidate==base_path.resolve() or candidate.is_relative_to(state_root):active_path=candidate;version=manifest['version']
bundle=joblib.load(active_path)
assert bundle['kind']=='fusion' and bundle.get('audio_used',False) is False
model_sha=hashlib.sha256(active_path.read_bytes()).hexdigest()
print(json.dumps(dict(ready=True,kind=bundle['kind'],version=version,model_path=str(active_path),training_feedback=bundle.get('training_feedback',0))),flush=True)
for line in sys.stdin:
 try:
  q=json.loads(line)
  if q.get('op')=='train':
   bundle,result=retrain(root,state_root,q['session_folder'],bundle,dynamics,version)
   if result['trained']:
    version=result['version'];active_path=Path(result['path']);model_sha=hashlib.sha256(active_path.read_bytes()).hexdigest()
   result['model_sha256']=model_sha;result['model_path']=str(active_path);print(json.dumps(result),flush=True);continue
  if q.get('op')=='reset':
   bundle=joblib.load(base_path);active_path=base_path;version+=1;model_sha=hashlib.sha256(base_path.read_bytes()).hexdigest();atomic_json(manifest_path,dict(version=version,path=str(base_path),mode='base',training_feedback=0));print(json.dumps(dict(reset=True,version=version,model_path=str(base_path))),flush=True);continue
  t=q['trigger'];samples=q['samples']
  def rows(k):return [[s[1],*s[2:5]] for s in samples if int(s[0])==k]
  a=rows(0);g=rows(1);lid=rows(2);lux=rows(3);raw=rows(4);fourth={s[1]:s[2] for s in samples if int(s[0])==5}
  def window(x):return [r for r in x if t-.15<=r[0]<=t+.4]
  r=dict(accel_g=window(a),gyro_dps=window(g),lid_degrees=[[s[0],s[1]] for s in window(lid)],light_lux=[[s[0],s[1]] for s in window(lux)],light_raw_channels=[[s[0],*s[1:4],fourth[s[0]]] for s in window(raw)],capture_source='automatic',trigger_t=t,lid_angle=q.get('lid'))
  light=np.array([[s[0],s[1],*v[1:4],fourth[s[0]]] for s,v in zip(lux,raw)],float)
  if not np.array_equal(np.array(lux)[:,0],np.array(raw)[:,0]):raise ValueError('Несовпадение меток света')
  angle=np.array([[s[0],s[1]] for s in lid],float)
  feats,_=extract(r);L,_,_,_=dynamics.get_features(r,light,angle);M=feats['motion'][None,:];L=L[None,:]
  prob=dynamics.mix(dynamics.probs(bundle['motion'],M),dynamics.probs(bundle['light'],L),bundle['weight'])
  p=dynamics.prediction(prob)[0]
  joint=prob[1][0,:,None]*prob[0][0,None,:]
  ranks=joint.ravel().argsort()[::-1][:3]
  candidates=[dict(col=int(k%20),row=int(k//20),rank=j+1) for j,k in enumerate(ranks)]
  response=dict(col=int(p[0]),row=int(p[1]),request_id=q.get('request_id'),candidates=candidates,px=prob[0][0].tolist(),py=prob[1][0].tolist(),model_sha256=model_sha,feature_version='motion652_light153',audio_used=False,version=version,model_path=str(active_path),motion_features=M[0].tolist(),light_features=L[0].tolist())
  if q.get('debug'):response['motion_features']=M[0].tolist();response['light_features']=L[0].tolist()
  print(json.dumps(response),flush=True)
 except Exception as e:print(json.dumps(dict(error=str(e))),flush=True)
