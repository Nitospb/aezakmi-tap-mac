from pathlib import Path
import json,importlib.util,numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('data_window_helpers',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import SESSION,prepare
z=np.load(ROOT.parent/'normalization-audit/features.npz');idx=z['idx'];raw=np.memmap(SESSION/'samples.bin',dtype=d.DT,mode='r');streams={}
for k in range(6):
 a=raw[raw['kind']==k];o=np.argsort(a['t'],kind='stable');a=a[o];streams[k]=np.column_stack([a['t'],a['x'],a['y'],a['z']])
events=[json.loads(x) for x in (SESSION/'events.jsonl').read_text().splitlines()];bad={e['trial_id'] for e in events if e.get('event')=='invalidate_trial'};saved={e['trial_id']:e['t'] for e in events if e.get('event')=='tap_saved'}
lookup={r['trial_index']:r for r in [json.loads(x) for x in (SESSION/'trials.jsonl').read_text().splitlines()] if r['trial_id'] not in bad}
T=np.linspace(-.65,.5,461);HT=np.linspace(-.65,.5,921);LT=np.linspace(-.65,.5,12)
W=[];H=[];G=[];L=[];trigger=[];feedback=[]
def interp(k,t,q,axes):
 a=streams[k];j=np.searchsorted(a[:,0],t+q[0],side='left');b=np.searchsorted(a[:,0],t+q[-1],side='right');a=a[max(0,j-1):min(len(a),b+1)]
 return np.stack([np.interp(t+q,a[:,0],a[:,v]) for v in axes],axis=-1)
for n,i in enumerate(idx):
 r=lookup[int(i)];t=r['trigger_t'] if r['capture_source']=='automatic' else prepare(r)[1]-.03
 W.append(np.column_stack([interp(1,t,T,[1,2,3]),interp(0,t,T,[1,2,3])]))
 H.append(np.column_stack([interp(1,t,HT,[1,2,3]),interp(0,t,HT,[1,2,3])]))
 G.append(interp(2,t,T,[1])[:,0]);L.append(np.column_stack([interp(3,t,LT,[1]),interp(4,t,LT,[1,2,3]),interp(5,t,LT,[1])]))
 trigger.append(t);feedback.append(saved[r['trial_id']]-t)
 if n%200==0:print('prepared',n,flush=True)
arrays=dict(W=np.asarray(W),H=np.asarray(H),G=np.asarray(G),L=np.asarray(L),M=z['motion'],F=z['light'],idx=idx,y=z['y'],w=z['w'],trigger=np.asarray(trigger),feedback_delay=np.asarray(feedback))
assert all(np.isfinite(v).all() for v in arrays.values());assert min(np.asarray(feedback)[z['w']==1])>.5
np.savez_compressed(ROOT/'data.npz',**arrays)
(ROOT/'data-audit.json').write_text(json.dumps(dict(n=len(idx),pre_seconds=.65,post_seconds=.5,minimum_auto_feedback_delay=float(min(np.asarray(feedback)[z['w']==1])),audio_used=False,notes=['Continuous samples.bin recovers longer context.','Canonical grid interpolates native ~796Hz IMU, ~155Hz angle and ~9.95Hz optical streams; not new sensor measurements.','Post-window ends before any green tap feedback for every automatic tap.','Manual timing inferred from motion, weight0.3, never main validation/test.']),indent=2));print('DATA_READY',flush=True)
