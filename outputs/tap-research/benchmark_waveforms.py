"""Research-inspired, high-rate impulse/derivative benchmark. No test-based tuning."""
from pathlib import Path
import json,gzip,numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
ROOT=Path(__file__).resolve().parent
rows=[json.loads(l) for l in gzip.open(ROOT.parent/'tap-dataset/trials.valid.jsonl.gz','rt')]
rows=sorted([r for r in rows if r.get('capture_source')=='automatic'],key=lambda r:r['trial_index'])
def build(r,mode):
 a=np.asarray(r['accel_g']);g=np.asarray(r['gyro_dps']);trigger=r['trigger_t']
 raw=[];dense=[]
 for stream in [g,a]:
  t=stream[:,0];v=stream[:,1:];base=v[t<trigger-.04].mean(axis=0);v=v-base
  raw.append((t,v))
 # Reproducible peak alignment within the first 120ms, then preserve ~800 Hz.
 t,v=raw[0];sel=np.where((t>=trigger-.015)&(t<=trigger+.12))[0]
 mag=np.linalg.norm(v[sel],axis=1);center=t[sel[np.argmax(mag)]]
 q=center-.025+np.arange(104)/800
 for t,v in raw:
  wave=np.stack([np.interp(q,t,v[:,k]) for k in range(3)],axis=1)
  if mode=='derivative':wave=np.diff(wave,axis=0)
  scale=max(float(np.sqrt((wave*wave).sum(axis=1)).max()),1e-8)
  dense.extend((wave/scale).ravel());dense.append(np.log1p(scale))
 dense.append(r['lid_angle']/180)
 return dense
X={m:np.array([build(r,m) for r in rows]) for m in ['raw','derivative']}
y=np.array([[r['col'],r['row']] for r in rows]);idx=np.array([r['trial_index'] for r in rows])
tr=idx<340;va=(idx>=340)&(idx<460);te=idx>=460

def metrics(t,p):
 d=np.abs(t-p);return dict(n=len(t),exact=float((d.max(axis=1)==0).mean()),within_one=float((d.max(axis=1)<=1).mean()),mean_distance=float(np.sqrt((d*d).sum(axis=1)).mean()),x_exact=float((d[:,0]==0).mean()),y_exact=float((d[:,1]==0).mean()))
def fit(A,labels,config):
 models=[]
 for k in range(labels.shape[1]):
  if config['family']=='svm':m=make_pipeline(StandardScaler(),SVC(C=config['C'],gamma=config['gamma'],kernel='rbf',class_weight='balanced'))
  else:m=ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,max_features=.5,class_weight='balanced',n_jobs=2,random_state=20261107)
  m.fit(A,labels[:,k]);models.append(m)
 return models

def pred(models,A):return np.column_stack([m.predict(A) for m in models])
experiments=[]
for mode in X:
 for C in [1,10,100]:
  for gamma in ['scale',.01]:
   config=dict(family='svm',C=C,gamma=gamma,mode=mode)
   models=fit(X[mode][tr],y[tr],config);score=metrics(y[va],pred(models,X[mode][va]));entry=dict(config=config,validation=score);experiments.append(entry);print(json.dumps(entry),flush=True)
 config=dict(family='extra_trees',mode=mode)
 models=fit(X[mode][tr],y[tr],config);entry=dict(config=config,validation=metrics(y[va],pred(models,X[mode][va])));experiments.append(entry);print(json.dumps(entry),flush=True)
best=max(experiments,key=lambda e:5*e['validation']['exact']+e['validation']['within_one']-.02*e['validation']['mean_distance'])
c=best['config'];models=fit(X[c['mode']][idx<460],y[idx<460],c);test=metrics(y[te],pred(models,X[c['mode']][te]))
resolution=[]
for nx,ny in [(2,2),(4,4),(5,4),(8,8),(20,13)]:
 labels=np.column_stack([np.minimum(nx-1,(y[:,0]+.5)*nx//20),np.minimum(ny-1,(y[:,1]+.5)*ny//13)]).astype(int)
 models=fit(X[c['mode']][idx<460],labels[idx<460],c)
 resolution.append(dict(grid=[nx,ny],test=metrics(labels[te],pred(models,X[c['mode']][te]))))
audit=dict(automatic_n=len(rows),manual_n_excluded=162,per_cell_automatic_counts={str(k):int(v) for k,v in zip(*np.unique(y[:,1]*20+y[:,0],return_counts=True))},unique_automatic_cells=int(len(np.unique(y,axis=0))),degrees_range=[min(r['lid_angle'] for r in rows),max(r['lid_angle'] for r in rows)],original_feature_waveform_effective_hz=31/.24,new_waveform_hz=800,window_ms=130)
report=dict(audit=audit,selected=best,experiments=experiments,test=test,multiresolution=resolution,limitations=['Same chronological test as earlier iterations; this is not fresh independent confirmation.','Multiresolution results change the task and do not demonstrate accuracy on 260 cells.','Research-inspired preprocessing, not a TapNet reproduction.'])
(ROOT/'waveform-benchmark.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
