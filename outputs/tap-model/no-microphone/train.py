"""New schema-4 tap training. Audio files/fields never read as model inputs.
Selection: pass 1 train, pass 2 validation. Evaluation: refit passes 1+2,
then score untouched pass 3. Final deployable joblib refits the selected model
on all eligible examples; reported test metrics belong to pre-refit model.
"""
from pathlib import Path
import sys,json,collections,math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import joblib
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT.parent))
from train_model import prepare
SESSION=BASE/'tap-data/2026-11-07T19-24-55Z-26D1D2'
SEED=20261107

def series_features(rows,center,channels):
 a=np.asarray(rows,dtype=float)
 if a.ndim!=2 or len(a)<2 or a.shape[1]!=channels+1:raise ValueError('missing extra sensor stream')
 a=a[np.argsort(a[:,0])];a=a[(a[:,0]>=center-.15)&(a[:,0]<=center+.40)]
 if len(a)<2:raise ValueError("extra stream too short near aligned impulse")
 t=a[:,0];v=a[:,1:]
 q=center+np.linspace(-.12,.32,13)
 f=[]
 for k in range(channels):
  x=v[:,k];before=x[t<center-.03];base=float(np.median(before)) if len(before) else float(x[0]);d=x-base
  wave=np.interp(q,t,d)
  span=max(1e-8,float(np.max(np.abs(d))))
  f.extend([base,float(d.min()),float(d.max()),float(d.mean()),float(np.std(d)),float(np.sqrt(np.mean(d*d))),float(d[-1]),float(np.ptp(x)),float(np.mean(np.abs(np.diff(d))))])
  f.extend(wave);f.extend(wave/span)
 return np.asarray(f)

def extract(r):
 old,center,weight=prepare(r)
 # Full-rate short waveform and normalized first differences in addition to v1 stats.
 f=[]
 for name in ['gyro_dps','accel_g']:
  a=np.asarray(r[name]);a=a[(a[:,0]>=center-.15)&(a[:,0]<=center+.40)];t=a[:,0];v=a[:,1:];before=v[t<center-.045];v=v-(before.mean(0) if len(before) else v[0])
  q=center-.025+np.arange(104)/800
  wave=np.stack([np.interp(q,t,v[:,k]) for k in range(3)],axis=1)
  scale=max(float(np.linalg.norm(wave,axis=1).max()),1e-8)
  # 52 points preserve 400 Hz waveform, derivative energy retains high-rate detail.
  f.extend((wave[::2]/scale).ravel())
  dw=np.diff(wave,axis=0)/scale
  for k in range(3):
   x=dw[:,k];f.extend([float(x.min()),float(x.max()),float(np.sqrt(np.mean(x*x))),float(np.abs(x).mean())])
  fft=np.abs(np.fft.rfft(wave/scale,axis=0))**2;freq=np.fft.rfftfreq(len(wave),1/800)
  for lo,hi in [(0,25),(25,75),(75,150),(150,400)]:f.extend(np.log1p(fft[(freq>=lo)&(freq<hi)].sum(axis=0)))
 imu=np.r_[old[:-2],old[-1],f] # exclude v1 static angle; group it with angle stream
 lid=series_features(r['lid_degrees'],center,1)
 light=np.r_[series_features(r['light_lux'],center,1),series_features(r['light_raw_channels'],center,4)]
 return dict(imu=imu,motion=np.r_[imu,lid],all=np.r_[imu,lid,light]),weight

def metrics(t,p):
 d=np.abs(t-p);dist=np.linalg.norm(d,axis=1);edge=(t[:,0]==0)|(t[:,0]==19)|(t[:,1]==0)|(t[:,1]==12)
 out=dict(n=len(t),exact=float((d.max(1)==0).mean()),within_one=float((d.max(1)<=1).mean()),within_two=float((d.max(1)<=2).mean()),mean_distance_cells=float(dist.mean()),median_distance_cells=float(np.median(dist)),x_exact=float((d[:,0]==0).mean()),y_exact=float((d[:,1]==0).mean()),side_x=float(((t[:,0]>=10)==(p[:,0]>=10)).mean()),side_y=float(((t[:,1]>=6.5)==(p[:,1]>=6.5)).mean()),unique_predicted_cells=int(len(np.unique(p,axis=0))),edge_n=int(edge.sum()),edge_exact=float((d[edge].max(1)==0).mean()) if edge.any() else None)
 for n in [2,3,4]:
  def zones(y):return np.minimum(n-1,((y[:,0]+.5)*n/20).astype(int))+n*np.minimum(n-1,((y[:,1]+.5)*n/13).astype(int))
  out[f'region_{n}x{n}']=float((zones(t)==zones(p)).mean())
 return out

def fit(X,y,w,c):
 models=[]
 for k in [0,1]:
  if c['family']=='svm':
   m=make_pipeline(StandardScaler(),SVC(C=c['C'],gamma='scale',class_weight='balanced',random_state=SEED));kw={'svc__sample_weight':w}
  elif c['family']=='linear':
   m=make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],max_iter=1500,class_weight='balanced'));kw={'logisticregression__sample_weight':w}
  elif c['family']=='trees':
   m=make_pipeline(StandardScaler(),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.6,class_weight='balanced',n_jobs=2,random_state=SEED));kw={'extratreesclassifier__sample_weight':w}
  else:
   m=make_pipeline(StandardScaler(),lgb.LGBMClassifier(n_estimators=200,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=4,colsample_bytree=.85,n_jobs=2,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True));kw={'lgbmclassifier__sample_weight':w}
  m.fit(X,y[:,k],**kw);models.append(m)
 return models

def pred(m,X):return np.column_stack([q.predict(X).astype(int) for q in m])
def key(e):m=e['validation'];return (m['exact'],m['within_one'],-m['mean_distance_cells'])

def main():
 meta=json.loads((SESSION/'metadata.json').read_text())
 events=[json.loads(x) for x in (SESSION/'events.jsonl').read_text().splitlines()]
 bad={r['trial_id'] for r in events if r.get('event')=='invalidate_trial'}
 raw=sorted([json.loads(x) for x in (SESSION/'trials.jsonl').read_text().splitlines()],key=lambda r:r['trial_index'])
 rows=[r for r in raw if r['trial_id'] not in bad]
 used=[];skipped=[];feature_rows={k:[] for k in ['imu','motion','all']};weights=[]
 for r in rows:
  try:
   feats,w=extract(r)
   if not all(np.isfinite(v).all() for v in feats.values()):raise ValueError('nonfinite features')
  except (ValueError,KeyError,IndexError) as e:skipped.append(dict(trial_id=r['trial_id'],trial_index=r['trial_index'],reason=str(e)));continue
  used.append(r);weights.append(w)
  for k in feats:feature_rows[k].append(feats[k])
 X={k:np.asarray(v) for k,v in feature_rows.items()};y=np.array([[r['col'],r['row']] for r in used]);idx=np.array([r['trial_index'] for r in used]);w=np.array(weights);auto=w==1
 train=idx<260;val=(idx>=260)&(idx<520)&auto;test=(idx>=520)&auto
 configs=[dict(family='linear',C=.1),dict(family='linear',C=1),dict(family='svm',C=10),dict(family='lgbm'),dict(family='trees')]
 experiments=[]
 for group in ['imu','motion','all']:
  for c in configs:
   mask=train # manual labels reduced to 0.3 weight, estimated motion timing only
   model=fit(X[group][mask],y[mask],w[mask],c)
   e=dict(group=group,config=c,validation=metrics(y[val],pred(model,X[group][val])));experiments.append(e);print(json.dumps(e),flush=True)
 selected=max(experiments,key=key);print('SELECTED',json.dumps(selected),flush=True)
 # Freeze best config for each ablation before opening test. No test-based model changes.
 choices={g:max([e for e in experiments if e['group']==g],key=key) for g in X}
 dev=idx<520;evaluations={};evaluation_models={};predictions={}
 for group,e in choices.items():
  models=fit(X[group][dev],y[dev],w[dev],e['config']);p=pred(models,X[group][test]);evaluations[group]=metrics(y[test],p);evaluation_models[group]=models;predictions[group]=p
  joblib.dump(dict(models=models,feature_group=group,config=e['config']),ROOT/f'evaluation-{group}.joblib',compress=3)
 group=selected['group'];held=predictions[group]
 manual=(idx>=520)&~auto
 manual_result=metrics(y[manual],pred(evaluation_models[group],X[group][manual])) if manual.any() else None
 final=fit(X[group],y,w,selected['config']);joblib.dump(dict(schema=1,models=final,feature_group=group,config=selected['config'],feature_count=X[group].shape[1],cols=20,rows=13,audio_used=False,feature_source='train.py extract'),ROOT/'model.joblib',compress=3)
 scaler=final[0].named_steps['standardscaler']
 (ROOT/'normalization.json').write_text(json.dumps(dict(method='z-score: (x-mean)/scale',fit_scope='all usable examples for final refit only; evaluation scalers fit first two passes only',sample_count=int(scaler.n_samples_seen_),feature_group=group,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),constant_features_use_scale_one=True),indent=2))
 center=np.tile([10,6],(test.sum(),1));random_chance=1/260
 report=dict(session=SESSION.name,dataset_valid=len(rows),used=len(used),automatic=int(auto.sum()),manual_estimated=int((~auto).sum()),skipped=skipped,feature_counts={k:v.shape[1] for k,v in X.items()},split=dict(train_pass1=int(train.sum()),validation_pass2_automatic=int(val.sum()),evaluation_fit_passes12=int(dev.sum()),test_pass3_automatic=int(test.sum()),test_pass3_manual=int(manual.sum())),selected=selected,experiments=experiments,ablation_choices=choices,test_by_feature_group=evaluations,test_selected=evaluations[group],test_manual_estimated=manual_result,center_baseline=metrics(y[test],center),uniform_exact_chance=random_chance,audio_used=False,normalization='Every model is a Pipeline(StandardScaler, classifier). Each scaler fits training data only; mean=0 and std=1 on nonconstant training features. Per-tap sensor deviations also subtract pre-impulse baseline.',selection='lexicographic: exact-cell accuracy, then within-one-cell accuracy, then smaller mean distance; validation only; independent X/Y classification, argmax classes, no coordinate averaging',limitations=['One session on one Mac. Pass 3 is temporally held out, not a new session.','Ground truth is the instructed cell center, not independently observed finger position.','Manual timing inferred from motion; weight 0.3 in training, reported separately in test.','Only 3 labeled taps per cell, before exclusions.','Ablation configurations selected on validation; test never used for selection.','Final model.joblib refit on all usable taps; test scores belong to saved evaluation models trained only on first two passes.'])
 (ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 heldrows=[r for r,b in zip(used,test) if b]
 (ROOT/'heldout-predictions.json').write_text(json.dumps([dict(trial_id=r['trial_id'],trial_index=r['trial_index'],truth=[r['col'],r['row']],prediction=p.tolist()) for r,p in zip(heldrows,held)],indent=2))
 np.savez_compressed(ROOT/'features.npz',**X,y=y,idx=idx,w=w)
 print('FINAL',json.dumps(dict(selected=selected,test=evaluations,manual=manual_result,split=report['split'])),flush=True)
if __name__=='__main__':main()
