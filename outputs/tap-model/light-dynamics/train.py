"""Test explicit ALS dynamics, longer pre-tap baseline and late fusion. No audio.
Selection frozen on pass2. Pass3 is a REUSED historical test, not fresh validation.
Never use samples after trigger+0.40s (before the collector's green feedback).
"""
from pathlib import Path
import sys,json,math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'no-microphone'))
from train import SESSION,prepare,metrics,SEED
CACHE=ROOT.parent/'no-microphone/features.npz'
DT=np.dtype([('kind','<u8'),('t','<f8'),('x','<f8'),('y','<f8'),('z','<f8')])

def aligned_raw():
 raw=np.memmap(SESSION/'samples.bin',dtype=DT,mode='r')
 streams=[]
 for k in [3,4,5]:
  a=raw[raw['kind']==k];a=np.sort(a,order='t');streams.append(a)
 a,b,c=streams
 if not (np.array_equal(a['t'],b['t']) and np.array_equal(a['t'],c['t'])):raise ValueError('ALS channel timestamp mismatch')
 light=np.column_stack([a['t'],a['x'],b['x'],b['y'],b['z'],c['x']])
 a=raw[raw['kind']==2];a=np.sort(a,order='t');angle=np.column_stack([a['t'],a['x']])
 return light,angle

def get_features(r,light,angle):
 _,center,w=prepare(r)
 trigger=r['trigger_t'] if r.get('capture_source')=='automatic' else center-.03
 # Real, recorded ALS reports: baseline ~0.6s, response up to 0.4s. No green feedback.
 lo=trigger-.75;hi=trigger+.40
 a=light[np.searchsorted(light[:,0],lo):np.searchsorted(light[:,0],hi,side='right')]
 t=a[:,0]-trigger;v=a[:,1:]
 pre=(t>=-.65)&(t<=-.12);post=(t>=-.025)&(t<=.40)
 if pre.sum()<3 or post.sum()<3:raise ValueError('insufficient native ALS reports for baseline/response')
 ratios=v[:,1:]/np.maximum(v[:,1:].sum(1,keepdims=True),1)
 values=np.column_stack([v,ratios])
 F=[];B=[];relwaves=[]
 for k in range(values.shape[1]):
  x=values[:,k];base=float(np.median(x[pre]));scale=max(abs(base),1 if k<5 else .01)
  trend=np.polyfit(t[pre],x[pre],1);background=np.polyval(trend,t)
  d=(x-base)/scale; detrended=(x-background)/scale
  pt=t[post];pd=d[post];dd=detrended[post]
  velocity=np.diff(x)/np.maximum(np.diff(t),1e-6)/scale;vt=(t[1:]+t[:-1])/2;pv=velocity[(vt>=-.025)&(vt<=.40)]
  B.extend([base,float(np.std(x[pre])),float(trend[0]/scale)])
  F.extend([float(pd.min()),float(pd.max()),float(pd[-1]),float(pd.mean()),float(pd.std()),float(np.sqrt(np.mean(pd*pd))),float(np.trapz(pd,pt)),float(dd.min()),float(dd.max()),float(np.sqrt(np.mean(dd*dd))),float(pv.min()) if len(pv) else 0,float(pv.max()) if len(pv) else 0,float(np.sqrt(np.mean(pv*pv))) if len(pv) else 0])
  # Interpolation only for fixed ML features; does not increase sensor sampling rate.
  wave=np.interp([.05,.15,.25,.35],t,d);F.extend(wave);relwaves.append(d)
 # Cross-modal associations at ACTUAL ALS report times.
 ar=angle[np.searchsorted(angle[:,0],lo):np.searchsorted(angle[:,0],hi,side='right')]
 C=[]
 def corr(x,y):
  x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y);return float(x@y/den) if den>1e-12 else 0.0
 for lag in [-.10,0,.10]:
  av=np.interp(a[:,0]+lag,ar[:,0],ar[:,1]);av=av-np.median(av[pre])
  for d in relwaves:C.extend([corr(d[post],av[post]),float(np.mean(d[post]*av[post]))])
 return np.asarray(F),np.asarray(B),np.asarray(C),dict(pre=int(pre.sum()),post=int(post.sum()),median_dt=float(np.median(np.diff(a[:,0]))))

def fit(A,y,w,c):
 models=[]
 for axis in [0,1]:
  family=c['family']
  if family=='linear':m=LogisticRegression(C=.1,max_iter=1200,class_weight='balanced')
  elif family=='svm':m=SVC(C=10,gamma='scale',probability=True,class_weight='balanced',random_state=SEED)
  elif family=='gbm':m=lgb.LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=4,n_jobs=2,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
  else:m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.6,class_weight='balanced',n_jobs=2,random_state=SEED)
  pipeline=make_pipeline(StandardScaler(),m);pipeline.fit(A,y[:,axis],**{list(pipeline.named_steps)[-1]+'__sample_weight':w});models.append(pipeline)
 return models

def probs(models,A):
 out=[]
 for k,m in enumerate(models):
  full=np.zeros((len(A),[20,13][k]));full[:,m.classes_.astype(int)]=m.predict_proba(A);out.append(full)
 return out

def prediction(p):return np.column_stack([a.argmax(1) for a in p])
def mix(a,b,weight):return [(1-weight)*x+weight*y for x,y in zip(a,b)]
def score_key(e):m=e['validation'];return m['exact'],m['within_one'],-m['mean_distance_cells']

def main():
 cache=np.load(CACHE);idx=cache['idx'];labels=cache['y'];weights=cache['w'];motion=cache['motion']
 ev=[json.loads(l) for l in (SESSION/'events.jsonl').read_text().splitlines()];bad={e['trial_id'] for e in ev if e.get('event')=='invalidate_trial'}
 lookup={r['trial_index']:r for r in [json.loads(l) for l in (SESSION/'trials.jsonl').read_text().splitlines()] if r['trial_id'] not in bad}
 light,angle=aligned_raw();features=[];contexts=[];cross=[];used=[];audit=[];skipped=[]
 for j,i in enumerate(idx):
  try:f,b,c,a=get_features(lookup[int(i)],light,angle)
  except (ValueError,IndexError) as e:skipped.append(dict(trial_index=int(i),reason=str(e)));continue
  assert np.isfinite(np.r_[f,b,c]).all();features.append(f);contexts.append(b);cross.append(c);used.append(j);audit.append(a)
 used=np.asarray(used);idx=idx[used];y=labels[used];w=weights[used];M=motion[used];L=np.asarray(features);B=np.asarray(contexts);C=np.asarray(cross);J=np.column_stack([M,L,C])
 train=idx<260;val=(idx>=260)&(idx<520)&(w==1);dev=idx<520;test=(idx>=520)&(w==1)
 baseline_cfg=dict(family='trees');motion_train=fit(M[train],y[train],w[train],baseline_cfg);mp=probs(motion_train,M[val]);results=[];candidates=[]
 base=dict(kind='motion_only',validation=metrics(y[val],prediction(mp)));candidates.append(base);print(json.dumps(base),flush=True)
 configs=[dict(family=s) for s in ['linear','svm','gbm','trees']]
 for c in configs:
  m=fit(L[train],y[train],w[train],c);lp=probs(m,L[val]);entry=dict(kind='light_only',config=c,validation=metrics(y[val],prediction(lp)));results.append(entry);print(json.dumps(entry),flush=True)
  for weight in [.05,.1,.2,.35,.5]:
   e=dict(kind='fusion',config=c,light_weight=weight,validation=metrics(y[val],prediction(mix(mp,lp,weight))));candidates.append(e)
 for family in ['trees','gbm']:
  c=dict(family=family);m=fit(J[train],y[train],w[train],c);e=dict(kind='joint_dynamic_light',config=c,validation=metrics(y[val],prediction(probs(m,J[val]))));candidates.append(e);print(json.dumps(e),flush=True)
 # Ambient-only control tests whether static lighting/context can look predictive.
 context=fit(B[train],y[train],w[train],dict(family='linear'));control_validation=metrics(y[val],prediction(probs(context,B[val])))
 best=max(candidates,key=score_key);best_fusion=max([e for e in candidates if e['kind']=='fusion'],key=score_key);best_light=max(results,key=score_key)
 print('FROZEN_SELECTION',json.dumps(best),flush=True)
 # Freeze choice above; now evaluate historical pass3 once for this experiment.
 mm=fit(M[dev],y[dev],w[dev],baseline_cfg);mtp=probs(mm,M[test]);baseline=metrics(y[test],prediction(mtp))
 def evaluate_choice(choice,mask_fit,mask_eval):
  if choice['kind']=='motion_only':models=fit(M[mask_fit],y[mask_fit],w[mask_fit],baseline_cfg);return prediction(probs(models,M[mask_eval])),dict(kind='motion_only',motion=models)
  if choice['kind']=='fusion':
   a=fit(M[mask_fit],y[mask_fit],w[mask_fit],baseline_cfg);b=fit(L[mask_fit],y[mask_fit],w[mask_fit],choice['config']);p=prediction(mix(probs(a,M[mask_eval]),probs(b,L[mask_eval]),choice['light_weight']));return p,dict(kind='fusion',motion=a,light=b,weight=choice['light_weight'])
  m=fit(J[mask_fit],y[mask_fit],w[mask_fit],choice['config']);return prediction(probs(m,J[mask_eval])),dict(kind='joint_dynamic_light',joint=m)
 p,bundle=evaluate_choice(best,dev,test);primary=metrics(y[test],p)
 if best['kind']=='fusion':fusion_result=primary
 else:fp,_=evaluate_choice(best_fusion,dev,test);fusion_result=metrics(y[test],fp)
 lm=fit(L[dev],y[dev],w[dev],best_light['config']);light_test=metrics(y[test],prediction(probs(lm,L[test])))
 (ROOT/'heldout-predictions.json').write_text(json.dumps([dict(trial_index=int(i),truth=t.tolist(),prediction=q.tolist()) for i,t,q in zip(idx[test],y[test],p)],indent=2))
 joblib.dump(bundle,ROOT/'evaluation-model.joblib',compress=3)
 report=dict(audio_used=False,used=len(idx),skipped=skipped,features=dict(motion=M.shape[1],light_dynamics=L.shape[1],light_angle_cross=C.shape[1],ambient_control=B.shape[1]),native_ALS_hz=float(1/np.median([a['median_dt'] for a in audit])),pre_reports_median=float(np.median([a['pre'] for a in audit])),post_reports_median=float(np.median([a['post'] for a in audit])),normalization='StandardScaler fitted on training subset only for every model',validation_candidates=candidates,light_only_candidates=results,ambient_control_validation=control_validation,selected=best,best_positive_light_fusion=best_fusion,test_selected=primary,test_motion_baseline=baseline,test_positive_light_fusion=fusion_result,test_light_only=light_test,split=dict(train=int(train.sum()),validation=int(val.sum()),eval_fit=int(dev.sum()),test=int(test.sum())),warning='Pass3 was evaluated in an earlier iteration. Reused historical test, not new independent confirmation. No selection based on pass3 in this script.',feature_notes=['Only native ALS reports from continuous samples.bin, no audio.','Baseline trigger−0.65..−0.12s; response −0.025..+0.40s; no green-feedback data.','Relative signal, pre-trend residual, signed response, native-time derivative, ratios of four uncalibrated channels. Channels are not claimed to be RGB.','Fixed-time interpolation provides features, not higher sensor sampling frequency.','Motion/light association features are not proof of vibration causality; hand shadow and display illumination remain confounds.'])
 (ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('RESULT',json.dumps(report),flush=True)
 # Final refit is separate from evaluation. Bundle stores normalization in each head.
 allmask=np.ones(len(idx),bool);_,final=evaluate_choice(best,allmask,allmask);final.update(audio_used=False,selection=best,feature_source='get_features + no-microphone/train.py extract')
 joblib.dump(final,ROOT/'model.joblib',compress=3);print('FINAL_MODEL_SAVED',flush=True)
if __name__=='__main__':main()
