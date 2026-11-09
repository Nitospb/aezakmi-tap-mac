"""Noise-aware microphone experiment. Audio selection uses pass2 only.
Historical pass3 reused for comparison; no test-based tuning; no new recording.
"""
from pathlib import Path
import sys,json,importlib.util,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from audio_features import AudioArchive,features
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audio_motion_helpers',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import SESSION,prepare,metrics,SEED

class Data:
 pass

def load():
 z=np.load(ROOT.parent/'normalization-audit/features.npz');data=Data();data.M=z['motion'];data.L=z['light'];data.y=z['y'];data.idx=z['idx'];data.w=z['w']
 ev=[json.loads(l) for l in (SESSION/'events.jsonl').read_text().splitlines()];bad={e['trial_id'] for e in ev if e.get('event')=='invalidate_trial'};lookup={r['trial_index']:r for r in [json.loads(l) for l in (SESSION/'trials.jsonl').read_text().splitlines()] if r['trial_id'] not in bad}
 archive=AudioArchive(ROOT.parents[2]/'work/tap-microphone.wav',SESSION/'audio-chunks.jsonl');data.fs=archive.fs
 vectors=[];controls=[];quality=[];valid=[];windows=[];background=[];audit=[]
 for j,i in enumerate(data.idx):
  r=lookup[int(i)];trigger=r['trigger_t'] if r.get('capture_source')=='automatic' else prepare(r)[1]-.03
  try:
   x,seams=archive.window(trigger-.38,trigger+.16);v,c,q=features(x,archive.fs);vectors.append(v);controls.append(c);quality.append(q['quality']);valid.append(True);windows.append(x);q.update(trial_index=int(i),seam_samples=seams);audit.append(q)
  except ValueError as e:
   vectors.append(None);controls.append(None);quality.append(0.);valid.append(False);windows.append(None);audit.append(dict(trial_index=int(i),error=str(e)))
  try:bg,_=archive.window(trigger-.94,trigger-.40);background.append(bg)
  except ValueError:background.append(None)
  if j%150==0:print('EXTRACT',j,flush=True)
 dim=next(v for v in vectors if v is not None).shape[0];data.A=np.array([v if v is not None else np.zeros(dim) for v in vectors]);data.B=np.array([v if v is not None else np.zeros(32) for v in controls]);data.q=np.array(quality);data.valid=np.array(valid);data.windows=windows;data.background=background;data.audit=audit
 np.savez_compressed(ROOT/'features.npz',audio=data.A,background=data.B,quality=data.q,valid=data.valid,idx=data.idx,w=data.w,y=data.y)
 return data

def augmented(data,mask,augment):
 ids=np.flatnonzero(mask&data.valid);A=[data.A[ids]];Y=[data.y[ids]];W=[data.w[ids]]
 if augment:
  pool=[i for i in ids if data.background[i] is not None];rng=np.random.default_rng(SEED)
  for snr in [12,6]:
   F=[]
   for i in ids:
    bg=data.background[int(rng.choice(pool))];x=data.windows[i];tap=x[int(.38*data.fs):int(.46*data.fs)];tap_rms=max(float(np.sqrt(np.mean(tap*tap))),1e-8);noise_rms=max(float(np.sqrt(np.mean(bg*bg))),1e-8);scale=min(10.,tap_rms/(10**(snr/20)*noise_rms));mixed=np.clip(x+scale*bg,-1,1)
    F.append(features(mixed,data.fs)[0])
   A.append(np.array(F));Y.append(data.y[ids]);W.append(data.w[ids]*.7)
 return np.vstack(A),np.vstack(Y),np.concatenate(W)

def fit(A,y,w,family):
 out=[]
 for k in [0,1]:
  if family=='svm':model=SVC(C=10,gamma='scale',probability=True,class_weight='balanced',random_state=SEED)
  elif family=='linear':model=LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')
  elif family=='gbm':model=lgb.LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=5,reg_lambda=4,n_jobs=2,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
  else:model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.6,class_weight='balanced',n_jobs=2,random_state=SEED)
  m=make_pipeline(StandardScaler(),model);last=list(m.named_steps)[-1];m.fit(A,y[:,k],**{last+'__sample_weight':w});out.append(m)
 return out

def probs(models,A):return d.probs(models,A)
def baseline(bundle,M,L):return d.mix(probs(bundle['motion'],M),probs(bundle['light'],L),bundle['weight'])
def combine(base,audio,alpha,gate,quality,valid):
 weight=alpha*(quality if gate=='quality' else np.ones(len(quality)))*valid
 return [a*(1-weight[:,None])+b*weight[:,None] for a,b in zip(base,audio)]
def key(e):s=e['validation'];return s['exact'],s['within_one'],-s['mean_distance_cells']

def main():
 data=load();idx=data.idx;y=data.y;w=data.w;tr=idx<260;va=(idx>=260)&(idx<520)&(w==1);dev=idx<520;te=(idx>=520)&(w==1)
 bm=d.fit(data.M[tr],y[tr],w[tr],dict(family='trees'));bl=d.fit(data.L[tr],y[tr],w[tr],dict(family='svm'));base_train=dict(motion=bm,light=bl,weight=.05);vp=baseline(base_train,data.M[va],data.L[va]);candidates=[dict(kind='baseline',validation=metrics(y[va],d.prediction(vp)))];audio_only=[]
 configs=[dict(family='linear',augment=False),dict(family='svm',augment=False),dict(family='trees',augment=False),dict(family='gbm',augment=False),dict(family='svm',augment=True)]
 for config in configs:
  A,Y,W=augmented(data,tr,config['augment']);models=fit(A,Y,W,config['family']);ap=probs(models,data.A[va]);only=metrics(y[va&data.valid],d.prediction(probs(models,data.A[va&data.valid])));audio_only.append(dict(config=config,validation=only))
  trial=[]
  for gate in ['none','quality']:
   for alpha in [.1,.25,.5,.75,1.]:
    p=d.prediction(combine(vp,ap,alpha,gate,data.q[va],data.valid[va]));e=dict(kind='audio_fusion',config=config,alpha=alpha,gate=gate,validation=metrics(y[va],p));candidates.append(e);trial.append(e)
  print('CANDIDATE',json.dumps(dict(config=config,audio_only=only,best=max(trial,key=key))),flush=True)
 # Background control deliberately excluded from the candidate pool.
 control=fit(data.B[tr&data.valid],y[tr&data.valid],w[tr&data.valid],'linear');control_validation=metrics(y[va&data.valid],d.prediction(probs(control,data.B[va&data.valid])))
 best=max(candidates,key=key);positive=max([e for e in candidates if e['kind']=='audio_fusion'],key=key);best_audio=max(audio_only,key=lambda e:key(e));print('FROZEN_SELECTION',json.dumps(best),flush=True)
 # Freeze all decisions before scoring the historical test.
 evalbase=joblib.load(ROOT.parent/'light-dynamics/evaluation-model.joblib');bp=baseline(evalbase,data.M[te],data.L[te]);base_metrics=metrics(y[te],d.prediction(bp))
 choice=best if best['kind']=='audio_fusion' else positive;A,Y,W=augmented(data,dev,choice['config']['augment']);models=fit(A,Y,W,choice['config']['family']);ap=probs(models,data.A[te]);positive_pred=d.prediction(combine(bp,ap,choice['alpha'],choice['gate'],data.q[te],data.valid[te]));positive_result=metrics(y[te],positive_pred)
 primary_pred=positive_pred if best['kind']=='audio_fusion' else d.prediction(bp);primary=metrics(y[te],primary_pred)
 # Fixed shuffled-audio negative control on the identical 217 trials.
 rng=np.random.default_rng(SEED);perm=rng.permutation(te.sum());shuffled=d.prediction(combine(bp,[p[perm] for p in ap],choice['alpha'],choice['gate'],data.q[te][perm],data.valid[te][perm]));shuffled_result=metrics(y[te],shuffled)
 single_config=best_audio['config']
 if single_config==choice['config']:am=models
 else:A,Y,W=augmented(data,dev,single_config['augment']);am=fit(A,Y,W,single_config['family'])
 audio_test=metrics(y[te&data.valid],d.prediction(probs(am,data.A[te&data.valid])))
 snr=np.array([a.get('snr_db',np.nan) for a in data.audit]);noise=np.array([a.get('background_rms',np.nan) for a in data.audit]);noise_threshold=float(np.nanmedian(noise[tr]));noisy=te&(noise>=noise_threshold);quiet=te&(noise<noise_threshold)
 # The test split covers every original automatic example, falling back to sensors for missing audio.
 group_metrics={}
 for name,mask in [('noisier_background',noisy),('quieter_background',quiet)]:
  sub=mask[te];group_metrics[name]=dict(n=int(sub.sum()),combined=metrics(y[te][sub],primary_pred[sub]),baseline=metrics(y[te][sub],d.prediction(bp)[sub]))
 report=dict(audio_used=best['kind']=='audio_fusion',selected=best,best_positive_audio=positive,candidates=candidates,audio_only_candidates=audio_only,background_only_validation=control_validation,test=primary,previous_same_test=base_metrics,positive_audio_test=positive_result,audio_only_test=audio_test,shuffled_audio_test=shuffled_result,noise_groups=group_metrics,quality_summary=dict(valid_trials=int(data.valid.sum()),invalid_trials=int((~data.valid).sum()),valid_test=int(data.valid[te].sum()),median_snr_db=float(np.nanmedian(snr)),median_quality=float(np.median(data.q)),test_zero_quality=int((data.q[te]==0).sum()),median_audio_alignment_ms=float(np.median([a['alignment_offset_s']*1000 for a in data.audit if 'alignment_offset_s'in a]))),feature_count=data.A.shape[1],sample_rate=data.fs,split=dict(train=int(tr.sum()),validation=int(va.sum()),evaluation_fit=int(dev.sum()),test=int(te.sum())),normalization='StandardScaler fitted on training audio features only; per-tap background PSD estimated from -340..-100 ms, never future/validation data.',augmentation='Only training background snippets from -940..-400 ms; mixed at nominal +12/+6 dB SNR, original labels weighted 1 and augmented labels 0.7. No validation/test audio in augmentation.',warnings=['Historical third pass reused, not a fresh independent recording.','No speech recognition; features are short-window spectrum/attack/decay/noise-relative shape.','Noise suppression cannot guarantee recovery of a tap fully masked by another transient.','Current-room background can differ from future background; needs new-session validation.','For missing audio, fused prediction falls back to existing sensors; test set not reduced.'])
 (ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(ROOT/'audio-audit.json').write_text(json.dumps(data.audit,indent=2));(ROOT/'heldout-predictions.json').write_text(json.dumps([dict(trial_index=int(i),truth=t.tolist(),prediction=p.tolist(),previous=b.tolist()) for i,t,p,b in zip(idx[te],y[te],primary_pred,d.prediction(bp))],indent=2));joblib.dump(dict(kind='audio_fusion',audio=models,baseline=evalbase,alpha=choice['alpha'],gate=choice['gate'],config=choice['config']),ROOT/'evaluation-model.joblib',compress=3);print('EVALUATION',json.dumps(dict(selected=best,test=primary,previous=base_metrics,audio_only=audio_test,positive=positive_result,shuffled=shuffled_result,quality=report['quality_summary'])),flush=True)
 # Final refit is stored separately from the evaluated model.
 finalbase=joblib.load(ROOT.parent/'light-dynamics/model.joblib');mask=np.ones(len(idx),bool);A,Y,W=augmented(data,mask,choice['config']['augment']);finalmodels=fit(A,Y,W,choice['config']['family']);joblib.dump(dict(kind='audio_fusion',audio=finalmodels,baseline=finalbase,alpha=choice['alpha'],gate=choice['gate'],config=choice['config'],selected=best,source='audio_features.py'),ROOT/'model.joblib',compress=3)
 (ROOT/'normalization.json').write_text(json.dumps([dict(mean=m.named_steps['standardscaler'].mean_.tolist(),scale=m.named_steps['standardscaler'].scale_.tolist(),fit_rows=int(m.named_steps['standardscaler'].n_samples_seen_)) for m in finalmodels],indent=2));print('FINAL_SAVED',flush=True)
if __name__=='__main__':main()
