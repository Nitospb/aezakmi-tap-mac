"""Bounded normalization comparison. Classifier settings held fixed; no audio.
Pass1 train, pass2 select; first two refit, historical pass3 test. No test-based tuning.
"""
from pathlib import Path
import importlib.util,json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from normalizers import FeatureNormalizer
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('light_for_norm',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import metrics,SEED

def fit(A,y,w,norm,group):
 models=[]
 for k in [0,1]:
  if group=='motion':m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.6,class_weight='balanced',n_jobs=2,random_state=SEED)
  else:m=SVC(C=10,gamma='scale',probability=True,class_weight='balanced',random_state=SEED)
  model=make_pipeline(FeatureNormalizer(norm),m);last=list(model.named_steps)[-1]
  model.fit(A,y[:,k],featurenormalizer__sample_weight=w,**{last+'__sample_weight':w});models.append(model)
 return models

def probs(m,A):return d.probs(m,A)
def pred(p):return d.prediction(p)
def key(e):s=e['validation'];return s['exact'],s['within_one'],-s['mean_distance_cells']
def main():
 z=np.load(ROOT/'features.npz');M=z['motion'];L=z['light'];idx=z['idx'];y=z['y'];w=z['w'];tr=idx<260;va=(idx>=260)&(idx<520)&(w==1);dev=idx<520;te=(idx>=520)&(w==1)
 candidates=[];normalization_checks={}
 for norm in ['standard','weighted','winsor','asinh']:
  motion=fit(M[tr],y[tr],w[tr],norm,'motion');light=fit(L[tr],y[tr],w[tr],norm,'light');mp=probs(motion,M[va]);lp=probs(light,L[va]);trial=[]
  for weight in [0,.05,.1,.2,.35]:
   e=dict(normalization=norm,light_weight=weight,validation=metrics(y[va],pred(d.mix(mp,lp,weight))));candidates.append(e);trial.append(e)
  normalization_checks[norm]={}
  for group,A,models in [('motion',M,motion),('light',L,light)]:
   n=models[0].named_steps['featurenormalizer'];T=n.transform(A[tr]);weights=np.ones(tr.sum()) if norm=='standard' else w[tr];mean=np.average(T,axis=0,weights=weights);std=np.sqrt(np.average((T-mean)**2,axis=0,weights=weights));nonconstant=n.scale_!=1
   assert np.abs(mean).max()<1e-7;assert np.abs(std[nonconstant]-1).max()<1e-7
   normalization_checks[norm][group]=dict(fit_n=n.n_samples_seen_,max_abs_mean=float(np.abs(mean).max()),max_abs_z=float(np.abs(T).max()))
  print('NORMALIZATION',json.dumps(max(trial,key=key)),flush=True)
 # Keep positive-light candidates as requested; zero-light included as diagnostic control.
 best=max([e for e in candidates if e['light_weight']>0],key=key);print('FROZEN_SELECTION',json.dumps(best),flush=True)
 models={g:fit(A[dev],y[dev],w[dev],best['normalization'],g) for g,A in [('motion',M),('light',L)]}
 p=pred(d.mix(probs(models['motion'],M[te]),probs(models['light'],L[te]),best['light_weight']));score=metrics(y[te],p)
 oldbundle=joblib.load(ROOT.parent/'light-dynamics/evaluation-model.joblib');old=pred(d.mix(probs(oldbundle['motion'],M[te]),probs(oldbundle['light'],L[te]),oldbundle['weight']));oldscore=metrics(y[te],old)
 bundle=dict(kind='fusion',**models,weight=best['light_weight'],normalization=best['normalization'],audio_used=False);joblib.dump(bundle,ROOT/'evaluation-model.joblib',compress=3)
 held=[dict(trial_index=int(i),truth=t.tolist(),prediction=a.tolist(),previous=b.tolist()) for i,t,a,b in zip(idx[te],y[te],p,old)]
 (ROOT/'heldout-predictions.json').write_text(json.dumps(held,indent=2))
 report=dict(audio_used=False,selection=best,candidates=candidates,normalization_checks=normalization_checks,test=score,previous_same_test=oldscore,split=dict(train=int(tr.sum()),validation=int(va.sum()),evaluation_fit=int(dev.sum()),test=int(te.sum())),classifiers='Same ExtraTrees motion + SVC light settings as current app; normalization and validation-selected mixture only changed.',warnings=['Historical third pass already used in earlier iterations; not fresh independent confirmation.','Quantiles/centers/scales fitted only on each training subset.','Manual examples weighted 0.3 in classifiers and non-standard normalizers.','Final all-data model is separate from evaluation.'])
 (ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('EVALUATION',json.dumps(dict(selection=best,test=score,previous=oldscore)),flush=True)
 final={g:fit(A,y,w,best['normalization'],g) for g,A in [('motion',M),('light',L)]};final.update(kind='fusion',weight=best['light_weight'],normalization=best['normalization'],audio_used=False,selection=best)
 joblib.dump(final,ROOT/'model.joblib',compress=3);print('FINAL_SAVED',flush=True)
if __name__=='__main__':main()
