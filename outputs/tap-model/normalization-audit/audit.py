from pathlib import Path
import importlib.util,sys,json,numpy as np,joblib
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('light_audit',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import SESSION
z=np.load(ROOT.parent/'no-microphone/features.npz');idx=z['idx'];w=z['w'];M=z['motion'];y=z['y']
events=[json.loads(x) for x in (SESSION/'events.jsonl').read_text().splitlines()];bad={e['trial_id'] for e in events if e.get('event')=='invalidate_trial'}
lookup={r['trial_index']:r for r in [json.loads(x) for x in (SESSION/'trials.jsonl').read_text().splitlines()] if r['trial_id'] not in bad}
light,angle=d.aligned_raw();L=np.asarray([d.get_features(lookup[int(i)],light,angle)[0] for i in idx])
np.savez_compressed(ROOT/'features.npz',motion=M,light=L,idx=idx,w=w,y=y)
report={'audio_used':False,'feature_counts':{'motion':M.shape[1],'light':L.shape[1]},'checks':{},'constant_features':{},'outliers':{}}
for stage,name,mask in [('evaluation','evaluation-model.joblib',idx<520),('final','model.joblib',np.ones(len(idx),bool))]:
 bundle=joblib.load(ROOT.parent/'light-dynamics'/name)
 for group,A in [('motion',M),('light',L)]:
  for axis,head in enumerate(bundle[group]):
   scaler=head.named_steps['standardscaler'];n=int(scaler.n_samples_seen_);assert n==mask.sum()
   np.testing.assert_allclose(scaler.mean_,A[mask].mean(0),rtol=1e-9,atol=1e-9);B=scaler.transform(A[mask]);nonconstant=A[mask].std(0)>1e-9
   assert np.isfinite(B).all();meanerr=float(np.abs(B.mean(0)).max());stderror=float(np.abs(B[:,nonconstant].std(0)-1).max());assert meanerr<1e-7 and stderror<1e-7
   report['checks'][f'{stage}_{group}_{axis}']=dict(samples=n,max_abs_mean=meanerr,max_std_error=stderror)
  if stage=='evaluation':
   report['constant_features'][group]=int((~nonconstant).sum());T=scaler.transform(A[mask]);test=scaler.transform(A[~mask]);report['outliers'][group]=dict(train_max_abs_z=float(np.abs(T).max()),train_fraction_abs_z_above_8=float((np.abs(T)>8).mean()),test_max_abs_z=float(np.abs(test).max()),test_fraction_abs_z_above_8=float((np.abs(test)>8).mean()),manual_train_max_abs_z=float(np.abs(T[w[mask]<1]).max()))
report['conclusion']='Saved StandardScaler instances are fitted on correct training subsets. All nonconstant training features have mean~0,std~1. No NaN/Inf. Scaling is unweighted although classifier downweights manual labels; robust/weighted variants will be compared without touching test for selection.'
(ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
