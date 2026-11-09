"""Select discrete cell/axis classifiers by exact-hit and spatial evaluation."""
from pathlib import Path
import numpy as np
import json,gzip
import lightgbm as lgb
from train_model import prepare,metrics,features
ROOT=Path(__file__).resolve().parent
rows=sorted([json.loads(l) for l in gzip.open(ROOT.parent/'tap-dataset/trials.valid.jsonl.gz','rt')],key=lambda r:r['trial_index'])
cache=ROOT.parent.parent/'work/classification-data.npz'
used=[];F=[];w=[]
for r in rows:
 try:f,t,weight=prepare(r)
 except ValueError:continue
 F.append(f);w.append(weight);used.append(r)
X=np.stack(F);w=np.array(w);y=np.array([[r['col'],r['row']] for r in used]);idx=np.array([r['trial_index'] for r in used]);auto=w==1
np.savez_compressed(cache,X=X,y=y,w=w,idx=idx)
train=idx<340;val=(idx>=340)&(idx<460)&auto;test=(idx>=460)&auto

def measure(truth,p):
 m=metrics(truth+0.5,p+0.5)
 d=np.abs(truth-p);far=d.max(axis=1)>=6
 edge=(truth[:,0]<=2)|(truth[:,0]>=17)|(truth[:,1]<=1)|(truth[:,1]>=11)
 center=(p[:,0]>=7)&(p[:,0]<=12)&(p[:,1]>=4)&(p[:,1]<=8)
 m.update(far_error_rate=float(far.mean()),edge_n=int(edge.sum()),edge_exact=float((d[edge].max(axis=1)==0).mean()) if edge.any() else None,edge_mean_distance=float(np.sqrt((d[edge]**2).sum(axis=1)).mean()) if edge.any() else None,edge_to_center_rate=float(center[edge].mean()) if edge.any() else None,center_prediction_rate=float(center.mean()),unique_predicted_cells=int(len(np.unique(p,axis=0))))
 # Exact hits lead, nearby hits provide partial credit; half-screen mistakes
 # and edge-to-center mistakes are explicitly penalized during selection.
 m['selection_score']=5*m['exact_cell']+m['within_1_cell']+0.4*m['within_2_cells']-0.5*m['far_error_rate']-0.5*(m['edge_to_center_rate'] or 0)
 return m

def fit(mask,c):
 if c['kind']=='nearest':
  scale=np.maximum(X[mask].std(axis=0),0.03);mean=X[mask].mean(axis=0)
  return dict(kind='nearest',X=(X[mask]-mean)/scale,labels=y[mask],mean=mean,scale=scale)
 edge=(y[:,0]<=2)|(y[:,0]>=17)|(y[:,1]<=1)|(y[:,1]>=11)
 weights=w[mask]*(1+c.get('edge_weight',0)*edge[mask])
 targets=[y[:,0],y[:,1]] if c['kind']=='axes' else [y[:,1]*20+y[:,0]]
 ms=[]
 for target in targets:
  m=lgb.LGBMClassifier(objective='multiclass',n_estimators=c['trees'],learning_rate=.045,num_leaves=c['leaves'],max_depth=c['depth'],min_child_samples=c['min_child'],reg_lambda=2.0,colsample_bytree=.85,n_jobs=2,verbosity=-1,random_state=20261107,deterministic=True,force_col_wise=True)
  m.fit(X[mask],target[mask],sample_weight=weights);ms.append(m)
 return dict(kind=c['kind'],models=ms)

def predict(model,A):
 if model['kind']=='nearest':
  Q=(A-model['mean'])/model['scale'];d=((Q[:,None,:]-model['X'][None,:,:])**2).mean(axis=2)
  return model['labels'][d.argmin(axis=1)]
 if model['kind']=='axes':return np.column_stack([m.predict(A).astype(int) for m in model['models']])
 z=model['models'][0].predict(A).astype(int);return np.column_stack([z%20,z//20])
configs=[]
for manual in [False,True]:
 for edge in [0,1]:
  configs.append(dict(kind='axes',manual=manual,edge_weight=edge,trees=180,depth=3,leaves=7,min_child=4))
  configs.append(dict(kind='axes',manual=manual,edge_weight=edge,trees=300,depth=4,leaves=15,min_child=3))
configs += [dict(kind='cells',manual=True,edge_weight=1,trees=100,depth=3,leaves=7,min_child=1),dict(kind='nearest',manual=False),dict(kind='nearest',manual=True)]
results=[]
for c in configs:
 mask=train & (np.ones(len(X),bool) if c['manual'] else auto)
 model=fit(mask,c);p=predict(model,X[val]);score=measure(y[val],p)
 entry=dict(config=c,validation=score);results.append(entry);print(json.dumps(entry),flush=True)
best=max(results,key=lambda r:r['validation']['selection_score']);c=best['config']
dev=(idx<460)&(np.ones(len(X),bool) if c['manual'] else auto)
eval_model=fit(dev,c);p=predict(eval_model,X[test]);evaluation=measure(y[test],p)
old=json.load(open(ROOT/'heldout-predictions.json'))
lookup={r['trial_index']:r for r in old};oldp=[]
for i in idx[test]:
 r=lookup[int(i)];oldp.append([min(19,max(0,int(np.floor(r['pred_col_float'])))),min(12,max(0,int(np.floor(r['pred_row_float']))))])
baseline=measure(y[test],np.array(oldp))
finalmask=np.ones(len(X),bool) if c['manual'] else auto
final=fit(finalmask,c)
export=dict(schema=2,kind=final['kind'],cols=20,rows=13,feature_count=X.shape[1],feature_version=1,loss='multiclass log loss; decode the highest-probability class, never the mean coordinate; selection rewards exact/near hits and penalizes distant/edge-to-center errors')
if final['kind']=='nearest':
 export['templates']=final['X'].tolist();export['labels']=final['labels'].tolist();export['mean']=final['mean'].tolist();export['scale']=final['scale'].tolist()
else:
 export['heads']=[dict(classes=m.classes_.tolist(),trees=m.booster_.dump_model()['tree_info']) for m in final['models']]
(ROOT/'model-classifier.json').write_text(json.dumps(export,separators=(',',':')))
fixture=[]
for r in [r for r in used if r.get('capture_source')=='automatic'][:5]:
 f,t=features(r['accel_g'],r['gyro_dps'],r['trigger_t'],r.get('lid_angle'));pred=predict(final,f[None,:])[0]+.5
 fixture.append(dict(accel=r['accel_g'],gyro=r['gyro_dps'],trigger=r['trigger_t'],lid=r.get('lid_angle'),features=f.tolist(),prediction=pred.tolist()))
(ROOT/'classifier-parity-fixtures.json').write_text(json.dumps(fixture,separators=(',',':')))
report=dict(selected=best,experiments=results,evaluation=evaluation,previous_regression_same_test=baseline,training_n=int(finalmask.sum()),evaluation_warning='This is the same chronological holdout used in v1. No new independent session has been collected. Parameters selected on validation only, not this test.',classifier='Discrete maximum-probability decision, no coordinate averaging. Edge weighting is chosen on validation, not an unconditional ban on the center.')
(ROOT/'classifier-metrics.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps(report),flush=True)
