"""Edge-aware, per-axis classification with explicit partial spatial credit."""
from pathlib import Path
import json,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent
z=np.load(ROOT.parent.parent/'work/classification-data.npz');X=z['X'];y=z['y'];w=z['w'];idx=z['idx'];auto=w==1
train=idx<340;val=(idx>=340)&(idx<460)&auto;test=(idx>=460)&auto
CELLS=np.array([(c,r) for r in range(13) for c in range(20)])
def band(labels,n):return np.where(labels<(n/3),0,np.where(labels>=n*2/3,2,1))
def decode(prob,mode):
 px,py=prob
 if mode=='map':return np.column_stack([px.argmax(axis=1),py.argmax(axis=1)])
 result=[]
 for a,b in zip(px,py):
  probs=a[CELLS[:,0]]*b[CELLS[:,1]]
  edge=(CELLS[:,0]==0)|(CELLS[:,0]==19)|(CELLS[:,1]==0)|(CELLS[:,1]==12)
  corner=((CELLS[:,0]==0)|(CELLS[:,0]==19))&((CELLS[:,1]==0)|(CELLS[:,1]==12))
  score=(10+8*edge+4*corner)*probs+2*a[CELLS[:,0]]+2*b[CELLS[:,1]]
  ax=np.array([a[band(np.arange(20),20)==k].sum() for k in range(3)])
  by=np.array([b[band(np.arange(13),13)==k].sum() for k in range(3)])
  score+=1.5*ax[band(CELLS[:,0],20)]+1.5*by[band(CELLS[:,1],13)]
  # Cost of calling the middle region when the probability mass is on the frame.
  frame_prob=1-(1-a[0]-a[-1])*(1-b[0]-b[-1])
  center=(band(CELLS[:,0],20)==1)&(band(CELLS[:,1],13)==1)
  score-=4*frame_prob*center
  result.append(CELLS[score.argmax()])
 return np.array(result)
def measure(t,p):
 d=np.abs(t-p);exact=d.max(axis=1)==0;edge=(t[:,0]==0)|(t[:,0]==19)|(t[:,1]==0)|(t[:,1]==12)
 corner=((t[:,0]==0)|(t[:,0]==19))&((t[:,1]==0)|(t[:,1]==12))
 samex=band(t[:,0],20)==band(p[:,0],20);samey=band(t[:,1],13)==band(p[:,1],13)
 center=(band(p[:,0],20)==1)&(band(p[:,1],13)==1)
 reward=(10+8*edge+4*corner)*exact+2*(d[:,0]==0)+2*(d[:,1]==0)+1.5*samex+1.5*samey+1*(d.max(axis=1)<=1)-4*(edge&center)-1*(d.max(axis=1)>=6)
 return dict(n=len(t),exact_cell=float(exact.mean()),x_exact=float((d[:,0]==0).mean()),y_exact=float((d[:,1]==0).mean()),x_third_correct=float(samex.mean()),y_third_correct=float(samey.mean()),within_1_cell=float((d.max(axis=1)<=1).mean()),mean_distance_cells=float(np.sqrt((d*d).sum(axis=1)).mean()),far_error_rate=float((d.max(axis=1)>=6).mean()),edge_n=int(edge.sum()),edge_exact=float(exact[edge].mean()) if edge.any() else 0,edge_to_middle_rate=float(center[edge].mean()) if edge.any() else 0,center_prediction_rate=float(center.mean()),selection_reward=float(reward.mean()))
def fit(mask,c):
 mean=X[mask].mean(axis=0);scale=np.maximum(X[mask].std(axis=0),.03);T=(X[mask]-mean)/scale
 models=[]
 for k,n in [(0,20),(1,13)]:
  labels=y[mask,k]
  if c['family']=='logistic':
   m=LogisticRegression(C=c['C'],solver='lbfgs',max_iter=1500,class_weight='balanced')
   frame=(labels==0)|(labels==n-1);weights=w[mask]*(1+c['edge_weight']*frame)
   m.fit(T,labels,sample_weight=weights)
  elif c['family']=='lda':
   # Balanced class priors retain equal status for the frame and the middle.
   classes=np.unique(labels);priors=np.ones(len(classes))
   priors[(classes==0)|(classes==n-1)] *= 1+c['edge_weight']
   priors/=priors.sum()
   m=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=priors);m.fit(T,labels)
  else:
   m=lgb.LGBMClassifier(objective='multiclass',n_estimators=250,learning_rate=.035,num_leaves=7,max_depth=3,min_child_samples=4,reg_lambda=3,colsample_bytree=.85,n_jobs=2,verbosity=-1,random_state=20261107,deterministic=True,force_col_wise=True)
   weights=w[mask]*(1+c['edge_weight']*((labels==0)|(labels==n-1)))
   m.fit(T,labels,sample_weight=weights)
  models.append(m)
 return dict(models=models,mean=mean,scale=scale)
def probabilities(m,A):
 T=(A-m['mean'])/m['scale'];result=[]
 for model,n in zip(m['models'],[20,13]):
  p=np.zeros((len(A),n));p[:,model.classes_.astype(int)]=model.predict_proba(T);result.append(p)
 return result
configs=[]
for edge in [0,2]:
 for C in [.01,.1,1]:configs.append(dict(family='logistic',C=C,edge_weight=edge))
 configs.append(dict(family='lda',edge_weight=edge))
 configs.append(dict(family='gbm',edge_weight=edge))
results=[]
for c in configs:
 mask=train # Manual confirmations have reduced weight for supported estimators.
 m=fit(mask,c);prob=probabilities(m,X[val])
 for mode in ['map','hierarchical']:
  p=decode(prob,mode);entry=dict(config=c,decoder=mode,validation=measure(y[val],p));results.append(entry);print(json.dumps(entry),flush=True)
best=max(results,key=lambda r:r['validation']['selection_reward']);c=best['config'];mode=best['decoder']
evaluation_model=fit(idx<460,c);pred=decode(probabilities(evaluation_model,X[test]),mode);evaluation=measure(y[test],pred)
old=json.load(open(ROOT/'heldout-predictions.json'));lookup={r['trial_index']:r for r in old};oldp=[]
for i in idx[test]:
 r=lookup[int(i)];oldp.append([np.clip(np.floor(r['pred_col_float']),0,19),np.clip(np.floor(r['pred_row_float']),0,12)])
previous=measure(y[test],np.array(oldp))
final=fit(np.ones(len(X),bool),c)
export=dict(schema=3,kind='axes' if c['family']=='gbm' else 'linear_axes',decoder=mode,cols=20,rows=13,feature_count=X.shape[1],feature_version=1,mean=final['mean'].tolist(),scale=final['scale'].tolist(),loss='class-balanced per-axis classification; extra edge training weight; discrete hierarchical reward for exact cell, X, Y, thirds, and edge/corner hits; no coordinate averaging')
if c['family']=='gbm':export['heads']=[dict(classes=m.classes_.tolist(),trees=m.booster_.dump_model()['tree_info']) for m in final['models']]
else:export['linear_heads']=[dict(classes=m.classes_.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist()) for m in final['models']]
(ROOT/'model-classifier.json').write_text(json.dumps(export,separators=(',',':')))
fixture=json.load(open(ROOT/'parity-fixtures.json'))
for case in fixture:
 case['prediction']=(decode(probabilities(final,np.asarray(case['features'])[None,:]),mode)[0]+.5).tolist()
(ROOT/'classifier-parity-fixtures.json').write_text(json.dumps(fixture,separators=(',',':')))
report=dict(selected=best,evaluation=evaluation,previous_regression_same_test=previous,training_n=len(X),experiments=results,reward='10 exact +8 exact on frame +4 exact corner +2 correct X +2 correct Y +1.5 correct horizontal third +1.5 correct vertical third +1 within one cell -4 edge-to-middle -1 error >=6 cells',evaluation_warning='Reuses the v1 chronological test; validation selects parameters. A new user session is still needed for independent confirmation.')
(ROOT/'classifier-metrics.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps(dict(selected=best,evaluation=evaluation,previous=previous)),flush=True)
