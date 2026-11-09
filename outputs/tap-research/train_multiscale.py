"""Train and evaluate nested 2x2/3x3/4x4 regions alongside fine X/Y classes."""
from pathlib import Path
import json,numpy as np
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent
z=np.load(ROOT.parent.parent/'work/classification-data.npz');X=z['X'];y=z['y'];w=z['w'];idx=z['idx'];auto=w==1
train=idx<340;val=(idx>=340)&(idx<460)&auto;test=(idx>=460)&auto
CELLS=np.array([(c,r) for r in range(13) for c in range(20)])
def zones(A,n):return np.minimum(n-1,((A[:,1]+.5)*n/13).astype(int))*n+np.minimum(n-1,((A[:,0]+.5)*n/20).astype(int))
def fit(mask,labels,config,mean,scale):
 A=(X[mask]-mean)/scale
 if config['family']=='linear':m=LogisticRegression(C=config['C'],solver='lbfgs',max_iter=1200,class_weight='balanced')
 else:m=lgb.LGBMClassifier(objective='multiclass',learning_rate=.04,n_estimators=200,num_leaves=7,max_depth=3,min_child_samples=6,reg_lambda=4,colsample_bytree=.85,n_jobs=2,verbosity=-1,random_state=20261107,deterministic=True,force_col_wise=True)
 m.fit(A,labels[mask],sample_weight=w[mask]);return m

def evaluate(t,p):
 d=np.abs(t-p);m=dict(n=len(t),exact_cell=float((d.max(axis=1)==0).mean()),within_one_cell=float((d.max(axis=1)<=1).mean()),mean_distance_cells=float(np.sqrt((d*d).sum(axis=1)).mean()))
 total=10*(d.max(axis=1)==0).astype(float)
 for n,points in [(2,1),(3,2),(4,3)]:
  hit=zones(t,n)==zones(p,n);m[str(n)+'x'+str(n)]=float(hit.mean());total+=points*hit
 sx=(t[:,0]>=10)==(p[:,0]>=10);sy=(t[:,1]>=6.5)==(p[:,1]>=6.5)
 m['side_x']=float(sx.mean());m['side_y']=float(sy.mean());total+=.5*sx+.5*sy
 edge=(t[:,0]==0)|(t[:,0]==19)|(t[:,1]==0)|(t[:,1]==12);total+=4*edge*(d.max(axis=1)==0)
 m['multiscale_points']=float(total.mean());return m

def normalize(mask):return X[mask].mean(axis=0),np.maximum(X[mask].std(axis=0),.03)
def probs(m,A,mean,scale):return m.predict_proba((A-mean)/scale)
def assemble(mask,choices):
 mean,scale=normalize(mask);fine=[]
 for k in [0,1]:fine.append(fit(mask,y[:,k],dict(family='linear',C=1),mean,scale))
 region=[fit(mask,zones(y,n),choices[n],mean,scale) for n in [2,3,4]]
 return mean,scale,fine,region

def decode(bundle,A,weight):
 mean,scale,fine,region=bundle
 px=probs(fine[0],A,mean,scale);py=probs(fine[1],A,mean,scale)
 score=np.log(np.maximum(px[:,CELLS[:,0]],1e-12))+np.log(np.maximum(py[:,CELLS[:,1]],1e-12))
 for n,m in zip([2,3,4],region):score+=weight*np.log(np.maximum(probs(m,A,mean,scale)[:,zones(CELLS,n)],1e-12))
 return CELLS[score.argmax(axis=1)]
configs=[dict(family='linear',C=.1),dict(family='linear',C=1),dict(family='gbm')]
mean,scale=normalize(train);choices={};experiments=[]
for n in [2,3,4]:
 trial=[]
 for c in configs:
  m=fit(train,zones(y,n),c,mean,scale);acc=float((m.predict((X[val]-mean)/scale)==zones(y[val],n)).mean());e=dict(grid=[n,n],config=c,validation_accuracy=acc);trial.append(e);experiments.append(e);print(json.dumps(e),flush=True)
 choices[n]=max(trial,key=lambda e:e['validation_accuracy'])['config']
bundle=assemble(train,choices);fusions=[]
for weight in [0,.25,.5,1]:
 score=evaluate(y[val],decode(bundle,X[val],weight));fusions.append(dict(weight=weight,validation=score));print(json.dumps(fusions[-1]),flush=True)
best=max(fusions,key=lambda e:e['validation']['multiscale_points']);weight=best['weight']
eval_bundle=assemble(idx<460,choices);evaluation=evaluate(y[test],decode(eval_bundle,X[test],weight))
direct={}
for n,m in zip([2,3,4],eval_bundle[3]):direct[str(n)+'x'+str(n)]=float((m.predict((X[test]-eval_bundle[0])/eval_bundle[1])==zones(y[test],n)).mean())
final=assemble(np.ones(len(X),bool),choices);mean,scale,fine,region=final
export=dict(schema=4,kind='linear_axes',decoder='multiscale',feature_count=X.shape[1],feature_version=1,cols=20,rows=13,mean=mean.tolist(),scale=scale.tolist(),multiscale_weight=weight)
export['linear_heads']=[dict(classes=m.classes_.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist()) for m in fine]
export['region_heads']=[]
for n,m in zip([2,3,4],region):
 h=dict(nx=n,ny=n,classes=m.classes_.tolist())
 if hasattr(m,'coef_'):h.update(kind='linear',coef=m.coef_.tolist(),intercept=m.intercept_.tolist())
 else:h.update(kind='trees',trees=m.booster_.dump_model()['tree_info'])
 export['region_heads'].append(h)
(ROOT/'model-multiscale.json').write_text(json.dumps(export,separators=(',',':')))
fixture=json.load(open(ROOT.parent/'tap-model/parity-fixtures.json'))
for case in fixture:
 A=np.asarray(case['features'])[None,:];case['prediction']=(decode(final,A,weight)[0]+.5).tolist();case['independent_regions']=[dict(n=n,zone=int(m.predict((A-mean)/scale)[0])) for n,m in zip([2,3,4],region)];xx,yy=case['prediction'];case['regions']=[dict(n=n,zone=min(n-1,int(yy*n/13))*n+min(n-1,int(xx*n/20))) for n in [2,3,4]]
(ROOT/'multiscale-parity-fixtures.json').write_text(json.dumps(fixture,separators=(',',':')))
report=dict(choices=choices,fusion=best,region_experiments=experiments,fusion_experiments=fusions,evaluation=evaluation,independent_region_head_accuracy=direct,score='10 exact-cell +1 correct 2x2 region +2 correct 3x3 region +3 correct 4x4 region +0.5 correct left/right side +0.5 correct upper/lower side +4 extra exact-on-frame',warning='Region hits are coarse labels, not exact-cell accuracy. 3x3 here means a fixed 9-region partition of the full screen, not the 8 neighboring fine cells. Same historical test, not new independent validation.')
(ROOT/'multiscale-metrics.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
