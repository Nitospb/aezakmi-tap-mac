"""Shared protocol for 30 no-microphone experiments. No evaluation on pass3 here."""
from pathlib import Path
import json,time,os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent
SEED=20261108
TIME=np.linspace(-.65,.5,461)
LIGHT_TIME=np.linspace(-.65,.5,12)
def load():return dict(np.load(ROOT/'data.npz'))
def metrics(t,p):
 d=np.abs(t-p);dist=np.linalg.norm(d,axis=1);edge=(t[:,0]==0)|(t[:,0]==19)|(t[:,1]==0)|(t[:,1]==12)
 m=dict(n=len(t),exact=float((d.max(1)==0).mean()),within_one=float((d.max(1)<=1).mean()),within_two=float((d.max(1)<=2).mean()),mean_distance_cells=float(dist.mean()),median_distance_cells=float(np.median(dist)),far_half_screen=float(((d[:,0]>=10)|(d[:,1]>=7)).mean()),far_third_screen=float(((d[:,0]>=7)|(d[:,1]>=5)).mean()),side_x=float(((t[:,0]>=10)==(p[:,0]>=10)).mean()),side_y=float(((t[:,1]>=6.5)==(p[:,1]>=6.5)).mean()),edge_exact=float((d[edge].max(1)==0).mean()) if edge.any() else 0,unique_predicted_cells=int(len(np.unique(p,axis=0))))
 m['score']=3*m['exact']+m['within_one']+.25*m['within_two']-.5*m['far_half_screen']-.1*float(np.linalg.norm(d/np.array([20,13]),axis=1).mean())
 return m

def fit(X,y,w,config):
 models=[]
 for k in range(y.shape[1]):
  c=config;family=c.get('family','trees')
  if family=='svm':m=SVC(C=c.get('C',10),gamma=c.get('gamma','scale'),probability=True,class_weight='balanced',random_state=SEED)
  elif family=='linear':m=LogisticRegression(C=c.get('C',.1),max_iter=1500,class_weight='balanced')
  elif family=='forest':m=RandomForestClassifier(n_estimators=c.get('n_estimators',300),min_samples_leaf=c.get('min_samples_leaf',2),max_features=c.get('max_features','sqrt'),class_weight='balanced_subsample',n_jobs=1,random_state=SEED)
  elif family=='gbm':m=lgb.LGBMClassifier(n_estimators=c.get('n_estimators',180),learning_rate=c.get('learning_rate',.04),num_leaves=c.get('num_leaves',7),max_depth=c.get('max_depth',3),min_child_samples=c.get('min_child_samples',5),reg_lambda=c.get('reg_lambda',4),n_jobs=1,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)
  else:m=ExtraTreesClassifier(n_estimators=c.get('n_estimators',220),min_samples_leaf=c.get('min_samples_leaf',2),max_features=c.get('max_features',.6),class_weight='balanced',n_jobs=1,random_state=SEED)
  stages=[StandardScaler()]
  if c.get('select_k'):stages.append(SelectKBest(f_classif,k=min(int(c['select_k']),X.shape[1])))
  if c.get('pca'):stages.append(PCA(n_components=c['pca'],whiten=c.get('whiten',False),svd_solver='full'))
  stages.append(m);model=make_pipeline(*stages);weights=w.copy()
  if c.get('edge_weight'):weights*=1+c['edge_weight']*((y[:,0]==0)|(y[:,0]==19)|(y[:,1]==0)|(y[:,1]==12))
  model.fit(X,y[:,k],**{list(model.named_steps)[-1]+'__sample_weight':weights});models.append(model)
 return models

def probabilities(models,X,classes=(20,13)):
 out=[]
 for k,m in enumerate(models):
  p=np.zeros((len(X),classes[k]));p[:,m.classes_.astype(int)]=m.predict_proba(X);out.append(p)
 return out

def decode(p):return np.column_stack([q.argmax(1) for q in p])
def save_result(name,X,config,models,p,truth,indices,extra=None):
 report=dict(name=name,feature_count=X.shape[1],config=config,validation=metrics(truth,decode(p)),extra=extra or {})
 (ROOT/f'{name}-result.json').write_text(json.dumps(report,indent=2));np.savez_compressed(ROOT/f'{name}-validation.npz',px=p[0],py=p[1],truth=truth,idx=indices)
 print(json.dumps(report),flush=True);return report

def evaluate(name,X,config):
 d=load();X=np.asarray(X,float)
 if X.ndim!=2 or len(X)!=len(d['idx']) or not np.isfinite(X).all():raise ValueError('Invalid feature matrix')
 train=d['idx']<260;val=(d['idx']>=260)&(d['idx']<520)&(d['w']==1)
 if config.get('manual') is False:train &= d['w']==1
 models=fit(X[train],d['y'][train],d['w'][train],config);p=probabilities(models,X[val]);return save_result(name,X,config,models,p,d['y'][val],d['idx'][val])
