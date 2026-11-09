"""Axis-specific physical feature subsets, training-only reduction of shared M.
Deployment: get_features -> fit_models/predict_probabilities. No target/cue input.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import common

BASE=652

def _parts(d):
    w=np.asarray(d['W'],float)
    t=common.TIME
    base=w[:,(t>=-.30)&(t<=-.08)].mean(1)
    z=w-base[:,None,:]
    g=np.asarray(d['G'],float)
    x=[];y=[]
    for lo,hi in [(0,.04),(.04,.10),(.10,.22),(-.025,.16)]:
        q=z[:,(t>=lo)&(t<=hi)]
        gyro=q[:,:,:3];acc=q[:,:,3:]
        gm=gyro.mean(1);am=acc.mean(1)
        ge=np.sqrt((gyro**2).mean(1)+1e-8)
        ae=np.sqrt((acc**2).mean(1)+1e-8)
        # Signed rotational response and twist proportions capture lateral mechanics.
        x.extend([gm, gm/(np.linalg.norm(gm,axis=1,keepdims=True)+.04),
                  gyro.max(1)+gyro.min(1), np.cross(gm,am)/(np.linalg.norm(am,axis=1,keepdims=True)+.08)])
        # Pitch/vertical impulses and rotation-to-linear response for vertical position.
        y.extend([am, ge, gm/(ae+.08), am/(np.linalg.norm(am,axis=1,keepdims=True)+.08)])
        cov=np.mean(gyro*acc,axis=1)
        x.append(cov[:,[0,2]])
        y.append(cov[:,[1]])
        gg=g[:,(t>=lo)&(t<=hi)]
        y.append(np.column_stack([gg.mean(1)-g[:,t<-.08].mean(1),gg[:,-1]-gg[:,0],np.ptp(gg,axis=1)]))
    return np.concatenate(x,axis=1),np.concatenate(y,axis=1)

def get_features(d):
    x,y=_parts(d)
    return np.nan_to_num(np.column_stack([d['M'],x,y]),nan=0,posinf=0,neginf=0)

XEND=652+56

def fit_models(X,y,w,config):
    models=[]
    for k in range(2):
        physical=X[:,BASE:XEND] if k==0 else X[:,XEND:]
        # The 652 legacy features are reduced separately using only fitting labels.
        sel=SelectKBest(f_classif,k=config.get('m_k',80))
        reduced=sel.fit_transform(X[:,:BASE],y[:,k])
        a=np.column_stack([reduced,physical])
        scale=StandardScaler().fit(a);a=scale.transform(a)
        if config['family']=='svm':
            model=SVC(C=config['C'],gamma='scale',probability=True,class_weight='balanced',random_state=common.SEED)
        else:
            model=ExtraTreesClassifier(n_estimators=360,min_samples_leaf=2,max_features=.65,class_weight='balanced',n_jobs=1,random_state=common.SEED)
        model.fit(a,y[:,k],sample_weight=w)
        models.append((sel,scale,model))
    return models

def predict_probabilities(models,X):
    out=[]
    for k,(sel,scale,model) in enumerate(models):
        physical=X[:,BASE:XEND] if k==0 else X[:,XEND:]
        a=scale.transform(np.column_stack([sel.transform(X[:,:BASE]),physical]))
        p=np.zeros((len(X),[20,13][k]));p[:,model.classes_.astype(int)]=model.predict_proba(a);out.append(p)
    return out

if __name__=='__main__':
    d=common.load();X=get_features(d)
    train=d['idx']<260;val=(d['idx']>=260)&(d['idx']<520)&(d['w']==1)
    for name,c in [('exp26_axis_tree',dict(family='trees',m_k=80)),('exp26_axis_svm',dict(family='svm',m_k=80,C=10))]:
        models=fit_models(X[train],d['y'][train],d['w'][train],c)
        p=predict_probabilities(models,X[val])
        common.save_result(name,X,c,models,p,d['y'][val],d['idx'][val],extra={'axis_physical_features':[56,64],'shared_selected_per_axis':80,'deployment':'get_features, fit_models, predict_probabilities'})
