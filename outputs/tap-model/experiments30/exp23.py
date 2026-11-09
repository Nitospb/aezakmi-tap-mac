"""Confident independent side heads guard fine-axis predictions against far errors."""
from pathlib import Path
import json
import joblib
import numpy as np
import common

CONFIG={'family':'svm','C':10,'gamma':'scale','manual':True}

def get_features(data):
    return np.nan_to_num(np.column_stack([data['M'],data['F']]),nan=0,posinf=0,neginf=0)

def fit_side_heads(data,train_mask=None):
    if train_mask is None: train_mask=data['idx']<260
    X=get_features(data)
    sides=np.column_stack([data['y'][:,0]>=10,data['y'][:,1]>=7]).astype(int)
    return common.fit(X[train_mask],sides[train_mask],data['w'][train_mask],CONFIG)

def guard_probabilities(fine_probabilities,side_probabilities,threshold=.8):
    """Mask opposite half only on confident contradiction; retain every same-half cell."""
    out=[]
    for k,(fine,side) in enumerate(zip(fine_probabilities,side_probabilities)):
        q=np.array(fine,dtype=float,copy=True)
        cut=(10,7)[k]
        proposed=side.argmax(1)
        confident=side.max(1)>=threshold
        original_side=(fine.argmax(1)>=cut).astype(int)
        route=confident & (original_side!=proposed)
        for s in (0,1):
            rows=np.flatnonzero(route & (proposed==s))
            forbidden=np.flatnonzero((np.arange(q.shape[1])>=cut)!=s)
            q[np.ix_(rows,forbidden)]=0
        mass=q.sum(1)
        # Baselines with structural zeros can have no mass on proposed side.
        # In that case preserve the original distribution rather than invent a center.
        empty=mass<=0
        q[empty]=fine[empty]
        q/=q.sum(1,keepdims=True)
        out.append(q)
    return out

def predict_probabilities(models,data,fine_probabilities,threshold=.8):
    sides=common.probabilities(models,get_features(data),classes=(2,2))
    return guard_probabilities(fine_probabilities,sides,threshold)

def main():
    root=Path(__file__).resolve().parent
    d=common.load(); X=get_features(d)
    val=(d['idx']>=260)&(d['idx']<520)&(d['w']==1)
    baseline=dict(np.load(root/'baseline-current-validation.npz'))
    assert np.array_equal(baseline['idx'],d['idx'][val])
    assert np.array_equal(baseline['truth'],d['y'][val])
    heads=fit_side_heads(d)
    joblib.dump(heads,root/'exp23-side-heads.joblib')
    sides=common.probabilities(heads,X[val],classes=(2,2))
    fine=[baseline['px'],baseline['py']]
    for threshold in (.8,.9):
        name=f'exp23-guard{int(threshold*100)}'
        p=guard_probabilities(fine,sides,threshold)
        pred=common.decode(p)
        extra={'baseline':common.metrics(d['y'][val],common.decode(fine)),
               'changed_predictions':int(np.any(pred!=common.decode(fine),axis=1).sum()),
               'side_x_accuracy':float(((sides[0].argmax(1))==(d['y'][val,0]>=10)).mean()),
               'side_y_accuracy':float(((sides[1].argmax(1))==(d['y'][val,1]>=7)).mean()),
               'confident_x':int((sides[0].max(1)>=threshold).sum()),
               'confident_y':int((sides[1].max(1)>=threshold).sum())}
        common.save_result(name,X,{**CONFIG,'threshold':threshold,'routing':'confident contradiction hard half mask'},heads,p,d['y'][val],d['idx'][val],extra)
        np.savez_compressed(root/f'{name}-validation.npz',px=p[0],py=p[1],predictions=pred,truth=d['y'][val],idx=d['idx'][val],side_px=sides[0],side_py=sides[1])

if __name__=='__main__':main()
