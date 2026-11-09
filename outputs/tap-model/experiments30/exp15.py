"""Shift-tolerant waveform templates; custom helpers, no audio.
Deployment: X=get_features(live_data), model=fit_templates(Xtrain,ytrain,wtrain,
{'k': 3}), p=template_probabilities(model,X); common.decode(p).
Refit only on parent-approved training partition, then pickle model dictionary.
"""
import numpy as np
from scipy.spatial.distance import cdist
import common

SHIFTS=np.arange(-4,5)  # 800Hz: +/-5ms

def get_features(data):
    h=np.asarray(data['H'],dtype=float)
    times=np.linspace(-.65,.5,h.shape[1])
    center=np.flatnonzero((times>=-.025-1e-8)&(times<=.15+1e-8))
    baseline=h[:,(times>=-.065)&(times<=-.03)].mean(axis=1,keepdims=True)
    centered=h-baseline
    variants=[]
    for shift in SHIFTS:
        a=centered[:,center+shift,:].copy()
        # Normalize the gyro and acceleration vector trajectories separately.
        # Preserve signed XYZ directions and relative amplitudes within each group.
        for sl in (slice(0,3),slice(3,6)):
            norm=np.sqrt(np.sum(a[:,:,sl]**2,axis=(1,2),keepdims=True))
            a[:,:,sl]/=np.maximum(norm,1e-12)
        flat=a.reshape(len(a),-1)
        flat/=np.maximum(np.linalg.norm(flat,axis=1,keepdims=True),1e-12)
        variants.append(flat)
    return np.stack(variants,axis=1).reshape(len(h),-1)

def fit_templates(X,y,w,config):
    v=X.reshape(len(X),len(SHIFTS),-1)
    return {'templates':v[:,len(SHIFTS)//2].copy(), 'y':np.array(y),
            'w':np.array(w), 'k':int(config['k'])}

def template_probabilities(model,X):
    v=X.reshape(len(X),len(SHIFTS),-1)
    distance=np.full((len(v),len(model['templates'])),np.inf)
    # Squared Euclidean on normalized vectors is twice cosine distance.
    # cdist bounds memory to one validation x training matrix per shift.
    for shift in range(len(SHIFTS)):
        distance=np.minimum(distance,cdist(v[:,shift],model['templates'],metric='sqeuclidean'))
    nearest=np.argsort(distance,axis=1,kind='stable')[:,:model['k']]
    weights=model['w'][nearest]
    out=[]
    for axis,classes in enumerate((20,13)):
        p=np.zeros((len(v),classes))
        for j in range(model['k']):
            np.add.at(p,(np.arange(len(v)),model['y'][nearest[:,j],axis]),weights[:,j])
        p/=p.sum(axis=1,keepdims=True)
        out.append(p)
    return out

if __name__=='__main__':
    d=common.load();X=get_features(d)
    train=d['idx']<260
    val=(d['idx']>=260)&(d['idx']<520)&(d['w']==1)
    for k in (1,3):
        config={'family':'shift_templates','k':k,'shift_samples':SHIFTS.tolist(),
                'window_seconds':[-.025,.15],'normalization':'separate gyro/accel trajectory L2',
                'distance':'minimum squared Euclidean across shifts (normalized cosine equivalent)',
                'voting':'sample-weighted discrete axis class vote'}
        model=fit_templates(X[train],d['y'][train],d['w'][train],config)
        p=template_probabilities(model,X[val])
        common.save_result(f'exp15_template_k{k}',X,config,model,p,d['y'][val],d['idx'][val],
          extra={'deployment':'Use exp15.get_features, fit_templates, template_probabilities; common.decode. Refit on approved training data, persist model dict.',
                 'notes':'Shift query only; training uses central template. Cosine and Euclidean equivalent after normalization. No coordinate averaging.'})
