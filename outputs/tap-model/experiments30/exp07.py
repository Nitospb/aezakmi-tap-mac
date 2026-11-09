"""Haar impulse features on high-rate motion, with motion/light context."""
import numpy as np
import common

HTIME=np.linspace(-.65,.5,921)

def get_features(data):
    h=np.asarray(data['H'],dtype=float)
    pre=h[:,(HTIME>=-.30)&(HTIME<=-.05),:]
    center=np.median(pre,axis=1)
    scale=1.4826*np.median(np.abs(pre-center[:,None,:]),axis=1)
    # Sensor-wise floor avoids exploding coefficients on quiet stationary traces.
    scale=np.maximum(scale,np.array([.002,.002,.002,.005,.005,.005]))
    positions=np.linspace(np.searchsorted(HTIME,0),np.searchsorted(HTIME,.25),256)
    low=np.floor(positions).astype(int); high=np.minimum(low+1,h.shape[1]-1)
    a=positions-low
    raw=h[:,low,:]*(1-a)[None,:,None]+h[:,high,:]*a[None,:,None]-center[:,None,:]
    normalized=raw/scale[:,None,:]
    blocks=[np.asarray(data['M']),np.asarray(data['F']),center, np.log1p(scale)]
    # Preserve signed axis coefficients; magnitudes add orientation-invariant response.
    for signal in (raw,normalized):
        norm=np.stack([np.linalg.norm(signal[:,:,:3],axis=2),np.linalg.norm(signal[:,:,3:],axis=2)],axis=2)
        x=np.concatenate([signal,norm],axis=2)
        for level in range(8):
            detail=(x[:,0::2,:]-x[:,1::2,:])/np.sqrt(2.)
            x=(x[:,0::2,:]+x[:,1::2,:])/np.sqrt(2.)
            energy=np.mean(detail**2,axis=1)
            blocks.extend([np.log1p(energy),np.mean(np.abs(detail),axis=1),np.max(np.abs(detail),axis=1)])
            # Temporal ordering is explicit for the 8--250ms subbands.
            if detail.shape[1]<=32:
                blocks.append(detail.reshape(len(h),-1))
        blocks.append(x.reshape(len(h),-1))
    return np.nan_to_num(np.concatenate(blocks,axis=1),nan=0.,posinf=1e12,neginf=-1e12)

if __name__=='__main__':
    d=common.load();X=get_features(d)
    common.evaluate('exp07_tree',X,dict(family='trees',n_estimators=400,min_samples_leaf=2,max_features=.6))
    common.evaluate('exp07_svm',X,dict(family='svm',C=10,gamma='scale'))
