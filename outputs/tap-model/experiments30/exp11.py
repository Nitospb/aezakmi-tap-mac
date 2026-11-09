"""Gravity-aligned inertial representation; no labels or metadata as features."""
import numpy as np
import common


def get_features(data):
    raw=np.asarray(data['W'],dtype=float)
    t=common.TIME
    pre=(t<-.20)&(t>=-.60)
    baseline=np.median(raw[:,pre,:],axis=1)
    gravity=baseline[:,3:]
    gn=np.linalg.norm(gravity,axis=1,keepdims=True)
    vertical=gravity/np.maximum(gn,1e-8)
    # Project the body X direction onto the plane normal to gravity.
    seed=np.tile(np.array([1.,0.,0.]),(len(raw),1))
    parallel=np.abs(vertical[:,0])>.9
    seed[parallel]=np.array([0.,1.,0.])
    horizontal=seed-np.sum(seed*vertical,axis=1,keepdims=True)*vertical
    horizontal/=np.maximum(np.linalg.norm(horizontal,axis=1,keepdims=True),1e-8)
    horizontal2=np.cross(vertical,horizontal)
    basis=np.stack([horizontal,horizontal2,vertical],axis=2)
    centered=raw-baseline[:,None,:]
    rotated=np.concatenate([np.einsum('ntc,ncd->ntd',centered[:,:,:3],basis),np.einsum('ntc,ncd->ntd',centered[:,:,3:],basis)],axis=2)
    noise=np.std(rotated[:,pre,:],axis=1)
    feats=[np.asarray(data['M']),np.asarray(data['F']),baseline,gn,vertical,noise]
    # Physical units and normalized shapes are both retained.
    for z in (rotated,centered):
        for lo,hi in [(-.60,-.2),(-.15,-.03),(-.03,.025),(.025,.075),(.075,.15),(.15,.3),(.3,.5),(-.05,.3)]:
            a=z[:,(t>=lo)&(t<hi),:]
            feats += [a.mean(1),a.std(1),a.min(1),a.max(1),np.sqrt(np.mean(a*a,axis=1)),np.sum(a,axis=1)*(t[1]-t[0])]
        # Signed low-frequency waveform keeps impulse and ringing direction.
        for ids in np.array_split(np.where((t>=-.08)&(t<=.32))[0],20):
            feats.append(z[:,ids,:].mean(1))
    # Cross-axis relations in gravity frame, with original physical scale.
    active=rotated[:,(t>=-.03)&(t<=.2),:]
    for i in range(6):
        for j in range(i+1,6):
            feats.append(np.mean(active[:,:,i]*active[:,:,j],axis=1,keepdims=True))
    peak=np.max(np.abs(active),axis=1)
    feats.extend([peak,peak/np.maximum(noise,1e-7),np.log1p(peak/np.maximum(noise,1e-7))])
    return np.nan_to_num(np.concatenate(feats,axis=1),nan=0.,posinf=0.,neginf=0.)


if __name__=='__main__':
    d=common.load();X=get_features(d)
    common.evaluate('exp11_tree',X,dict(family='trees',n_estimators=400,min_samples_leaf=2,max_features=.5))
    common.evaluate('exp11_svm',X,dict(family='svm',C=10,gamma='scale'))
