"""Trigger-relative high-rate dynamics, with separate early and decay phases."""
import numpy as np
import common


def get_features(data):
    h=np.asarray(data['H'], dtype=float)
    t=np.linspace(-.65,.5,h.shape[1])
    base=h[:,(t>=-.20)&(t<=-.03)].mean(axis=1)
    z=h-base[:,None,:]
    dt=float(t[1]-t[0])
    der=np.gradient(z,dt,axis=1)
    def sampled(a, times):
        ix=np.abs(t[:,None]-np.asarray(times)[None,:]).argmin(axis=0)
        return a[:,ix,:].reshape(len(h),-1)
    feats=[base, sampled(z,np.linspace(0,.125,51)),
           sampled(der,np.linspace(0,.125,26)),
           sampled(z,np.linspace(.1375,.4,22))]
    for a,b in [(-.10,0),(0,.025),(.025,.05),(.05,.075),(.075,.125),(.125,.2),(.2,.3),(.3,.4)]:
        sel=(t>=a)&(t<b);v=z[:,sel];dv=der[:,sel]
        feats.extend([v.mean(1),v.std(1),v.min(1),v.max(1),np.sqrt((v*v).mean(1)),np.mean(np.abs(dv),axis=1)])
    # Timing is a feature, never an alignment operation.
    for a,b in [(0,.125),(.125,.4),(0,.4)]:
        sel=(t>=a)&(t<=b);v=z[:,sel];ts=t[sel]
        peak=np.abs(v).argmax(axis=1)
        feats.extend([ts[peak],np.take_along_axis(v,peak[:,None,:],axis=1)[:,0,:]])
        for channels in [slice(0,3),slice(3,6)]:
            energy=np.sum(v[:,:,channels]**2,axis=2)
            feats.extend([ts[energy.argmax(1)][:,None],
                          (np.sum(energy*ts,axis=1)/(energy.sum(1)+1e-12))[:,None]])
    feats.append(np.asarray(data['M'],dtype=float))
    return np.nan_to_num(np.concatenate(feats,axis=1),nan=0,posinf=0,neginf=0)


if __name__=='__main__':
    x=get_features(common.load())
    common.evaluate('exp06_svm',x,dict(family='svm',C=10,gamma='scale'))
    common.evaluate('exp06_trees',x,dict(family='trees',n_estimators=350,min_samples_leaf=2,max_features=.6))
