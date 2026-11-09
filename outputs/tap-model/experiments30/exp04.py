"""Joint-peak aligned temporal pyramid: high-rate motion, lid dynamics, light context."""
import numpy as np
import common

HTIME=np.linspace(-.65,.5,921)

def get_features(data):
    h=np.asarray(data['H'],float)
    base=h[:,HTIME<-.12].mean(axis=1,keepdims=True)
    z=h-base
    # A per-record scale balances gyro and accelerometer without any label dependence.
    energy=np.linalg.norm(z[:,:,:3],axis=2)/(np.sqrt(np.mean(z[:,:,:3]**2,axis=(1,2)))[:,None]+1e-8)
    energy+=np.linalg.norm(z[:,:,3:],axis=2)/(np.sqrt(np.mean(z[:,:,3:]**2,axis=(1,2)))[:,None]+1e-8)
    search=np.flatnonzero((HTIME>=-.12)&(HTIME<=.12))
    peaks=search[np.argmax(energy[:,search],axis=1)]
    features=[base[:,0,:],HTIME[peaks,None]]
    for lo,hi in [(-.04,.06),(-.04,.15),(-.04,.35)]:
        offsets=np.arange(round(lo/.00125),round(hi/.00125)+1)
        ix=np.clip(peaks[:,None]+offsets[None,:],0,h.shape[1]-1)
        a=np.take_along_axis(z,ix[:,:,None],axis=1)
        n=a.shape[1]
        absolute=np.abs(a); power=np.sum(a*a,axis=1)
        pos=np.argmax(absolute,axis=1)
        signed_peak=np.take_along_axis(a,pos[:,None,:],axis=1)[:,0,:]
        features += [a.mean(1),a.std(1),a.min(1),a.max(1),signed_peak,np.sqrt((a*a).mean(1)),absolute.mean(1),a.sum(1)*.00125, pos/(n-1),a[:,0],a[:,-1]]
        diff=np.diff(a,axis=1)
        features += [np.abs(diff).mean(1),diff.std(1),((a[:,1:]*a[:,:-1])<0).mean(1)]
        # Normalized energy centroids and skew distinguish impulse direction and ringdown.
        t=np.linspace(0,1,n)[None,:,None]
        features += [(a*a*t).sum(1)/(power+1e-10), (a**3).mean(1)/(a.std(1)**3+1e-10)]
        centered=a-a.mean(1,keepdims=True)
        cov=np.einsum('nti,ntj->nij',centered,centered)/n
        ii,jj=np.triu_indices(6)
        features.append(cov[:,ii,jj])
        sd=a.std(1)
        features.append(cov[:,ii,jj]/(sd[:,ii]*sd[:,jj]+1e-10))
        for group in [slice(0,3),slice(3,6)]:
            norm=np.linalg.norm(a[:,:,group],axis=2)
            features.append(np.column_stack([norm.mean(1),norm.max(1),norm.std(1),np.mean(norm[:,:max(1,n//3)]**2,1)/(np.mean(norm**2,1)+1e-10),np.mean(norm[:,n//2:]**2,1)/(np.mean(norm**2,1)+1e-10)]))
        # Lid follows the same absolute peak alignment on its native sample times.
        g=np.asarray(data['G'],float)
        gt=np.linspace(-.65,.5,g.shape[1])
        gg=np.stack([np.interp(HTIME[ix[k]],gt,g[k]) for k in range(len(g))])
        gd=np.diff(gg,axis=1)/.00125
        features.append(np.column_stack([gg.mean(1),gg.std(1),gg[:,0],gg[:,-1],gg[:,-1]-gg[:,0],gd.mean(1),gd.std(1),gd.min(1),gd.max(1),np.abs(gd).mean(1)]))
    features.append(np.asarray(data['F'],float))
    return np.nan_to_num(np.concatenate(features,axis=1),nan=0,posinf=0,neginf=0)

if __name__=='__main__':
    d=common.load()
    x=get_features(d)
    common.evaluate('exp04_tree',x,dict(family='trees',n_estimators=420,min_samples_leaf=2,max_features=.65))
    common.evaluate('exp04_svm',x,dict(family='svm',C=10,gamma='scale'))
