"""Physical motion integrals; independent of trial metadata and labels."""
import numpy as np
import common


def get_features(data):
    t=common.TIME
    w=np.asarray(data['W'],float)
    g=np.asarray(data['G'],float)
    dt=t[1]-t[0]
    baseline=(t>=-.55)&(t<=-.2)
    centered=w-np.median(w[:,baseline,:],axis=1)[:,None,:]
    parts=[np.asarray(data['M'],float),np.asarray(data['F'],float)]
    # Signed integrated gyro is angular displacement; integrated acceleration
    # is velocity change. Baseline removal avoids persistent gravity offsets.
    for lo,hi in [(-.12,0),(0,.035),(0,.09),(0,.2),(0,.45),(-.06,.12)]:
        m=(t>=lo)&(t<=hi)
        q=centered[:,m,:]
        integral=np.trapz(q,dx=dt,axis=1)
        absolute=np.trapz(np.abs(q),dx=dt,axis=1)
        norms=np.linalg.norm(integral.reshape(-1,2,3),axis=2)
        direction=integral/(np.repeat(norms,3,axis=1)+1e-8)
        coherence=integral/(absolute+1e-8)
        velocity=np.cumsum(q[:,:,3:],axis=1)*dt
        displacement=np.trapz(velocity,dx=dt,axis=1)
        torque=np.cross(integral[:,:3],integral[:,3:])
        # Time centroid of energy captures impact/ringing asymmetry.
        energy=q*q
        centroid=np.sum(energy*t[m][None,:,None],axis=1)/(np.sum(energy,axis=1)+1e-8)
        parts.extend([integral,absolute,direction,coherence,norms,
                      displacement,torque,centroid])
    # Compare early impact displacement to subsequent ringing over each axis.
    integrals=[]
    for lo,hi in [(0,.035),(.035,.12),(.12,.3)]:
        integrals.append(np.trapz(centered[:,(t>=lo)&(t<=hi),:],dx=dt,axis=1))
    for a,b in zip(integrals[:-1],integrals[1:]):
        parts.extend([a*b/(np.abs(a)*np.abs(b)+1e-8),
                      a/(np.abs(a)+np.abs(b)+1e-8)])
    # The angle sensor has a low-amplitude, delayed but physically meaningful
    # response. Keep levels, angular increments, slopes and curvature separate.
    gb=np.median(g[:,baseline],axis=1)
    lid=g-gb[:,None]
    for lo,hi in [(-.12,0),(0,.04),(.04,.1),(.1,.2),(.2,.4)]:
        m=(t>=lo)&(t<=hi)
        z=lid[:,m];tt=t[m];tc=tt-tt.mean()
        slope=np.sum(z*tc[None,:],axis=1)/np.sum(tc*tc)
        diff=np.diff(z,axis=1)/dt
        parts.append(np.column_stack([z.mean(1),z[:,-1]-z[:,0],slope,
                      np.trapz(z,dx=dt,axis=1),np.max(z,axis=1),np.min(z,axis=1),
                      np.mean(diff,axis=1),np.std(diff,axis=1)]))
    return np.nan_to_num(np.column_stack(parts),nan=0.,posinf=0.,neginf=0.)

if __name__=='__main__':
    x=get_features(common.load())
    common.evaluate('exp09_svm',x,dict(family='svm',C=10,gamma='scale'))
    common.evaluate('exp09_trees',x,dict(family='trees',n_estimators=400,min_samples_leaf=2,max_features=.65))
