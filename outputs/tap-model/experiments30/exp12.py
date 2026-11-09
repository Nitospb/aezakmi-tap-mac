"""Noise-aware impulse whitening, robust pretap baseline, SNR/sign/duration."""
import numpy as np
import common

def get_features(data):
    W=np.asarray(data['W'],float)
    t=common.TIME
    pre=W[:,(t>=-.60)&(t<=-.16)]
    baseline=np.median(pre,axis=1)
    noise=1.4826*np.median(np.abs(pre-baseline[:,None,:]),axis=1)
    z=W-baseline[:,None,:]
    active=(t>=-.10)&(t<=.20)
    pulse=z[:,active]
    peak=np.max(np.abs(pulse),axis=1)
    # Bound amplification by the individual pulse and a numerical floor.
    den=np.maximum(noise, np.maximum(peak*.12,1e-7))
    q=np.clip(z/den[:,None,:],-12,12)
    snr=peak/np.maximum(noise,1e-7)
    blocks=[np.asarray(data['M'],float),np.log1p(snr),np.log1p(noise),baseline]
    at=np.argmax(np.abs(pulse),axis=1)
    pv=np.take_along_axis(pulse,at[:,None,:],axis=1)[:,0,:]
    blocks.extend([np.sign(pv),pv/np.maximum(peak,1e-7),t[active][at]])
    for lo,hi in [(-.10,-.025),(-.025,.025),(.025,.075),(.075,.15),(.15,.30),(-.05,.20)]:
        s=q[:,(t>=lo)&(t<hi)]
        blocks.extend([np.mean(s,axis=1),np.std(s,axis=1),np.max(s,axis=1),np.min(s,axis=1),np.mean(s*s,axis=1)])
    pa=q[:,active]
    for fraction in [.2,.5,.8]:
        above=np.abs(pulse)>=fraction*peak[:,None,:]
        blocks.append(above.mean(axis=1))
        blocks.append((above*np.sign(pulse)).mean(axis=1))
    # Signed coarse waveform retains temporal shape after noise normalization.
    sel=q[:,(t>=-.075)&(t<=.225)]
    for chunk in np.array_split(sel,20,axis=1):blocks.append(chunk.mean(axis=1))
    return np.nan_to_num(np.column_stack(blocks),nan=0,posinf=1e6,neginf=-1e6)

if __name__=='__main__':
    d=common.load();X=get_features(d)
    common.evaluate('exp12_tree',X,dict(family='trees',n_estimators=360,min_samples_leaf=3,max_features=.65))
    common.evaluate('exp12_svm',X,dict(family='svm',C=3,gamma='scale'))
