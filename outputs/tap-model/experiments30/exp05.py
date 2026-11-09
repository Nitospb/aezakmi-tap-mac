"""Earliest adaptive acceleration-onset alignment; no labels used by features."""
import numpy as np
import common
HTIME=np.linspace(-.65,.5,921)

def get_features(data):
    H=np.asarray(data['H'],float); G=np.asarray(data['G'],float)
    bg=(HTIME>=-.58)&(HTIME<=-.25)
    search=np.flatnonzero((HTIME>=-.20)&(HTIME<=.18))
    offsets=np.linspace(-.040,.180,177)
    rows=[]
    for h,g in zip(H,G):
        center=np.median(h[bg],axis=0)
        z=h-center
        # A short causal average suppresses interpolated single-sample glitches.
        acc=np.linalg.norm(z[:,3:],axis=1)
        smooth=np.convolve(acc,np.ones(3)/3,mode='same')
        noise=np.median(smooth[bg]); mad=1.4826*np.median(np.abs(smooth[bg]-noise))
        threshold=noise+max(6*mad,.0002)
        crossing=(smooth>threshold)
        sustained=crossing & np.roll(crossing,-1) & np.roll(crossing,-2)
        candidates=search[sustained[search]]
        onset=int(candidates[0]) if len(candidates) else int(search[np.argmax(smooth[search])])
        t=HTIME[onset]; sample_t=t+offsets
        segment=np.column_stack([np.interp(sample_t,HTIME,z[:,j]) for j in range(6)])
        scales=np.array([max(np.quantile(np.linalg.norm(segment[:,:3],axis=1),.9),.005),max(np.quantile(np.linalg.norm(segment[:,3:],axis=1),.9),.0001)])
        norm=segment/np.repeat(scales,3)
        gyro=np.linalg.norm(z[:,:3],axis=1)
        post=np.flatnonzero((HTIME>=t)&(HTIME<=t+.18))
        gp=post[np.argmax(gyro[post])]; ap=post[np.argmax(acc[post])]
        lid=np.interp(sample_t,common.TIME,g)
        lid=lid-np.median(g[common.TIME<-.25])
        derivative=np.gradient(lid,offsets)
        ls=max(np.quantile(np.abs(derivative),.9),.0001)
        feats=[norm.ravel(),lid[::4],(derivative/ls)[::2],np.log1p(scales*1000),[t,HTIME[gp]-t,HTIME[ap]-t,np.log1p(ls),noise,threshold]]
        # Signed early waveform integrals and extrema retain direction at impact.
        for a,b in [(-.04,0),(0,.015),(.015,.04),(.04,.08),(.08,.18)]:
            s=norm[(offsets>=a)&(offsets<b)]
            feats.extend([s.mean(0),s.min(0),s.max(0),np.sqrt((s*s).mean(0))])
        rows.append(np.concatenate([np.ravel(x) for x in feats]))
    return np.nan_to_num(np.asarray(rows),nan=0,posinf=0,neginf=0)

if __name__=='__main__':
    d=common.load(); X=get_features(d)
    common.evaluate('exp05_svm',X,dict(family='svm',C=10,gamma='scale'))
    common.evaluate('exp05_trees',X,dict(family='trees',n_estimators=400,min_samples_leaf=2,max_features=.6))
