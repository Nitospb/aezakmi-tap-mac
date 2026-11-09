"""Long pre-event detrended lid/optical coupling; no absolute optical levels."""
import numpy as np
from common import TIME,LIGHT_TIME,evaluate,load


def get_features(data):
    # Fit baseline through -150 ms, retaining a gap before the impact.
    g=np.asarray(data['G'],float)
    optical=np.log1p(np.maximum(np.asarray(data['L'],float),0))
    traces=[g]+[optical[:,:,j] for j in range(5)]
    times=[TIME]+[LIGHT_TIME]*5
    features=[]; relative=[]
    for j,(x,t) in enumerate(zip(traces,times)):
        b=(t<=-.15); tb=t[b]; centered=tb-tb.mean()
        mean=x[:,b].mean(1)
        slope=(x[:,b]*centered).sum(1)/(centered**2).sum()
        residual=x[:,b]-mean[:,None]-slope[:,None]*centered
        noise=np.sqrt(np.mean(residual**2,axis=1))
        scale=noise+(1e-5 if j==0 else .001)
        pred=mean[:,None]+slope[:,None]*(t-tb.mean())
        r=(x-pred)/scale[:,None]
        post=t>=0; tp=t[post]; rp=r[:,post]
        features.extend([slope/scale,noise/scale])
        for lo,hi in [(0,.1),(.1,.2),(.2,.35),(.35,.501)]:
            features.append(r[:,(t>=lo)&(t<hi)].mean(1))
        for lo,hi in [(0,.2),(.2,.501)]:
            mask=(t>=lo)&(t<hi); tc=t[mask]-t[mask].mean()
            features.append((r[:,mask]*tc).sum(1)/np.maximum((tc**2).sum(),1e-8))
        peak=np.argmax(abs(rp),axis=1)
        features.extend([np.sqrt(np.mean(rp**2,axis=1)),rp[np.arange(len(rp)),peak],tp[peak],np.mean(abs(rp),axis=1),np.mean(rp>0,axis=1),rp[:,-1]-rp[:,0]])
        relative.append(np.array([np.interp(LIGHT_TIME,t,row) for row in r]))
    # Native optical cadence, with one-sample lead/lag comparisons to lid motion.
    gp=relative[0][:,LIGHT_TIME>=0]
    for lp in relative[1:]:
        lp=lp[:,LIGHT_TIME>=0]
        for lag in [-1,0,1]:
            a,b=(gp[:,:-1],lp[:,1:]) if lag==1 else ((gp[:,1:],lp[:,:-1]) if lag==-1 else (gp,lp))
            a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
            features.append((a*b).sum(1)/np.maximum(np.sqrt((a*a).sum(1)*(b*b).sum(1)),1e-10))
    compact=np.clip(np.nan_to_num(np.column_stack(features)), -1e5,1e5)
    return np.column_stack([data['M'],data['F'],compact])

if __name__=='__main__':
    x=get_features(load())
    evaluate('exp27_tree',x,dict(family='trees',n_estimators=350,min_samples_leaf=2,max_features=.6))
    evaluate('exp27_svm',x,dict(family='svm',C=10,gamma='scale'))
