"""Compact cross-sensor lag, coherence and lid coupling descriptors."""
import numpy as np
import common


def get_features(data):
    W=np.asarray(data['W'],float)
    G=np.asarray(data['G'],float)
    t=common.TIME
    baseline=W[:,t<-.15,:]
    mu=baseline.mean(axis=1,keepdims=True)
    noise=np.sqrt(np.mean((baseline-mu)**2,axis=1))+1e-7
    Z=W-mu
    out=[np.asarray(data['M'],float),np.asarray(data['F'],float)]
    def correlation(a,b):
        aa=a-a.mean(axis=1,keepdims=True)
        bb=b-b.mean(axis=1,keepdims=True)
        return np.sum(aa*bb,axis=1)/(np.sqrt(np.sum(aa*aa,axis=1)*np.sum(bb*bb,axis=1))+1e-12)
    for lo,hi in [(-.025,.065),(.065,.22),(-.025,.22)]:
        z=Z[:,(t>=lo)&(t<hi),:]
        cols=[]
        for i in range(3):
            for j in range(3,6):
                a,b=z[:,:,i],z[:,:,j]
                cov=np.mean((a-a.mean(1,keepdims=True))*(b-b.mean(1,keepdims=True)),axis=1)
                cols.extend([correlation(a,b),np.arcsinh(cov/(noise[:,i]*noise[:,j]))])
                cor=[]
                for lag in [-8,-4,-2,0,2,4,8]:
                    if lag<0:cor.append(correlation(a[:,:lag],b[:,-lag:]))
                    elif lag>0:cor.append(correlation(a[:,lag:],b[:,:-lag]))
                    else:cor.append(correlation(a,b))
                cor=np.stack(cor,axis=1)
                best=np.argmax(np.abs(cor),axis=1)
                cols.extend([cor[np.arange(len(z)),best],np.array([-8,-4,-2,0,2,4,8])[best]/8,cor[:,0],cor[:,-1]])
        # Relative total sensor energy captures coherent motion amplitude.
        eg=np.mean(z[:,:,:3]**2,axis=(1,2));ea=np.mean(z[:,:,3:]**2,axis=(1,2))
        cols.append(np.log((eg+1e-10)/(ea+1e-10)))
        dg=np.diff(G[:,(t>=lo)&(t<hi)],axis=1)
        for i in range(6):
            cols.extend([correlation(z[:,1:,i],dg),correlation(np.diff(z[:,:,i],axis=1),dg)])
        out.append(np.column_stack(cols))
    return np.nan_to_num(np.concatenate(out,axis=1),nan=0,posinf=0,neginf=0)

if __name__=='__main__':
    d=common.load();X=get_features(d)
    common.evaluate('exp10_tree',X,dict(family='trees',n_estimators=400,min_samples_leaf=2,max_features=.7))
    common.evaluate('exp10_svm',X,dict(family='svm',C=10,gamma='scale'))
