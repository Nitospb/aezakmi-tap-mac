"""Signed modal resonance and post-trigger damping, with baseline motion/lid."""
import numpy as np
import common


def get_features(data):
    h = np.asarray(data['H'], dtype=float)
    ht = np.linspace(-.65, .5, h.shape[1])
    g = np.asarray(data['G'], dtype=float)
    gt = np.linspace(-.65, .5, g.shape[1])
    baseline = h[:, (ht >= -.20) & (ht < -.03)].mean(axis=1)
    h = h - baseline[:, None, :]
    out = [np.asarray(data['M'], dtype=float)]
    eps = 1e-10
    energies = []
    spectra = []
    # Consistent frequency bands across early and late response windows.
    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 401)]
    for lo, hi in [(0, .1), (.1, .25), (.25, .4), (0, .4)]:
        x = h[:, (ht >= lo) & (ht < hi)]
        n = x.shape[1]
        fft = np.fft.rfft(x * np.hanning(n)[None, :, None], axis=1)
        f = np.fft.rfftfreq(n, d=(ht[1]-ht[0]))
        power = np.abs(fft)**2 / max(n*n, 1)
        total = power.sum(axis=1) + eps
        band = np.stack([power[:, (f >= a) & (f < b)].sum(axis=1) for a,b in bands], axis=1)
        out += [np.log1p(band).reshape(len(h), -1), (band / total[:, None, :]).reshape(len(h), -1)]
        spectra.append(band)
        prob = power / total[:, None, :]
        centroid = (prob * f[None, :, None]).sum(axis=1)
        bandwidth = np.sqrt((prob * (f[None, :, None]-centroid[:, None, :])**2).sum(axis=1))
        entropy = -(prob * np.log(prob+eps)).sum(axis=1)/np.log(max(len(f),2))
        dom = power[:, 1:].argmax(axis=1) + 1
        out += [centroid, bandwidth, entropy, f[dom]]
        rms = np.sqrt(np.mean(x*x, axis=1)+eps)
        energies.append(rms)
        out += [x.mean(axis=1), rms, x.max(axis=1), x.min(axis=1), np.mean(np.diff(np.sign(x),axis=1)!=0,axis=1)]
        cov = np.einsum('ntc,ntd->ncd', x, x)/n
        corr = cov/(rms[:, :, None]*rms[:, None, :]+eps)
        rows, cols = np.triu_indices(6, 1)
        out += [corr[:, rows, cols]]
        if lo == 0 and hi == .4:
            # Complex coefficients retain trigger-relative polarity and modal phase.
            selected = (f >= 2) & (f <= 80)
            z = fft[:, selected]/n
            scale = rms[:, None, :]+eps
            out += [(z.real/scale).reshape(len(h),-1), (z.imag/scale).reshape(len(h),-1)]
    for i,j in [(1,0),(2,0),(2,1)]:
        out.append(np.log((energies[i]+eps)/(energies[j]+eps)))
        out.append(np.log((spectra[i]+eps)/(spectra[j]+eps)).reshape(len(h),-1))
    # Envelope slope and its signed early/late impulse at fixed trigger-relative bins.
    env = []
    for lo in np.arange(0, .4, .05):
        x = h[:, (ht >= lo) & (ht < lo+.05)]
        env.append(np.sqrt(np.mean(x*x,axis=1)+eps))
        out.append(x.mean(axis=1))
    env = np.stack(env,axis=1)
    out.append(np.log1p(env).reshape(len(h),-1))
    t = np.arange(8)-3.5
    out.append((np.log(env+eps)*t[None,:,None]).sum(axis=1)/(t*t).sum())
    # Fractional lid trajectory, never rounded to integer angles.
    gb = g[:, (gt >= -.2)&(gt < -.03)].mean(axis=1)
    out.append(gb[:,None])
    for lo,hi in [(-.2,0),(0,.1),(.1,.25),(.25,.4)]:
        z = g[:, (gt >= lo)&(gt < hi)]-gb[:,None]
        out += [z.mean(axis=1)[:,None], z.std(axis=1)[:,None], (z[:,-1]-z[:,0])[:,None], z.max(axis=1)[:,None], z.min(axis=1)[:,None]]
    return np.nan_to_num(np.concatenate(out,axis=1),nan=0.,posinf=0.,neginf=0.)


if __name__ == '__main__':
    d = common.load()
    x = get_features(d)
    common.evaluate('exp08_trees',x,dict(family='trees',n_estimators=320,min_samples_leaf=2,max_features=.6))
    common.evaluate('exp08_svm',x,dict(family='svm',C=10,gamma='scale'))
