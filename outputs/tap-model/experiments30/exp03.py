"""Experiment 03: adaptive impulse duration and time-warped IMU signatures.

Only observed gyro, acceleration, fractional angle and optical features are used.
The feature transform has no fitted dataset-wide state.
"""
import numpy as np
import common


def get_features(data):
    W = np.asarray(data['W'], dtype=np.float64)
    G = np.asarray(data['G'], dtype=np.float64)
    t = np.linspace(-.65, .5, W.shape[1])
    dt = t[1] - t[0]
    baseline = (t >= -.35) & (t <= -.08)
    search = (t >= -.015) & (t <= .12)
    event = (t >= -.02) & (t <= .20)
    result = []
    for k, window in enumerate(W):
        centered = window - window[baseline].mean(axis=0)
        group_scale = np.array([np.sqrt(np.mean(centered[event, :3]**2)),
                                np.sqrt(np.mean(centered[event, 3:]**2))])
        scale = np.repeat(np.maximum(group_scale, 1e-9), 3)
        balanced = centered / scale
        energy = np.sum(balanced**2, axis=1)
        energy = np.convolve(energy, np.ones(5)/5, mode='same')
        candidates = np.flatnonzero(search)
        p = candidates[np.argmax(energy[candidates])]
        lo = min(len(t)-1, p + round(.08/dt))
        hi = min(len(t)-1, p + round(.4/dt))
        post = energy[p:hi+1]
        c90 = p + np.searchsorted(np.cumsum(post), .9 * np.sum(post))
        # Require a continuous 20 ms quiet run before accepting a decay end.
        quiet = energy <= max(.1 * energy[p], 1e-12)
        run = max(1, round(.02/dt))
        decay = hi
        for end in range(lo, hi-run+2):
            if np.all(quiet[end:end+run]):
                decay = end
                break
        ends = [decay, int(np.clip(c90, lo, hi))]
        noise = np.sqrt(np.mean(centered[baseline]**2, axis=0))
        angle = G[k] - G[k, baseline].mean()
        row = [np.array([t[p], *group_scale, t[decay]-t[p], t[ends[1]]-t[p],
                         np.sum(post)*dt, energy[p],
                         np.sum(energy[lo:hi+1])/max(np.sum(post), 1e-12)])]
        for end in ends:
            start = max(0, p-round(.025/dt))
            sample_t = np.linspace(t[start], t[end], 64)
            warped = np.column_stack([np.interp(sample_t,t,centered[:,j]) for j in range(6)])
            rms = np.sqrt(np.mean(warped**2,axis=0))
            balanced_warp = warped/scale
            shape = warped/np.maximum(rms,1e-9)
            row.extend([balanced_warp.ravel(), shape.ravel(),
                        np.interp(sample_t,t,angle),
                        np.concatenate([rms, np.max(np.abs(warped),axis=0),
                                        warped.mean(axis=0),
                                        np.trapz(warped,sample_t,axis=0),
                                        np.log1p(rms/np.maximum(noise,1e-9))])])
        result.append(np.concatenate(row))
    adaptive = np.nan_to_num(np.asarray(result),nan=0.,posinf=1e6,neginf=-1e6)
    # Preserve useful non-warped amplitudes and optical baselines.
    return np.column_stack([adaptive,np.asarray(data['M']),np.asarray(data['F'])])


if __name__ == '__main__':
    data = common.load()
    X = get_features(data)
    for C in (3., 15.):
        common.evaluate(f'exp03_svm_c{C:g}',X,{'family':'svm','C':C})
