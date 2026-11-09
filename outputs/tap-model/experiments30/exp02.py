"""Joint-peak aligned 300 ms inertial waveform; no audio inputs."""
import numpy as np


def get_features(data):
    W = np.asarray(data['W'], dtype=np.float64)
    times = np.linspace(-.65, .5, W.shape[1])
    offsets = np.linspace(-.075, .225, 61)
    feats = []
    for wave in W:
        baseline = np.median(wave[(times >= -.55) & (times <= -.25)], axis=0)
        wave = wave - baseline
        # Joint inertial energy: modality normalization prevents gyro units
        # from suppressing the acceleration contribution to peak alignment.
        norms = np.column_stack([np.linalg.norm(wave[:, :3], axis=1),
                                 np.linalg.norm(wave[:, 3:], axis=1)])
        search = (times >= -.13) & (times <= .13)
        scales = np.maximum(np.max(norms[search], axis=0), 1e-8)
        energy = np.sum((norms / scales) ** 2, axis=1)
        energy = np.convolve(energy, np.ones(5) / 5., mode='same')
        candidates = np.flatnonzero(search)
        peak = candidates[np.argmax(energy[search])]
        aligned = np.column_stack([np.interp(times[peak] + offsets, times, wave[:, j])
                                   for j in range(6)])
        blocks = []
        amps = []
        # Normalize each modality in each phase, retain its amplitude and
        # late/early ratio so ringdown decay is not erased by normalization.
        for sl in [slice(0, 21), slice(21, 61)]:
            part = aligned[sl]
            for start in [0, 3]:
                v = part[:, start:start + 3]
                amplitude = max(np.sqrt(np.mean(v ** 2)), 1e-8)
                blocks.append((v / amplitude).ravel())
                amps.append(amplitude)
        amps = np.asarray(amps)
        blocks.append(np.log1p(amps))
        blocks.append(np.log((amps[2:] + 1e-8) / (amps[:2] + 1e-8)))
        blocks.append(np.concatenate([np.mean(aligned, axis=0),
                                      np.std(aligned, axis=0),
                                      np.max(aligned, axis=0),
                                      np.min(aligned, axis=0)]))
        feats.append(np.concatenate(blocks))
    return np.nan_to_num(np.asarray(feats), nan=0., posinf=0., neginf=0.)


if __name__ == '__main__':
    import common
    data = common.load()
    X = get_features(data)
    print('feature matrix', X.shape, flush=True)
    common.evaluate('exp02-svm', X, {'family': 'svm', 'C': 16.0})
    common.evaluate('exp02-trees', X, {'family': 'trees'})
