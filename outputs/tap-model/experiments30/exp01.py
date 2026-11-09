"""Experiment 01: aligned 80 ms IMU impulse, amplitude and normalized shape."""
import numpy as np
import common


def get_features(data):
    """Pure per-event numerical extraction, requiring only the IMU window."""
    W = np.asarray(data.get('H', data['W']), dtype=np.float64)
    t = np.linspace(-.65, .5, W.shape[1])
    background = (t >= -.35) & (t <= -.08)
    search = (t >= -.015) & (t <= .12)
    event = (t >= -.02) & (t <= .16)
    offsets = np.linspace(-.025, .055, 64)
    features = []
    for window in W:
        centered = window - np.mean(window[background], axis=0)
        # Balance gyro and acceleration without dataset-wide statistics.
        group_rms = np.array([
            np.sqrt(np.mean(centered[event, :3] ** 2)),
            np.sqrt(np.mean(centered[event, 3:] ** 2)),
        ])
        scale = np.repeat(np.maximum(group_rms, 1e-9), 3)
        balanced = centered / scale
        energy = np.sum(balanced ** 2, axis=1)
        # Mild 7.5 ms smoothing prevents a single noisy sample setting alignment.
        energy = np.convolve(energy, np.ones(3) / 3, mode='same')
        candidates = np.flatnonzero(search)
        peak_time = t[candidates[np.argmax(energy[search])]]
        sample_t = peak_time + offsets
        impulse = np.column_stack([
            np.interp(sample_t, t, centered[:, axis]) for axis in range(6)
        ])
        impulse_scale = np.array([
            np.sqrt(np.mean(impulse[:, :3] ** 2)),
            np.sqrt(np.mean(impulse[:, 3:] ** 2)),
        ])
        normalized = impulse / np.repeat(np.maximum(impulse_scale, 1e-9), 3)
        rms = np.sqrt(np.mean(impulse ** 2, axis=0))
        noise = np.sqrt(np.mean(centered[background] ** 2, axis=0))
        descriptors = np.concatenate([
            [peak_time], np.log1p(group_rms), np.log1p(impulse_scale),
            np.log1p(rms), np.log1p(np.max(np.abs(impulse), axis=0)),
            np.mean(impulse, axis=0),
            np.log1p(rms / np.maximum(noise, 1e-9)),
            np.mean(normalized[:24], axis=0),
            np.mean(normalized[24:40], axis=0),
            np.mean(normalized[40:], axis=0),
        ])
        features.append(np.concatenate([normalized.ravel(), descriptors]))
    return np.nan_to_num(np.asarray(features), nan=0.0, posinf=1e6, neginf=-1e6)


if __name__ == '__main__':
    data = common.load()
    X = get_features(data)
    for C in (3.0, 15.0):
        common.evaluate(f'exp01_svm_c{C:g}', X, {'family': 'svm', 'C': C})
