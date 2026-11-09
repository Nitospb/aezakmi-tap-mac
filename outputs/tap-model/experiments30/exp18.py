"""Random forests using optimal thresholds, bootstrap, and sparse split features."""
import numpy as np
import common

def get_features(data):
    return np.concatenate((data['M'], data['F']), axis=1)

CONFIGS = (
    ('exp18_forest_sqrt_leaf2', dict(family='forest', n_estimators=400, min_samples_leaf=2, max_features='sqrt')),
    ('exp18_forest_fraction015_leaf3', dict(family='forest', n_estimators=400, min_samples_leaf=3, max_features=.15)),
)

if __name__ == '__main__':
    features = get_features(common.load())
    for name, config in CONFIGS:
        common.evaluate(name, features, config)
