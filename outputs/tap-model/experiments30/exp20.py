"""Automatic-only regularized GBM on existing motion features."""
import numpy as np
import common

def get_features(data):
    return np.asarray(data['M'], dtype=float)

CONFIGS = [
    ('exp20_gbm_depth3', dict(family='gbm', manual=False, n_estimators=240, learning_rate=.035, num_leaves=7, max_depth=3, min_child_samples=5, reg_lambda=8)),
    ('exp20_gbm_depth4', dict(family='gbm', manual=False, n_estimators=240, learning_rate=.035, num_leaves=15, max_depth=4, min_child_samples=3, reg_lambda=8)),
]

if __name__ == '__main__':
    features = get_features(common.load())
    for name, config in CONFIGS:
        common.evaluate(name, features, config)
