"""Train-only PCA decorrelation of motion, lid and dynamic light features."""
import numpy as np
import common


def get_features(data):
    return np.concatenate([data['M'], data['F']], axis=1)


CONFIGS = [
    ('exp14_pca48_whiten_svm', dict(family='svm', C=10, gamma='scale', pca=48, whiten=True)),
    ('exp14_pca96_svm', dict(family='svm', C=10, gamma='scale', pca=96, whiten=False)),
]

if __name__ == '__main__':
    X = get_features(common.load())
    for name, config in CONFIGS:
        common.evaluate(name, X, config)
