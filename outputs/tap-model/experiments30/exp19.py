"""Automatic-only shrinkage LDA on motion/lid features."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import common


def get_features(data):
    return np.nan_to_num(np.asarray(data['M'], dtype=float), nan=0., posinf=0., neginf=0.)


def fit_models(X, y, w=None, config=None):
    config = config or {'shrinkage': 'auto'}
    if w is not None:
        automatic = np.asarray(w) == 1
        X, y = X[automatic], y[automatic]
    models = []
    for axis in range(2):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage=config.get('shrinkage', 'auto')))
        model.fit(X, y[:, axis])
        models.append(model)
    return models


def predict_probabilities(models, X):
    return common.probabilities(models, X)


if __name__ == '__main__':
    data = common.load()
    X = get_features(data)
    train = (data['idx'] < 260) & (data['w'] == 1)
    val = (data['idx'] >= 260) & (data['idx'] < 520) & (data['w'] == 1)
    for name, shrinkage in [('exp19_lda_auto', 'auto'), ('exp19_lda_08', .8)]:
        config = {'family': 'lda', 'solver': 'lsqr', 'shrinkage': shrinkage, 'manual': False}
        models = fit_models(X[train], data['y'][train], config=config)
        p = predict_probabilities(models, X[val])
        common.save_result(name, X, config, models, p, data['y'][val], data['idx'][val], extra={'training_n': int(train.sum()), 'features': 'M only', 'training': 'automatic only'})
