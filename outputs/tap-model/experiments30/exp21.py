"""Independent cumulative ordinal logistic models with descending PAVA."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import common


def get_features(data):
    return np.nan_to_num(np.concatenate([data['M'], data['F']], axis=1).astype(float))


def fit_models(X, y, w, config):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    axes = []
    for axis, size in enumerate((20, 13)):
        boundaries = []
        for b in range(size - 1):
            target = (y[:, axis] > b).astype(int)
            if len(np.unique(target)) == 1:
                boundaries.append(float(target[0]))
            else:
                model = LogisticRegression(C=config.get('C', .1), max_iter=1500,
                                           solver='lbfgs', n_jobs=1,
                                           random_state=common.SEED)
                model.fit(Z, target, sample_weight=w)
                boundaries.append(model)
        axes.append(boundaries)
    return {'scaler': scaler, 'axes': axes}


def descending_pava(values):
    """Euclidean projection onto nonincreasing survival probabilities."""
    result = np.empty_like(values)
    for row_id, row in enumerate(values):
        means, counts = [], []
        for value in row:
            means.append(float(value))
            counts.append(1)
            while len(means) > 1 and means[-2] < means[-1]:
                count = counts[-2] + counts[-1]
                mean = (means[-2] * counts[-2] + means[-1] * counts[-1]) / count
                means[-2:] = [mean]
                counts[-2:] = [count]
        result[row_id] = np.repeat(means, counts)
    return result


def predict_probabilities(models, X):
    Z = models['scaler'].transform(X)
    output = []
    for boundaries in models['axes']:
        survival = np.column_stack([
            np.full(len(Z), m) if isinstance(m, float) else m.predict_proba(Z)[:, 1]
            for m in boundaries])
        survival = descending_pava(survival)
        p = -np.diff(np.column_stack([np.ones(len(Z)), survival, np.zeros(len(Z))]), axis=1)
        output.append(np.maximum(p, 0))
    return output


if __name__ == '__main__':
    data = common.load()
    X = get_features(data)
    train = data['idx'] < 260
    val = (data['idx'] >= 260) & (data['idx'] < 520) & (data['w'] == 1)
    for C in (.1, 1.):
        config = {'family': 'ordinal_logistic', 'C': C, 'features': 'M+F',
                  'decoder': 'survival_pava_then_class_argmax'}
        models = fit_models(X[train], data['y'][train], data['w'][train], config)
        p = predict_probabilities(models, X[val])
        assert all(np.isfinite(a).all() and np.allclose(a.sum(1), 1) for a in p)
        common.save_result('exp21_ordinal_C' + str(C).replace('.', 'p'), X, config,
                           models, p, data['y'][val], data['idx'][val])
