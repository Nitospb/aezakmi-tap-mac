"""Spatial multiclass LightGBM; independent class argmax per coordinate."""
import numpy as np
import lightgbm as lgb
import common


def get_features(data):
    return np.nan_to_num(np.column_stack((data['M'], data['F'])), nan=0., posinf=0., neginf=0.)


def softmax(raw):
    z = raw - raw.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def spatial_objective(strength, classes):
    def objective(raw, dataset):
        y = dataset.get_label().astype(int)
        p = softmax(raw)
        cost = ((np.arange(classes)[None, :] - y[:, None]) / (classes - 1)) ** 2
        centered = cost - (p * cost).sum(axis=1, keepdims=True)
        grad = p + strength * p * centered
        grad[np.arange(len(y)), y] -= 1.
        # Exact spatial diagonal is p_j(1-2p_j)(c_j-E[c]); it can
        # be negative. Use its absolute value plus the conventional
        # doubled multinomial CE diagonal for a safe positive approximation.
        hess = 2 * p * (1-p) + strength * np.abs(p * (1-2*p) * centered)
        hess = np.maximum(hess, 1e-6)
        weights = dataset.get_weight()
        if weights is not None:
            grad *= weights[:, None]
            hess *= weights[:, None]
        return grad, hess
    return objective


def fit_models(X, y, w, config):
    models = []
    for axis, classes in enumerate((20, 13)):
        params = dict(objective=spatial_objective(config['spatial_lambda'], classes),
                      num_class=classes, learning_rate=.04, num_leaves=7,
                      max_depth=3, min_data_in_leaf=5, lambda_l2=4.,
                      num_threads=1, verbosity=-1, seed=common.SEED,
                      deterministic=True, force_col_wise=True, metric='None')
        dataset = lgb.Dataset(X, label=y[:, axis], weight=w)
        models.append(lgb.train(params, dataset, num_boost_round=180))
    return models


def predict_probabilities(models, X):
    return [softmax(m.predict(X, raw_score=True, num_threads=1)) for m in models]


if __name__ == '__main__':
    d = common.load()
    X = get_features(d)
    train = d['idx'] < 260
    val = (d['idx'] >= 260) & (d['idx'] < 520) & (d['w'] == 1)
    for strength in (.25, 1.):
        config = dict(family='custom_spatial_gbm', spatial_lambda=strength,
                      n_estimators=180, learning_rate=.04, num_leaves=7,
                      max_depth=3, min_child_samples=5, reg_lambda=4,
                      n_jobs=1, manual=True)
        name = 'exp16_spatial_' + str(strength).replace('.', 'p')
        models = fit_models(X[train], d['y'][train], d['w'][train], config)
        p = predict_probabilities(models, X[val])
        common.save_result(name, X, config, models, p, d['y'][val], d['idx'][val],
                           extra=dict(loss='CE + lambda E[((class-label)/(K-1))^2]',
                                      hessian='2p(1-p) + lambda abs(exact spatial diagonal), floor 1e-6',
                                      decoder='per-axis class argmax', features='M+F'))
