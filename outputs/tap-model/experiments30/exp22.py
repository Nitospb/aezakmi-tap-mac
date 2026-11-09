"""Consistent multiscale heads scored jointly over the original 260 cells."""
import json
import numpy as np
import common


def get_features(data):
    return np.nan_to_num(np.column_stack((data['M'], data['F'])), nan=0., posinf=0., neginf=0.)


def coarse_labels(y, bins):
    # Use physical cell centers with screen boundaries at 0,20 and 0,13.
    xy = np.floor((np.asarray(y) + .5) / np.array([20., 13.]) * bins).astype(int)
    return xy[:, 1] * bins + xy[:, 0]


def fit_models(X, y, w, config):
    labels = np.column_stack((y, coarse_labels(y, 2), coarse_labels(y, 4)))
    return common.fit(X, labels, w, dict(family='trees', n_estimators=350,
                                      min_samples_leaf=2, max_features=.6))


def predict(models, X, config):
    px, py, quadrant, region = common.probabilities(models, X, classes=(20,13,4,16))
    cells = np.column_stack((np.tile(np.arange(20), 13), np.repeat(np.arange(13), 20)))
    alpha = config['coarse_weight']
    logp = np.log(np.maximum(px[:, cells[:,0]], 1e-12)) + np.log(np.maximum(py[:, cells[:,1]], 1e-12))
    logp += alpha * (np.log(np.maximum(quadrant[:, coarse_labels(cells,2)],1e-12)) +
                     np.log(np.maximum(region[:, coarse_labels(cells,4)],1e-12)))
    joint = np.exp(logp - logp.max(axis=1,keepdims=True))
    joint /= joint.sum(axis=1,keepdims=True)
    pred = cells[joint.argmax(axis=1)]
    return dict(px=px, py=py, quadrant=quadrant, region=region, joint=joint, pred=pred)


if __name__ == '__main__':
    d = common.load()
    X = get_features(d)
    train = d['idx'] < 260
    val = (d['idx'] >= 260) & (d['idx'] < 520) & (d['w'] == 1)
    models = fit_models(X[train], d['y'][train], d['w'][train], {})
    for alpha in (.25, .75):
        name = 'exp22_multiscale_' + str(alpha).replace('.', 'p')
        config = dict(family='custom_multiscale_trees', n_estimators=350,
                      min_samples_leaf=2, max_features=.6, coarse_weight=alpha,
                      manual=True, n_jobs=1)
        out = predict(models, X[val], config)
        report = dict(name=name, feature_count=X.shape[1], config=config,
                      validation=common.metrics(d['y'][val],out['pred']),
                      extra=dict(features='M+F', decoder='argmax over 260 joint cell scores',
                                 joint_layout='row-major: id=row*20+column',
                                 fusion='log px + log py + alpha*(log quadrant + log region)',
                                 boundaries='floor((cell+.5)/screen_extent*bins)'))
        (common.ROOT / (name+'-result.json')).write_text(json.dumps(report,indent=2))
        np.savez_compressed(common.ROOT / (name+'-validation.npz'),
                            **out, truth=d['y'][val],idx=d['idx'][val])
        print(json.dumps(report),flush=True)
