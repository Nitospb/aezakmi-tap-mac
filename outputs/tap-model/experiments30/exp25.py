"""Edge weighted ExtraTrees on existing motion/lid M features only."""
import numpy as np
import common

def get_features(data):
    return np.asarray(data['M'], dtype=float)

if __name__ == '__main__':
    data = common.load()
    features = get_features(data)
    for weight in (0.5, 1.5):
        common.evaluate(f'exp25_edge{str(weight).replace(".", "p")}', features,
                        dict(family='trees', n_estimators=300, min_samples_leaf=2,
                             max_features=0.6, edge_weight=weight))
