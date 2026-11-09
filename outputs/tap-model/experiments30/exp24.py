"""Fixed automatic-label-only ExtraTrees ablation: motion versus motion+light."""
import numpy as np
import common

CONFIG = dict(family='trees', n_estimators=300, min_samples_leaf=2,
              max_features=.6, manual=False)
VARIANTS = ('motion', 'motion_light')

def get_features(data):
    """Pure live-compatible full feature bank; select_features fixes the variant."""
    return np.nan_to_num(np.column_stack((data['M'], data['F'])), nan=0., posinf=0., neginf=0.)

def select_features(data, variant='motion_light'):
    if variant not in VARIANTS:
        raise ValueError(variant)
    features = get_features(data)
    return features[:, :data['M'].shape[1]] if variant == 'motion' else features

if __name__ == '__main__':
    data = common.load()
    for variant in VARIANTS:
        common.evaluate('exp24_auto_' + variant, select_features(data, variant), CONFIG)
