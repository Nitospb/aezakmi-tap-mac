from pathlib import Path
import json,importlib.util,numpy as np
from common import load,metrics,save_result
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_baseline',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
z=load();tr=z['idx']<260;va=(z['idx']>=260)&(z['idx']<520)&(z['w']==1)
a=d.fit(z['M'][tr],z['y'][tr],z['w'][tr],dict(family='trees'));b=d.fit(z['F'][tr],z['y'][tr],z['w'][tr],dict(family='svm'));p=d.mix(d.probs(a,z['M'][va]),d.probs(b,z['F'][va]),.05)
report=save_result('baseline-current',z['M'],dict(family='existing_motion_light',light_weight=.05),None,p,z['y'][va],z['idx'][va]);(ROOT/'baseline.json').write_text(json.dumps(report,indent=2))
