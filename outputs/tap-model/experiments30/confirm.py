"""Development-only reverse-pass check. Test pass3 never opened here."""
from pathlib import Path
import importlib.util,json,numpy as np,joblib
import common,exp23,exp15,exp29
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('confirm_baseline',ROOT.parent/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
z=common.load();train=(z['idx']>=260)&(z['idx']<520);val=(z['idx']<260)&(z['w']==1)
base=dict(motion=d.fit(z['M'][train],z['y'][train],z['w'][train],dict(family='trees')),light=d.fit(z['F'][train],z['y'][train],z['w'][train],dict(family='svm')),weight=.05)
p=d.mix(d.probs(base['motion'],z['M'][val]),d.probs(base['light'],z['F'][val]),.05)
side=exp23.fit_side_heads(z,train);sp=common.probabilities(side,exp23.get_features(z)[val],classes=(2,2));gp=exp23.guard_probabilities(p,sp,.8)
X=exp15.get_features(z);tm=exp15.fit_templates(X[train],z['y'][train],z['w'][train],dict(k=1));tp=exp15.template_probabilities(tm,X[val]);cp=exp29.blend_probabilities(gp,tp,.1)
report={}
for name,prob in [('baseline-current',p),('exp23-guard80',gp),('exp29-consensus10',cp)]:
 score=common.metrics(z['y'][val],common.decode(prob));forward=json.loads((ROOT/f'{name}-result.json').read_text())['validation'];report[name]=dict(forward=forward,reverse=score,development_mean_score=(forward['score']+score['score'])/2)
(ROOT/'reverse-confirmation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
