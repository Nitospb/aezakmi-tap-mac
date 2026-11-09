"""Freeze development-selected guard, then evaluate historical pass3 once."""
from pathlib import Path
import json,numpy as np,joblib
import common,exp23
ROOT=Path(__file__).resolve().parent
z=common.load();dev=z['idx']<520;test=(z['idx']>=520)&(z['w']==1)
selection=dict(name='exp23-guard80',threshold=.8,criterion='Best shared forward validation score among candidates; exact/near/far improve jointly; reverse-pass confirmation improves score and far errors without exact loss. No test-based tuning.',reverse=json.loads((ROOT/'reverse-confirmation.json').read_text()),audio_used=False)
(ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
base=joblib.load(ROOT.parent/'light-dynamics/evaluation-model.joblib');p=common.probabilities(base['motion'],z['M'][test]);q=common.probabilities(base['light'],z['F'][test]);fine=[(1-base['weight'])*a+base['weight']*b for a,b in zip(p,q)]
heads=exp23.fit_side_heads(z,dev);side=common.probabilities(heads,exp23.get_features(z)[test],classes=(2,2));guard=exp23.guard_probabilities(fine,side,.8);old=common.decode(fine);new=common.decode(guard);truth=z['y'][test]
report=dict(selected=selection['name'],threshold=.8,audio_used=False,test_baseline=common.metrics(truth,old),test_guard=common.metrics(truth,new),changed=int((old!=new).any(1).sum()),fixed_exact=int(((old!=truth).any(1)&(new==truth).all(1)).sum()),lost_exact=int(((old==truth).all(1)&(new!=truth).any(1)).sum()),validation=json.loads((ROOT/'exp23-guard80-result.json').read_text())['validation'],reverse=selection['reverse'],warning='Pass3 is a reused historical test; no claim of new-session generalization.',data_window=json.loads((ROOT/'data-audit.json').read_text()))
joblib.dump(dict(kind='side_guard',base=base,heads=heads,threshold=.8,audio_used=False),ROOT/'evaluation-model.joblib',compress=3)
(ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(ROOT/'heldout-predictions.json').write_text(json.dumps([dict(trial_index=int(i),truth=t.tolist(),previous=a.tolist(),prediction=b.tolist()) for i,t,a,b in zip(z['idx'][test],truth,old,new)],indent=2));print('EVALUATION',json.dumps(report),flush=True)
# All-data fit only after reporting evaluation.
heads=exp23.fit_side_heads(z,np.ones(len(z['idx']),bool));base=joblib.load(ROOT.parent/'light-dynamics/model.joblib');joblib.dump(dict(kind='side_guard',base=base,heads=heads,threshold=.8,audio_used=False,feature_source='existing extract/get_features then exp23.get_features and guard_probabilities'),ROOT/'model.joblib',compress=3);print('FINAL_SAVED',flush=True)
