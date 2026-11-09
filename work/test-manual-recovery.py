from pathlib import Path
import sys,importlib.util,json,numpy as np,tempfile
root=Path('outputs/tap-model').resolve();sys.path.insert(0,str(root/'live'))
from feedback_training import recover_manual,feedback_rows
spec=importlib.util.spec_from_file_location('manual_feature_source',root/'light-dynamics/train.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from train import SESSION
raw=np.memmap(SESSION/'samples.bin',dtype=d.DT,mode='r');trows=[json.loads(l) for l in (SESSION/'trials.jsonl').read_text().splitlines()];r=next(r for r in trows if r['capture_source']=='automatic');t=r['trigger_t'];a=raw[(raw['t']>=t-.9)&(raw['t']<=t+.7)];a=np.sort(a,order='t');samples=np.column_stack([a['kind'],a['t'],a['x'],a['y'],a['z']])
label=dict(event='missed_tap_ground_truth',event_id='manual-one',prediction_id='missed-one',manual_samples=samples.tolist(),search_start=t-.08,interval_end=t+.7,control_t=t+.78,tap_timestamp_known=False,trigger_t=None,col=r['col'],row=r['row'])
recovered=recover_manual(label,d);assert np.asarray(recovered['M']).shape==(652,) and np.asarray(recovered['L']).shape==(153,);assert abs(recovered['estimated_trigger_t']-t)<.15;assert recovered['tap_timestamp_known'] is False
state=Path(tempfile.mkdtemp(prefix='manual-label-test-',dir='work'));session=state/'session';session.mkdir();(session/'labels.jsonl').write_text(json.dumps(label)+'\n');rows,skipped=feedback_rows(state,root,d);assert len(rows)==1 and not skipped and rows[0]['weight']==.5
again,_=feedback_rows(state,root,d);assert len(again)==1 and len((session/'recovered-manual.jsonl').read_text().splitlines())==1
blank=samples.copy()
for k in [0,1]:
 mask=blank[:,0]==k;blank[mask,2:]=[0,0,1] if k==0 else [0,0,0]
bad=dict(label,event_id='blank',prediction_id='blank',manual_samples=blank.tolist())
try:recover_manual(bad,d)
except ValueError:pass
else:raise AssertionError('Blank motion got an invented trigger')
assert label['trigger_t'] is None
print('PASS: missed-tap raw window -> estimated timing +652/153 features; weight0.5; training admission; derived cache; blank rejected without invented ground truth. Offset ms:',(recovered['estimated_trigger_t']-t)*1000)
from train_model import hp
weak=samples.copy();strength=[]
for k,limit in [(0,.008),(1,.32)]:
 rows=weak[weak[:,0]==k,1:];times,high,_=hp(rows);sel=(times>=t-.05)&(times<=t+.15);peak=float(np.linalg.norm(high[sel],axis=1).max());strength.append(limit/max(peak,1e-8))
factor=min(.5,*strength)
for k in [0,1]:
 mask=weak[:,0]==k;pre=weak[mask & (weak[:,1]<t-.08),2:];base=np.median(pre,axis=0);weak[mask,2:]=base+(weak[mask,2:]-base)*factor
weak_label=dict(label,event_id='weak-test',prediction_id='weak-test',manual_samples=weak.tolist());wr=recover_manual(weak_label,d);assert np.asarray(wr['M']).shape==(652,)
from feedback_training import retrain
base_bundle=__import__('joblib').load(root/'light-dynamics/model.joblib');updated,result=retrain(root,state,session,base_bundle,d,0);assert result['trained'] and result['training_feedback']==1 and result['manual_estimated']==1
assert updated['motion'][0].named_steps['standardscaler'].n_samples_seen_==780
print('PASS: weaker-than-live-threshold motion recovered; actual replay retrain includes manual sample (779+1), persisted separate version.')
