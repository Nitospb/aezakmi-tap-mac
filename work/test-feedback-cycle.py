from pathlib import Path
import json,subprocess,sys,tempfile,hashlib,numpy as np,joblib
root=Path('outputs/tap-model').resolve();state=Path(tempfile.mkdtemp(prefix='tap-feedback-test-',dir='work')).resolve();session=state/'test-session';session.mkdir()
z=np.load(root/'normalization-audit/features.npz');base=root/'light-dynamics/model.joblib';before=hashlib.sha256(base.read_bytes()).hexdigest()
with (session/'predictions.jsonl').open('w') as pp,(session/'labels.jsonl').open('w') as ll:
 for j in range(51):
  row=dict(prediction_id=f'test-{j}',multiple_onsets=False,model=dict(feature_version='motion652_light153',motion_features=z['motion'][j].tolist(),light_features=z['light'][j].tolist()))
  pp.write(json.dumps(row)+'\n')
  if j<50:ll.write(json.dumps(dict(event='mouse_ground_truth',event_id=f'label-{j}',prediction_id=f'test-{j}',col=int(z['y'][j,0]),row=int(z['y'][j,1])))+'\n')
 # Revised annotation is counted once, unlabelled trial51 never trains.
 ll.write(json.dumps(dict(event='mouse_ground_truth',event_id='label-revised',prediction_id='test-0',col=int(z['y'][0,0]),row=int(z['y'][0,1])))+'\n')
process=subprocess.Popen([sys.executable,str(root/'live/worker.py'),str(root),str(state)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
def response():
 line=process.stdout.readline();assert line,line;value=json.loads(line);assert 'error'not in value,value;return value
def command(q):process.stdin.write(json.dumps(q)+'\n');process.stdin.flush();return response()
ready=response();assert ready['ready'] and ready['version']==0
r=command(dict(op='train',session_folder=str(session)));assert r['trained'] and r['training_feedback']==50 and r['training_original']==779 and r['version']==1,r
model=joblib.load(r['path']);assert model['motion'][0].named_steps['standardscaler'].n_samples_seen_==829
same=command(dict(op='train',session_folder=str(session)));assert same['trained'] is False and same['version']==1
reset=command(dict(op='reset'));assert reset['version']==2;process.stdin.close();process.wait(timeout=20);assert process.returncode==0
restored=subprocess.run([sys.executable,str(root/'live/worker.py'),str(root),str(state)],input='',text=True,capture_output=True,check=True);r2=json.loads(restored.stdout.strip());assert r2['version']==2 and r2['ready']
assert hashlib.sha256(base.read_bytes()).hexdigest()==before
assert len(list((session/'models').glob('*.joblib')))==1
print('PASS: 50 explicit labels + 779 replay rows; dedup; unlabelled excluded; normalization829; no-op retrain; reset; restart; base model unchanged. Test files:',state)
