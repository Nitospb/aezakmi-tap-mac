"""Predict one saved trial (JSON object), without opening any audio files.
Usage: work/model-env/bin/python outputs/tap-model/no-microphone/predict.py trial.json
"""
from pathlib import Path
import sys,json,joblib
from train import extract
root=Path(__file__).resolve().parent
bundle=joblib.load(root/'model.joblib')
r=json.loads(Path(sys.argv[1]).read_text())
features,_=extract(r);x=features[bundle['feature_group']][None,:]
col,row=[int(m.predict(x)[0]) for m in bundle['models']]
print(json.dumps(dict(col=col,row=row,zone=row*20+col,x_normalized=(col+.5)/20,y_normalized=(row+.5)/13,audio_used=False)))
