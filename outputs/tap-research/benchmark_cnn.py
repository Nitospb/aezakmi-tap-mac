"""Small research-inspired multitask temporal CNN; not a TapNet reproduction."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2'
from pathlib import Path
import json,gzip,numpy as np,tensorflow as tf
ROOT=Path(__file__).resolve().parent
rows=sorted([json.loads(l) for l in gzip.open(ROOT.parent/'tap-dataset/trials.valid.jsonl.gz','rt') if json.loads(l).get('capture_source')=='automatic'],key=lambda r:r['trial_index'])
y=np.array([[r['col'],r['row']] for r in rows]);idx=np.array([r['trial_index'] for r in rows]);tr=idx<340;va=(idx>=340)&(idx<460);te=idx>=460

def wave(r,jitter=0):
 a=np.asarray(r['accel_g']);g=np.asarray(r['gyro_dps']);trigger=r['trigger_t']
 sel=np.where((g[:,0]>=trigger-.015)&(g[:,0]<=trigger+.12))[0]
 base=g[g[:,0]<trigger-.04,1:].mean(axis=0)
 center=g[sel[np.argmax(np.linalg.norm(g[sel,1:]-base,axis=1))],0]+jitter/800
 query=center-.025+np.arange(105)/800
 signals=[];context=[r['lid_angle']/180]
 for stream in [g,a]:
  v=np.stack([np.interp(query,stream[:,0],stream[:,k]) for k in [1,2,3]],axis=1)
  d=np.diff(v,axis=0);scale=max(float(np.linalg.norm(d,axis=1).max()),1e-8)
  signals.append(d/scale);context.append(np.log1p(scale))
 return np.concatenate(signals,axis=1).astype('float32'),np.array(context,dtype='float32')
W=[];M=[]
for r in rows:
 a,b=wave(r);W.append(a);M.append(b)
W=np.array(W);M=np.array(M)
train_indices=np.where(tr)[0];rng=np.random.default_rng(20261107)
wa=[];ma=[];ids=[]
for i in train_indices:
 for jitter in [-4,-2,0,2,4]:
  a,b=wave(rows[i],jitter);wa.append(a);ma.append(b);ids.append(i)
wa=np.array(wa);ma=np.array(ma);ids=np.array(ids)

def labels(indices):
 yy=y[indices];zone=yy[:,1]*20+yy[:,0]
 coarse=(yy[:,1]>=6.5).astype(int)*2+(yy[:,0]>=10).astype(int)
 return dict(cell=zone,x=yy[:,0],y=yy[:,1],coarse=coarse)

def metric(p,t):
 d=np.abs(p-t);return dict(n=len(t),exact=float((d.max(axis=1)==0).mean()),within_one=float((d.max(axis=1)<=1).mean()),mean_distance=float(np.sqrt((d*d).sum(axis=1)).mean()))

def decode(output):
 joint=output['cell']*np.sqrt(output['y'][:,:,None]*output['x'][:,None,:]).reshape(-1,260)
 zone=joint.argmax(axis=1);return np.column_stack([zone%20,zone//20])

class Select(tf.keras.callbacks.Callback):
 def __init__(self):super().__init__();self.best=-1;self.best_weights=None;self.best_metrics=None;self.since=0
 def on_epoch_end(self,epoch,logs=None):
  o=self.model([W[va],M[va]],training=False);p=decode({k:v.numpy() for k,v in o.items()});m=metric(p,y[va]);score=5*m['exact']+m['within_one']-.02*m['mean_distance']
  if score>self.best:self.best=score;self.best_weights=self.model.get_weights();self.best_metrics=m;self.since=0
  else:self.since+=1
  if epoch%20==0:print('epoch',epoch,'validation',m,flush=True)
  if self.since>=50:self.model.stop_training=True

def build(seed):
 tf.keras.utils.set_random_seed(seed)
 signal=tf.keras.Input((104,6));context=tf.keras.Input((3,))
 a=signal
 for width,kernel in [(16,7),(32,5),(48,3)]:
  a=tf.keras.layers.Conv1D(width,kernel,padding='same',strides=2,activation='relu')(a)
  a=tf.keras.layers.LayerNormalization()(a)
 a=tf.keras.layers.Flatten()(a);a=tf.keras.layers.Concatenate()([a,context])
 a=tf.keras.layers.Dense(96,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(a);a=tf.keras.layers.Dropout(.3)(a)
 heads={name:tf.keras.layers.Dense(n,activation='softmax',name=name)(a) for name,n in [('cell',260),('x',20),('y',13),('coarse',4)]}
 model=tf.keras.Model([signal,context],heads)
 model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),loss={k:'sparse_categorical_crossentropy' for k in heads},loss_weights=dict(cell=1.,x=.75,y=.75,coarse=.25))
 return model
results=[];models=[]
for seed in [7,19]:
 model=build(seed);select=Select();model.fit([wa,ma],labels(ids),batch_size=64,epochs=200,verbose=0,callbacks=[select]);model.set_weights(select.best_weights)
 results.append(dict(seed=seed,validation=select.best_metrics,selection_score=select.best));models.append(model)
 print('candidate',results[-1],flush=True)
best=int(np.argmax([r['selection_score'] for r in results]));model=models[best]
out=model([W[te],M[te]],training=False);pred=decode({k:v.numpy() for k,v in out.items()});test=metric(pred,y[te])
model.save(ROOT/'temporal-cnn.keras')
report=dict(architecture='Small 3-layer 1D CNN; shared embedding; cell/X/Y/quadrant heads; IMU derivatives at 800Hz; lid and magnitude context; time-shift augmentation on training only',candidates=results,selected_seed=results[best]['seed'],test=test,training_original_n=int(tr.sum()),training_augmented_n=len(ids),warning='Same historical test as prior iterations, not new independent data. Inspired by TapNet, not a faithful reproduction. No augmentation copies cross validation/test boundaries.')
(ROOT/'cnn-benchmark.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
