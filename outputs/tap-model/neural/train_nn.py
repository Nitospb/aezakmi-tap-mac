"""Small neural models for current schema4 data. No audio. Chronological pass split.
Pass3 is reused historical evaluation; model/config/epoch/fusion selected on pass2.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['TF_NUM_INTRAOP_THREADS']='2';os.environ['TF_NUM_INTEROP_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,json,importlib.util
import numpy as np
import tensorflow as tf
import joblib
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('als_features_nn',ROOT.parent/'light-dynamics/train.py');dyn=importlib.util.module_from_spec(spec);spec.loader.exec_module(dyn)
from train import extract,prepare,SESSION,metrics
SEED=20261107

def waveform(r,jitter=0):
 _,center,_=prepare(r);center+=jitter/800
 q=center-.025+np.arange(128)/800;W=[];context=[]
 for name in ['gyro_dps','accel_g']:
  a=np.asarray(r[name]);a=a[(a[:,0]>=center-.15)&(a[:,0]<=center+.40)];t=a[:,0];v=a[:,1:];before=v[t<center-.045];base=before.mean(0) if len(before) else v[0];v=v-base
  wave=np.stack([np.interp(q,t,v[:,k]) for k in range(3)],axis=1);scale=max(float(np.linalg.norm(wave,axis=1).max()),1e-8)
  W.extend([wave/scale,np.diff(wave,axis=0,prepend=wave[:1])/scale]);context.extend([np.log1p(scale),*base])
 a=np.asarray(r['lid_degrees']);v=a[:,1];t=a[:,0];pre=v[t<center-.03];base=np.median(pre) if len(pre) else v[0];query=center+np.linspace(-.12,.32,32);angle=np.interp(query,t,v-base);A=np.column_stack([angle,np.diff(angle,prepend=angle[:1])]);context.append(base/180)
 return np.concatenate(W,axis=1).astype('float32'),A.astype('float32'),np.asarray(context,dtype='float32')

def norm_fit(arrays):
 result=[]
 for a in arrays:
  axes=tuple(range(a.ndim-1));mean=a.mean(axis=axes);std=a.std(axis=axes);std=np.where(std<1e-7,1,std);result.append(dict(mean=mean.tolist(),scale=std.tolist()))
 return result

def norm_apply(arrays,norm):return [((a-np.array(n['mean']))/np.array(n['scale'])).astype('float32') for a,n in zip(arrays,norm)]
def targets(y):
 return dict(x=y[:,0],y=y[:,1],region2=(y[:,0]>=10).astype(int)+2*(y[:,1]>=6.5).astype(int),region4=np.minimum(3,((y[:,0]+.5)*4/20).astype(int))+4*np.minimum(3,((y[:,1]+.5)*4/13).astype(int)))

def build(family,shapes,seed):
 tf.keras.utils.set_random_seed(seed);reg=tf.keras.regularizers.l2(1e-3)
 inputs=[tf.keras.Input(shape=s) for s in shapes]
 if family=='mlp':
  a=tf.keras.layers.Dense(96,activation='relu',kernel_regularizer=reg)(inputs[0]);a=tf.keras.layers.Dropout(.35)(a);a=tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=reg)(a)
 else:
  a=inputs[0]
  for width,kernel in [(16,7),(24,5),(32,3)]:
   a=tf.keras.layers.Conv1D(width,kernel,strides=2,padding='same',activation='relu',kernel_regularizer=reg)(a);a=tf.keras.layers.LayerNormalization()(a)
  a=tf.keras.layers.Flatten()(a)
  b=tf.keras.layers.Conv1D(8,3,strides=2,padding='same',activation='relu')(inputs[1]);b=tf.keras.layers.Flatten()(b)
  c=tf.keras.layers.Dense(16,activation='relu',kernel_regularizer=reg)(inputs[2]);c=tf.keras.layers.Dropout(.3)(c)
  a=tf.keras.layers.Concatenate()([a,b,c,inputs[3]]);a=tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=reg)(a);a=tf.keras.layers.Dropout(.35)(a)
 out={k:tf.keras.layers.Dense(n,activation='softmax',name=k)(a) for k,n in [('x',20),('y',13),('region2',4),('region4',16)]}
 model=tf.keras.Model(inputs,out);model.compile(optimizer=tf.keras.optimizers.Adam(8e-4,clipnorm=2),loss={k:'sparse_categorical_crossentropy' for k in out},loss_weights=dict(x=1.,y=1.,region2=.25,region4=.25))
 return model

def probability(model,A):
 o=model(A,training=False);return [o[k].numpy() for k in ['x','y']]
def pred(p):return np.column_stack([a.argmax(1) for a in p])
def key(m):return m['exact'],m['within_one'],-m['mean_distance_cells']
def train(model,A,y,w,epochs,seed,val=None):
 rng=np.random.default_rng(seed);Y=targets(y);best=None;best_weights=None;best_epoch=0;stale=0
 for epoch in range(epochs):
  order=rng.permutation(len(y))
  for start in range(0,len(order),64):
   ix=order[start:start+64];model.train_on_batch([a[ix] for a in A],{k:v[ix] for k,v in Y.items()},sample_weight={k:w[ix] for k in Y})
  if val is not None:
   va,vy=val;metric=metrics(vy,pred(probability(model,va)))
   if best is None or key(metric)>key(best):best=metric;best_weights=model.get_weights();best_epoch=epoch+1;stale=0
   else:stale+=1
   if epoch%10==0:print(json.dumps(dict(epoch=epoch+1,validation=metric)),flush=True)
   if stale>=22:break
 if val is not None:model.set_weights(best_weights)
 return best,best_epoch

def get_data():
 cache=np.load(ROOT.parent/'no-microphone/features.npz');idx=cache['idx'];lookup={}
 ev=[json.loads(x) for x in (SESSION/'events.jsonl').read_text().splitlines()];bad={x['trial_id'] for x in ev if x.get('event')=='invalidate_trial'}
 for r in [json.loads(x) for x in (SESSION/'trials.jsonl').read_text().splitlines()]:
  if r['trial_id'] not in bad:lookup[r['trial_index']]=r
 light,angle=dyn.aligned_raw();W=[];G=[];C=[];L=[];jW=[];jG=[];jC=[];ids=[]
 for i in idx:
  r=lookup[int(i)];a,b,c=waveform(r);W.append(a);G.append(b);C.append(c);L.append(dyn.get_features(r,light,angle)[0])
 for j,i in enumerate(idx):
  for shift in [-2,0,2]:
   a,b,c=waveform(lookup[int(i)],shift);jW.append(a);jG.append(b);jC.append(c);ids.append(j)
 arrays=dict(W=np.array(W),G=np.array(G),C=np.array(C),L=np.array(L,dtype='float32'),M=np.column_stack([cache['motion'],L]).astype('float32'),jW=np.array(jW),jG=np.array(jG),jC=np.array(jC),aug_ids=np.array(ids),idx=idx,y=cache['y'],w=cache['w'])
 np.savez_compressed(ROOT/'neural-inputs.npz',**arrays);return arrays

def inputs(z,family,mask,augment=False):
 if family=='mlp':return [z['M'][mask]],np.flatnonzero(mask)
 if augment:
  am=mask[z['aug_ids']];ids=z['aug_ids'][am];return [z['jW'][am],z['jG'][am],z['L'][ids],z['jC'][am]],ids
 return [z[k][mask] for k in ['W','G','L','C']],np.flatnonzero(mask)

def main():
 z=get_data();idx=z['idx'];y=z['y'];w=z['w'];tr=idx<260;va=(idx>=260)&(idx<520)&(w==1);dev=idx<520;te=(idx>=520)&(w==1)
 # Fixed current tree/light baseline; no baseline tuning on neural test.
 bt=joblib.load(ROOT.parent/'no-microphone/evaluation-motion.joblib') # evaluation used only after selection
 candidates=[];stored=[]
 for family,seed in [('mlp',7),('cnn',7),('cnn',19)]:
  tf.keras.backend.clear_session();A,ids=inputs(z,family,tr,augment=family=='cnn');norm=norm_fit(A);A=norm_apply(A,norm);V,_=inputs(z,family,va);V=norm_apply(V,norm)
  model=build(family,[a.shape[1:] for a in A],seed);score,epoch=train(model,A,y[ids],w[ids],120,seed,val=(V,y[va]));name=f'{family}-{seed}';model.save(ROOT/f'validation-{name}.keras');entry=dict(family=family,seed=seed,epoch=epoch,validation=score,train_original=int(tr.sum()),train_augmented=len(ids));candidates.append(entry);stored.append((model,norm));print('CANDIDATE',json.dumps(entry),flush=True)
 best=max(range(len(candidates)),key=lambda i:key(candidates[i]['validation']));choice=candidates[best];family=choice['family'];seed=choice['seed'];epochs=choice['epoch'];print('FROZEN_SELECTION',json.dumps(choice),flush=True)
 # Refit first two passes for historical third-pass evaluation.
 A,ids=inputs(z,family,dev,augment=family=='cnn');norm=norm_fit(A);A=norm_apply(A,norm);T,_=inputs(z,family,te);T=norm_apply(T,norm)
 model=build(family,[a.shape[1:] for a in A],seed);train(model,A,y[ids],w[ids],epochs,seed);p=pred(probability(model,T));score=metrics(y[te],p)
 model.save(ROOT/'evaluation.keras');(ROOT/'evaluation-normalization.json').write_text(json.dumps(dict(family=family,norm=norm,fit_original_n=int(dev.sum()),fit_augmented_n=len(ids)),indent=2))
 base=np.column_stack([m.predict(z['M'][te,:652]) for m in bt['models']]).astype(int);baseline=metrics(y[te],base)
 report=dict(audio_used=False,normalization='Every input z-scored using training subset only; waveform/context dimensions normalized separately. Saved mean/scale reused at inference.',architecture='MLP or multi-input temporal CNN; categorical X20/Y13 plus auxiliary 2x2 and 4x4 heads. Decode X/Y argmax, no coordinate averaging.',augmentation='CNN time shift -2/0/+2 samples at 800Hz; replicas stay within the same original trial split. MLP no augmentation.',candidates=candidates,selected=choice,test=score,motion_baseline_same_test=baseline,split=dict(train=int(tr.sum()),validation=int(va.sum()),eval_fit=int(dev.sum()),test=int(te.sum())),warning='Third pass was evaluated in earlier iterations. Reused historical test, not independent confirmation. No test-based model/epoch selection.')
 (ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(ROOT/'heldout-predictions.json').write_text(json.dumps([dict(trial_index=int(i),truth=t.tolist(),prediction=q.tolist()) for i,t,q in zip(idx[te],y[te],p)],indent=2));print('EVALUATION',json.dumps(report),flush=True)
 # Final all-data refit is distinct from evaluation.
 allmask=np.ones(len(idx),bool);A,ids=inputs(z,family,allmask,augment=family=='cnn');norm=norm_fit(A);A=norm_apply(A,norm);final=build(family,[a.shape[1:] for a in A],seed);train(final,A,y[ids],w[ids],epochs,seed);final.save(ROOT/'model.keras');(ROOT/'normalization.json').write_text(json.dumps(dict(family=family,norm=norm,fit_original_n=len(idx),fit_augmented_n=len(ids),audio_used=False),indent=2));print('FINAL_SAVED',flush=True)
if __name__=='__main__':main()
