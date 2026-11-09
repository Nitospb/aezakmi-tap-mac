"""Train spatial LightGBM regression and export portable, numeric-only decision trees."""
from pathlib import Path
import json,gzip,math,collections,pickle,time
import numpy as np
import lightgbm as lgb

ALPHA=0.038

def hp(rows):
    a=np.asarray(rows,dtype=float)
    base=a[0,1:].copy(); out=np.empty((len(a),3))
    for i,v in enumerate(a[:,1:]):
        base += ALPHA*(v-base); out[i]=v-base
    return a[:,0],out,a[:,1:]

def features(accel,gyro,trigger,lid):
    at,ah,ar=hp(accel); gt,gh,gr=hp(gyro)
    # Align every recording to its first main impulse, not a user's reaction time.
    candidates=np.where((gt>=trigger-0.015)&(gt<=trigger+0.12))[0]
    if not len(candidates): raise ValueError('no main impulse')
    energy=np.linalg.norm(gh[candidates],axis=1)/2
    energy += np.interp(gt[candidates],at,np.linalg.norm(ah,axis=1))/0.04
    center=gt[candidates[int(np.argmax(energy))]]
    query=center-0.02+np.arange(32)*0.24/31
    waves=[]
    for t,a in [(gt,gh),(at,ah)]:
        waves.append(np.stack([np.interp(query,t,a[:,k]) for k in range(3)],axis=1))
    f=[]
    scales=[]
    for wave in waves:
        scale=max(float(np.linalg.norm(wave,axis=1).max()),1e-8); scales.append(scale)
        normalized=wave/scale
        f.extend([math.log1p(scale),math.log1p(float(np.sqrt((wave*wave).sum(axis=1).mean())))])
        for k in range(3):
            a=normalized[:,k]
            f.extend([float(a.max()),float(a.min()),float(np.sqrt(np.mean(a*a))),float(np.mean(a[:8])),float(np.mean(a[8:])),float(np.mean(np.abs(a))),float(np.argmax(np.abs(a)))/31,float(np.sum(a[1:]*a[:-1]<0))/31])
        for i,j in [(0,1),(0,2),(1,2)]:
            denom=math.sqrt(float((normalized[:,i]**2).sum()*(normalized[:,j]**2).sum()))
            f.append(float((normalized[:,i]*normalized[:,j]).sum())/max(denom,1e-12))
        f.extend(normalized.ravel().tolist())
    for t,a in [(gt,gr),(at,ar)]:
        before=a[t< trigger-0.04]
        f.extend((before.mean(axis=0) if len(before) else a[0]).tolist())
    f.extend([float(lid if lid is not None else 120)/180,math.log((scales[0]+1e-8)/(scales[1]+1e-8))])
    return np.asarray(f),float(center)

def prepare(r):
    auto=r.get('capture_source','automatic')=='automatic'
    accel=r['accel_g']; gyro=r['gyro_dps']
    if auto:
        f,t=features(accel,gyro,r['trigger_t'],r.get('lid_angle')); return f,t,1.0
    # A manual label covers an interval, not a known tap. Pick the strongest
    # isolated candidate, excluding the final control interaction.
    at,ah,_=hp(accel); gt,gh,_=hp(gyro)
    end=min(at[-1],gt[-1])-0.25
    score=np.linalg.norm(gh,axis=1)/2+np.interp(gt,at,np.linalg.norm(ah,axis=1))/0.04
    valid=np.where((gt>=max(at[0],gt[0])+0.15)&(gt<=end-0.4))[0]
    if not len(valid): raise ValueError('manual interval too short')
    p=int(valid[np.argmax(score[valid])]); center=gt[p]
    a=[x for x in accel if center-0.15<=x[0]<=center+0.4]
    g=[x for x in gyro if center-0.15<=x[0]<=center+0.4]
    if len(a)<100 or len(g)<100 or score[p]<0.15: raise ValueError('no reliable manual impulse')
    f,t=features(a,g,center-0.03,r.get('lid_angle')); return f,t,0.3

def metrics(y,p):
    pred=np.column_stack([np.clip(np.floor(p[:,0]),0,19),np.clip(np.floor(p[:,1]),0,12)]).astype(int)
    truth=np.floor(y).astype(int); delta=np.abs(pred-truth)
    dist=np.sqrt(((p-y)**2).sum(axis=1))
    return {'n':len(y),'exact_cell':float((delta.max(axis=1)==0).mean()),'within_1_cell':float((delta.max(axis=1)<=1).mean()),'within_2_cells':float((delta.max(axis=1)<=2).mean()),'within_3_cells':float((delta.max(axis=1)<=3).mean()),'mean_distance_cells':float(dist.mean()),'median_distance_cells':float(np.median(dist)),'rmse_cells':float(np.sqrt(np.mean((p-y)**2)*2)),'mae_x_cells':float(np.abs(p[:,0]-y[:,0]).mean()),'mae_y_cells':float(np.abs(p[:,1]-y[:,1]).mean())}

def train_pair(X,y,w,params):
    models=[]
    for k in range(2):
        model=lgb.LGBMRegressor(objective='regression_l2',learning_rate=0.035,n_estimators=params['n_estimators'],num_leaves=params['num_leaves'],max_depth=params['max_depth'],min_child_samples=params['min_child_samples'],reg_lambda=5.0,colsample_bytree=0.85,random_state=20261107,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(X,y[:,k],sample_weight=w); models.append(model)
    return models

def predict(models,X): return np.column_stack([m.predict(X) for m in models])

def main():
    root=Path(__file__).resolve().parent
    rows=[json.loads(x) for x in gzip.open(root.parent/'tap-dataset/trials.valid.jsonl.gz','rt')]
    rows.sort(key=lambda r:r['trial_index'])
    used=[]; skipped=[]; F=[]; weights=[]; centers=[]
    for r in rows:
        try: f,t,w=prepare(r)
        except ValueError as e: skipped.append({'trial_id':r['trial_id'],'reason':str(e)}); continue
        assert np.isfinite(f).all()
        used.append(r); F.append(f); weights.append(w); centers.append(t)
    X=np.stack(F); y=np.array([[r['col']+0.5,r['row']+0.5] for r in used]); w=np.array(weights)
    idx=np.array([r['trial_index'] for r in used]); auto=w==1
    train=idx<340; validation=(idx>=340)&(idx<460); test=idx>=460
    configurations=[dict(num_leaves=7,max_depth=3,min_child_samples=12,n_estimators=250),dict(num_leaves=15,max_depth=4,min_child_samples=10,n_estimators=350),dict(num_leaves=15,max_depth=4,min_child_samples=6,n_estimators=450)]
    experiments=[]
    for include_manual in [False,True]:
        tr=train & (np.ones(len(X),dtype=bool) if include_manual else auto)
        va=validation & auto
        for params in configurations:
            models=train_pair(X[tr],y[tr],w[tr],params)
            score=metrics(y[va],predict(models,X[va])); entry=dict(params=params,include_manual=include_manual,validation=score)
            experiments.append(entry); print(json.dumps(entry),flush=True)
    best=min(experiments,key=lambda x:x['validation']['rmse_cells'])
    dev=(idx<460)&(np.ones(len(X),dtype=bool) if best['include_manual'] else auto)
    evaluation_models=train_pair(X[dev],y[dev],w[dev],best['params'])
    pred=predict(evaluation_models,X[test]); truth=y[test]
    evaluations={'heldout_all':metrics(truth,pred),'heldout_automatic':metrics(y[test&auto],predict(evaluation_models,X[test&auto])),'center_baseline_automatic':metrics(y[test&auto],np.tile(y[dev].mean(axis=0),(sum(test&auto),1)))}
    if np.any(test&~auto): evaluations['heldout_manual_estimated']=metrics(y[test&~auto],predict(evaluation_models,X[test&~auto]))
    # Holdout is never used for model/parameter selection. Deployment refits the
    # chosen configuration on all available allowed training examples afterwards.
    finalmask=np.ones(len(X),dtype=bool) if best['include_manual'] else auto
    final=train_pair(X[finalmask],y[finalmask],w[finalmask],best['params'])
    exported={'schema':1,'cols':20,'rows':13,'feature_count':X.shape[1],'feature_version':1,'loss':'squared Euclidean distance in grid-cell units (sum of x/y squared regression errors)','models':[m.booster_.dump_model()['tree_info'] for m in final]}
    (root/'model.json').write_text(json.dumps(exported,separators=(',',':')))
    np.savez_compressed(root/'feature-check.npz',X=X[:5],pred=predict(final,X[:5]))
    fixture=[]
    for r in [q for q in used if q.get('capture_source')=='automatic'][:5]:
        f,_=features(r['accel_g'],r['gyro_dps'],r['trigger_t'],r.get('lid_angle'))
        fixture.append(dict(accel=r['accel_g'],gyro=r['gyro_dps'],trigger=r['trigger_t'],lid=r.get('lid_angle'),features=f.tolist(),prediction=predict(final,f[None,:])[0].tolist()))
    (root/'parity-fixtures.json').write_text(json.dumps(fixture,separators=(',',':')))
    report={'dataset_n':len(rows),'automatic_n':int(auto.sum()),'manual_candidates_n':int((~auto).sum()),'manual_excluded':skipped,'feature_count':X.shape[1],'chosen':best,'experiments':experiments,'evaluation':evaluations,'deployment_training_n':int(finalmask.sum()),'split':'chronological trial index: <340 train, 340..459 validation, >=460 untouched test; evaluate on untouched test after choosing configuration with validation only; final deployment refit on all allowed data','limitations':['One session on one Mac; this does not measure accuracy in a new session.','Manual tap timing is estimated from motion and is less reliable.','No independent finger location measurements: labels are prompted target centers.','Regression loss favors nearby errors over distant errors; this does not guarantee exact cell localization.']}
    (root/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (root/'heldout-predictions.json').write_text(json.dumps([{'trial_index':int(r['trial_index']),'source':r.get('capture_source'),'true_col':r['col'],'true_row':r['row'],'pred_col_float':float(p[0]),'pred_row_float':float(p[1])} for r,p in zip([r for r,m in zip(used,test) if m],pred)],indent=2))
    print('FINAL',json.dumps(evaluations),flush=True)
if __name__=='__main__': main()
