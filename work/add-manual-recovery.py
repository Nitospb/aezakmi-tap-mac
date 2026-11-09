from pathlib import Path
p=Path('outputs/tap-model/live/feedback_training.py');s=p.read_text()
insert='''
def recover_manual(label,dynamics):
 from train import extract
 from train_model import hp
 samples=np.asarray(label.get('manual_samples',[]),dtype=float)
 if samples.ndim!=2 or samples.shape[1]!=5:raise ValueError('No frozen sensor window for missed tap')
 def stream(k):
  a=samples[samples[:,0]==k,1:];return a[np.argsort(a[:,0])]
 a=stream(0);g=stream(1);lid=stream(2);lux=stream(3);raw=stream(4);fourth=stream(5)
 if len(a)<100 or len(g)<100 or len(lid)<2:raise ValueError('Not enough motion/context samples')
 at,ah,_=hp(a);gt,gh,_=hp(g)
 energy=np.linalg.norm(gh,axis=1)/2+np.interp(gt,at,np.linalg.norm(ah,axis=1))/.04
 end=min(at[-1],gt[-1],float(label['interval_end']))
 start=max(float(label.get('search_start',gt[0]+.15)),gt[0]+.15,at[0]+.15)
 valid=np.flatnonzero((gt>=start)&(gt<=end-.40))
 if not len(valid):raise ValueError('Too little clean data after missed tap; wait for response before pressing4')
 peak=int(valid[np.argmax(energy[valid])]);peak_t=float(gt[peak]);peak_value=float(energy[peak])
 before=energy[(gt>=peak_t-.35)&(gt<peak_t-.08)]
 median=float(np.median(before)) if len(before) else 0.;mad=float(1.4826*np.median(np.abs(before-median))) if len(before) else 0.
 threshold=max(.08,median+4*mad)
 if peak_value<threshold:raise ValueError('No clear impulse above local noise; raw window retained')
 others=valid[np.abs(gt[valid]-peak_t)>.20]
 if len(others) and float(energy[others].max())>max(.8*peak_value,threshold):raise ValueError('Several comparable impulses; raw window retained')
 half=median+.5*(peak_value-median);lo=peak;hi=peak
 while lo>0 and energy[lo-1]>half:lo-=1
 while hi+1<len(gt) and energy[hi+1]>half:hi+=1
 if gt[hi]-gt[lo]>.30:raise ValueError('Sustained movement, not an isolated tap; raw window retained')
 trigger=peak_t-.03
 def win(v):return v[(v[:,0]>=trigger-.15)&(v[:,0]<=trigger+.4)]
 ch4={r[0]:r[1] for r in fourth};ch012={r[0]:r[1:4] for r in raw}
 light=np.asarray([[r[0],r[1],*ch012[r[0]],ch4[r[0]]] for r in lux if r[0] in ch012 and r[0] in ch4])
 angle=lid[:,:2]
 # Automatic here selects the established feature-alignment code only; the saved
 # annotation remains manually timed with tap_timestamp_known=false.
 r=dict(capture_source='automatic',trigger_t=trigger,lid_angle=float(np.interp(trigger,lid[:,0],lid[:,1])),accel_g=win(a).tolist(),gyro_dps=win(g).tolist(),lid_degrees=win(angle).tolist(),light_lux=win(light[:,:2]).tolist(),light_raw_channels=np.column_stack([light[:,0],light[:,2:]])[(light[:,0]>=trigger-.15)&(light[:,0]<=trigger+.4)].tolist())
 M=extract(r)[0]['motion'];L=dynamics.get_features(r,light,angle)[0]
 return dict(M=M.tolist(),L=L.tolist(),estimated_trigger_t=trigger,estimated_peak_t=peak_t,tap_timestamp_known=False,source='manual_motion_recovery_v1',weight=.5,peak_relative_energy=peak_value,local_noise_threshold=threshold)

'''
s=s.replace('def feedback_rows(state_root):', insert+'def feedback_rows(state_root,root=None,dynamics=None):')
s=s.replace('predictions={};labels={};skipped=[]','predictions={};labels={};manuals={};skipped=[]')
s=s.replace("elif label.get('event')=='missed_tap_ground_truth':skipped.append(dict(id=label.get('event_id'),reason='Manual missed tap has no exact trigger; raw interval retained for later processing'))", "elif label.get('event')=='missed_tap_ground_truth':manuals[label.get('prediction_id',label['event_id'])]=(label,folder)")
s=s.replace(' return rows,skipped\n\ndef retrain', ''' for pid,(label,folder) in manuals.items():
  try:
   if dynamics is None:raise ValueError('Manual recovery requires feature model')
   revision=label['event_id'];cachepath=folder/'recovered-manual.jsonl'
   cached={r['event_id']:r for r in records(cachepath)}
   if revision in cached:derived=cached[revision]['derived']
   else:
    derived=recover_manual(label,dynamics)
    with cachepath.open('a') as f:f.write(json.dumps(dict(event_id=revision,prediction_id=pid,derived=derived))+'\\n');f.flush();os.fsync(f.fileno())
   M=np.asarray(derived['M'],float);L=np.asarray(derived['L'],float)
   if M.shape!=(652,) or L.shape!=(153,) or not np.isfinite(np.r_[M,L]).all():raise ValueError('Bad recovered feature vectors')
   y=[int(label['col']),int(label['row'])]
   if not(0<=y[0]<20 and 0<=y[1]<13):raise ValueError('Invalid manual target')
   rows.append(dict(id=pid,revision=revision,M=M,L=L,y=y,weight=.5,estimated_trigger_t=derived['estimated_trigger_t']))
  except (KeyError,TypeError,ValueError) as e:skipped.append(dict(id=pid,reason=str(e)))
 return rows,skipped

def retrain''')
s=s.replace('rows,skipped=feedback_rows(state_root);', 'rows,skipped=feedback_rows(state_root,root,dynamics);')
s=s.replace("if not new:return bundle,dict(trained=False,reason='Нет новых пригодных отметок мышью'", "if not new:return bundle,dict(trained=False,reason=('Не удалось однозначно выделить пропущенный импульс; записи сохранены' if skipped else 'Нет новых пригодных отметок мышью')")
s=s.replace("w=np.r_[z['w'],np.full(len(rows),2.)]", "w=np.r_[z['w'],[r.get('weight',2.) for r in rows]]")
s=s.replace('feedback_weight=2.,feature_version', "feedback_weight=dict(detected_mouse=2.,manual_estimated=.5),feature_version")
s=s.replace("new_or_revised=new,skipped=skipped,elapsed_seconds", "new_or_revised=new,manual_estimated=sum('estimated_trigger_t'in r for r in rows),skipped=skipped,elapsed_seconds")
p.write_text(s)
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text().replace('pendingID=nil;manualControl=nil;resumeAfterTrain=false', 'pendingID=nil;if manualID==nil{manualControl=nil};resumeAfterTrain=false');p.write_text(s)
p=Path('outputs/tap-model/live/TrainingControls.swift');s=p.read_text().replace('self.flash("v\\(self.model!.version) · использовано \\(count) отметок · проверяй на новых тапах")', 'let skipped=(info["skipped"] as? [[String:Any]])?.count ?? 0;self.flash("v\\(self.model!.version) · обучено \\(count) отметок · неясных \\(skipped) · проверяй новые тапы")');p.write_text(s)
