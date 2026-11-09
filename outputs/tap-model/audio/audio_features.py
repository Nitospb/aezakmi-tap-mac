"""Audio tap features anchored to the IMU clock; no speech recognition.
Noise estimated from pre-tap audio; no raw speech/music embeddings.
"""
from pathlib import Path
import json,numpy as np
from scipy.io import wavfile
from scipy import signal

class AudioArchive:
 def __init__(self,wav,index):
  self.fs,self.samples=wavfile.read(wav,mmap=True)
  if self.samples.ndim!=1:raise ValueError('Expected the recorded mono input')
  rows=[json.loads(l) for l in Path(index).read_text().splitlines()]
  self.starts=np.array([r['t'] for r in rows]);self.frames=np.array([r['frames'] for r in rows]);self.offset=np.array([r['file_frame_start'] for r in rows]);self.ends=self.starts+self.frames/self.fs
  assert all(r['sample_rate']==self.fs and r['channels']==1 for r in rows)
  assert len(self.samples)==int(self.offset[-1]+self.frames[-1]);assert np.all(np.diff(self.starts)>0)
 def window(self,lo,hi):
  q=lo+np.arange(int(round((hi-lo)*self.fs)))/self.fs;x=np.zeros(len(q),dtype=np.float32);ok=np.zeros(len(q),bool)
  a=max(0,int(np.searchsorted(self.starts,lo,side='right'))-1);b=int(np.searchsorted(self.starts,hi,side='right'))
  for j in range(a,b):
   use=(q>=self.starts[j])&(q<self.ends[j]);n=(q[use]-self.starts[j])*self.fs
   z=self.samples[self.offset[j]:self.offset[j]+self.frames[j]];x[use]=np.interp(n,np.arange(len(z)),z);ok[use]=True
  missing=np.flatnonzero(~ok)
  if len(missing):
   # Only tiny host-timestamp rounding seams, never recording pauses.
   longest=max(map(len,np.split(missing,np.where(np.diff(missing)>1)[0]+1)))
   if missing[0]==0 or missing[-1]==len(q)-1 or longest>10:raise ValueError('audio time gap or missing pre-tap coverage')
   x[missing]=np.interp(missing,np.flatnonzero(ok),x[ok])
  if not np.isfinite(x).all():raise ValueError('nonfinite PCM')
  return x,len(missing)

def filterbank(fs,nfft,n=32):
 f=np.fft.rfftfreq(nfft,1/fs);mel=lambda x:2595*np.log10(1+x/700);inv=lambda x:700*(10**(x/2595)-1)
 edges=inv(np.linspace(mel(80),mel(min(16000,fs*.46)),n+2));bank=[]
 for a,b,c in zip(edges[:-2],edges[1:-1],edges[2:]):
  row=np.maximum(0,np.minimum((f-a)/(b-a),(c-f)/(c-b)));bank.append(row/max(row.sum(),1e-12))
 return f,np.array(bank)

def features(x,fs):
 # Input is trigger-0.38..trigger+0.16. No collector green-feedback sound.
 f,t,Z=signal.stft(x,fs=fs,window='hann',nperseg=1024,noverlap=896,boundary=None,padded=False);t=t-.38;P=np.abs(Z)**2
 before=(t>=-.34)&(t<=-.10);search=(t>=-.035)&(t<=.10)
 if before.sum()<8 or search.sum()<3:raise ValueError('audio window too short')
 noise=np.median(P[:,before],axis=1);floor=max(float(noise.mean())*.02,1e-16)
 excess=np.maximum(P-noise[:,None],0);whiten=np.log1p(excess/(noise[:,None]+floor));band=(f>=300)&(f<=14000)
 score=whiten[band].mean(0);prior=np.exp(-.5*((t-.012)/.045)**2);ii=np.flatnonzero(search);peak=int(ii[np.argmax(score[ii]*prior[ii])]);center=t[peak];rt=t-center
 _,B=filterbank(fs,1024);bp=B@P;bn=B@noise;be=np.maximum(bp-bn[:,None],0);bs=np.log1p(be/(bn[:,None]+floor))
 result=[]
 for lo,hi in [(-.010,.012),(.012,.035),(.035,.080)]:
  mask=(rt>=lo)&(rt<hi);result.extend(bs[:,mask].mean(1) if mask.any() else bs[:,peak])
 active=(rt>=-.005)&(rt<=.065);spectrum=be[:,active].mean(1);shape=np.log(spectrum+max(float(spectrum.mean())*.01,1e-16));result.extend(shape-shape.mean())
 env_peaks=[];decays=[]
 for lo,hi in [(80,300),(300,900),(900,2200),(2200,5000),(5000,9000),(9000,16000)]:
  mask=(f>=lo)&(f<hi);env=excess[mask].sum(0);bg=max(float(noise[mask].sum()),floor);values=np.log1p(env/bg);pk=max(float(values[active].max()),1e-6)
  result.extend(np.interp([-.015,0,.008,.016,.030,.045,.060,.080,.100],rt,values)/pk);env_peaks.append(pk)
  early=values[(rt>=0)&(rt<.025)].mean();late=values[(rt>=.045)&(rt<.080)].mean();decays.append(float(late/(early+1e-6)))
 result.extend(env_peaks);result.extend(decays)
 energy=excess[:,active].mean(1);total=max(float(energy.sum()),1e-16);centroid=float((energy*f).sum()/total);roll=float(f[min(len(f)-1,np.searchsorted(np.cumsum(energy),.85*total))]);flat=float(np.exp(np.log(energy+1e-16).mean())/(energy.mean()+1e-16))
 pre_power=float(P[band][:,before].sum(0).mean());post_power=float(P[band][:,active].sum(0).mean());snr_db=float(10*np.log10((post_power+1e-16)/(pre_power+1e-16)))
 baseline_score=float(np.median(score[before]));prominence=float(score[peak]/max(baseline_score,1e-6))
 clip_fraction=float((np.abs(x)>=.999).mean());quality=float(np.clip((snr_db-1)/14,0,1)*np.clip((prominence-1)/4,0,1)*(1 if clip_fraction<.01 else .25))
 result.extend([centroid/fs,roll/fs,flat,snr_db/20,np.log1p(prominence),center,float(score[peak]/max(float(score[active].mean()),1e-6)),clip_fraction])
 # Background-only control, not included in normal tap input.
 control=np.log(bn+floor);control-=control.mean()
 v=np.asarray(result,float)
 if not np.isfinite(v).all():raise ValueError('nonfinite audio features')
 return v,control,dict(snr_db=snr_db,quality=quality,alignment_offset_s=float(center),clipping_fraction=clip_fraction,background_rms=float(np.sqrt(np.mean(x[:int(.28*fs)]**2))))
