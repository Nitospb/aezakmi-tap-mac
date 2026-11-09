"""Train-fitted feature normalization; safe to load in the local inference worker."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class FeatureNormalizer(BaseEstimator,TransformerMixin):
 def __init__(self,kind='standard'):self.kind=kind
 def _shape(self,X):
  X=np.asarray(X,dtype=np.float64)
  if not np.isfinite(X).all():raise ValueError('Non-finite input features')
  return X
 def _nonlinear(self,X):
  if self.kind=='winsor':return np.clip(X,self.lower_,self.upper_)
  if self.kind=='asinh':return np.arcsinh((X-self.center_)/self.robust_scale_)
  return X
 def fit(self,X,y=None,sample_weight=None):
  X=self._shape(X);self.n_features_in_=X.shape[1];self.n_samples_seen_=len(X)
  if self.kind not in ['standard','weighted','winsor','asinh']:raise ValueError(self.kind)
  weights=np.ones(len(X)) if self.kind=='standard' or sample_weight is None else np.asarray(sample_weight)
  self.lower_=np.quantile(X,.01,axis=0);self.upper_=np.quantile(X,.99,axis=0)
  self.center_=np.median(X,axis=0);iqr=np.quantile(X,.75,axis=0)-np.quantile(X,.25,axis=0);std=X.std(0)
  self.robust_scale_=np.where(iqr>1e-12,iqr/1.349,np.where(std>1e-12,std,1))
  A=self._nonlinear(X);self.mean_=np.average(A,axis=0,weights=weights);variance=np.average((A-self.mean_)**2,axis=0,weights=weights);self.scale_=np.sqrt(variance);self.scale_=np.where(self.scale_>1e-12,self.scale_,1)
  return self
 def transform(self,X):
  X=self._shape(X)
  if X.shape[1]!=self.n_features_in_:raise ValueError('Feature count mismatch')
  return (self._nonlinear(X)-self.mean_)/self.scale_
