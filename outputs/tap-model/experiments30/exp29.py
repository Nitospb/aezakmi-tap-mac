"""Consensus-only same-half template blend; fixed alpha 0.1/0.25."""
import json
import numpy as np
from common import ROOT,decode,metrics

def blend_probabilities(guarded, template, alpha=0.1):
    """Blend each axis only on MAP-half agreement; cuts X10, Y7.

    Template is masked to the guarded MAP's whole half, then normalized.
    Guarded mass outside that half is retained and downweighted on agreement.
    """
    result=[]
    for base,tmp,cut in zip(guarded,template,(10,7)):
        base=np.asarray(base,float);tmp=np.asarray(tmp,float)
        side=base.argmax(1)>=cut
        agree=side==(tmp.argmax(1)>=cut)
        mask=(np.arange(base.shape[1])[None,:]>=cut)==side[:,None]
        masked=np.where(mask,tmp,0.)
        total=masked.sum(1,keepdims=True)
        masked=np.divide(masked,total,out=np.zeros_like(masked),where=total>0)
        use=agree & (total[:,0]>0)
        out=base.copy()
        out[use]=(1-alpha)*base[use]+alpha*masked[use]
        result.append(out)
    return result

if __name__=='__main__':
    b=np.load(ROOT/'exp23-guard80-validation.npz')
    t=np.load(ROOT/'exp15_template_k1-validation.npz')
    assert np.array_equal(b['idx'],t['idx'])
    assert np.array_equal(b['truth'],t['truth'])
    baseline=decode((b['px'],b['py']))
    for alpha in (.1,.25):
        name=f'exp29-consensus{int(alpha*100)}'
        p=blend_probabilities((b['px'],b['py']),(t['px'],t['py']),alpha)
        pred=decode(p)
        assert np.array_equal(pred>=np.array([10,7]),baseline>=np.array([10,7]))
        report=dict(name=name,config=dict(alpha=alpha,cuts=[10,7],baseline='exp23-guard80',template='exp15_template_k1'),validation=metrics(b['truth'],pred),changed_predictions=int(np.any(pred!=baseline,axis=1).sum()))
        np.savez_compressed(ROOT/f'{name}-validation.npz',px=p[0],py=p[1],pred=pred,truth=b['truth'],idx=b['idx'])
        (ROOT/f'{name}-result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
