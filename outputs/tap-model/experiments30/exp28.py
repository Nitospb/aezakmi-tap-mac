"""Fixed confidence gates for baseline decisions; no coordinate model changes."""
from pathlib import Path
import json
import numpy as np
from common import metrics
ROOT=Path(__file__).resolve().parent
RULES={'min_axis_peak_010':('peak',0.10),'min_axis_margin_002':('margin',0.02)}
def confidence(px,py,kind='peak'):
    """Return minimum of the two axis peaks or top1-minus-top2 margins."""
    arrays=[np.asarray(px,float),np.asarray(py,float)]
    if any(q.ndim!=2 or q.shape[1]<2 or not np.isfinite(q).all() for q in arrays):
        raise ValueError('Expected finite NxC axis probability arrays, C >= 2')
    if arrays[0].shape[0]!=arrays[1].shape[0]:raise ValueError('Axis lengths differ')
    if kind=='peak':return np.minimum(*[q.max(axis=1) for q in arrays])
    if kind=='margin':
        top=[np.partition(q,-2,axis=1)[:,-2:] for q in arrays]
        return np.minimum(*[q[:,1]-q[:,0] for q in top])
    raise ValueError('Unknown confidence kind')
def accept(px,py,rule='min_axis_peak_010'):
    kind,threshold=RULES[rule]
    return confidence(px,py,kind)>=threshold

def main():
    d=dict(np.load(ROOT/'baseline-current-validation.npz'))
    t=d['truth'];px=d['px'];py=d['py'];idx=d['idx']
    assert ((idx>=260)&(idx<520)).all()
    p=np.column_stack([px.argmax(1),py.argmax(1)])
    delta=np.abs(t-p);far=(delta[:,0]>=10)|(delta[:,1]>=7);wrong=delta.max(1)>0
    report={'experiment':'exp28','purpose':'Decision support only; rejection is abstention, not corrected classification.', 'baseline_all':metrics(t,p),'rules':{}}
    saved={'idx':idx,'truth':t,'px':px,'py':py,'prediction':p}
    for name,(kind,threshold) in RULES.items():
        c=confidence(px,py,kind);a=accept(px,py,name);saved[name]=a
        bins=[]
        # Fixed equal-width descriptive bins, not additional candidate gates.
        for lo,hi in zip([0,.05,.1,.2,.4],[.05,.1,.2,.4,1.000001]):
            b=(c>=lo)&(c<hi)
            bins.append({'low':lo,'high':min(hi,1),'n':int(b.sum()),'far_rate':float(far[b].mean()) if b.any() else None})
        report['rules'][name]={'kind':kind,'threshold':threshold,'coverage':float(a.mean()),'accepted':metrics(t[a],p[a]) if a.any() else None,'rejected_n':int((~a).sum()),'accepted_errors_count':int((a&wrong).sum()),'accepted_errors_per_all_examples':float((a&wrong).mean()),'accepted_far_count':int((a&far).sum()),'accepted_far_per_all_examples':float((a&far).mean()),'fraction_all_errors_retained':float((a&wrong).sum()/wrong.sum()) if wrong.any() else None,'fraction_far_errors_retained':float((a&far).sum()/far.sum()) if far.any() else None,'rejected_far_count':int((~a&far).sum()),'confidence_bins':bins}
    report['caveats']=['Exactly two predeclared gates; no model training or test evaluation.','Accepted-subset accuracy is conditional on coverage, not an improvement in all-example accuracy.','Confidence is uncalibrated. Fixed bins describe validation only; no deployment guarantees.','All-example baseline predictions and error rates remain unchanged; rejected cases require another action.']
    (ROOT/'exp28-summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/'exp28-validation.npz',**saved)
    print(json.dumps(report))
if __name__=='__main__':main()
