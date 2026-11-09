"""Cost-sensitive discrete decoding; exactly two predefined risk strengths.

No fitting and no label-derived decoder parameters. Use decode_probabilities for
live/refitted baseline probabilities; its inputs are dense class probabilities.
"""
from pathlib import Path
import json
import numpy as np
from common import ROOT, metrics

STRENGTHS = (1.0, 3.0)

def decode_probabilities(probabilities, risk_strength=1.0):
    """Pick plausible top-three class minimizing MAP loss + expected capped risk.

    Each axis is decoded independently. Candidates must have probability at least
    20% of the maximum and rank in the top three. Capped normalized absolute
    distance limits outlier influence. Never averages predicted coordinates.
    """
    result = []
    for axis, raw in enumerate(probabilities):
        q = np.asarray(raw, dtype=float)
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-15)
        half_screen = (10.0, 7.0)[axis]
        classes = np.arange(q.shape[1])
        cost = np.minimum(np.abs(classes[:, None] - classes[None, :]) / half_screen, 1.0)
        risk = q @ cost.T
        objective = -np.log(np.maximum(q, 1e-15)) + risk_strength * risk
        top_three = np.argsort(-q, axis=1, kind='stable')[:, :3]
        allowed = np.zeros(q.shape, dtype=bool)
        np.put_along_axis(allowed, top_three, True, axis=1)
        allowed &= q >= 0.2 * q.max(axis=1, keepdims=True)
        objective[~allowed] = np.inf
        result.append(objective.argmin(axis=1))
    return np.column_stack(result)

def main():
    source = np.load(ROOT / 'baseline-current-validation.npz')
    p = [source['px'], source['py']]
    truth, idx = source['truth'], source['idx']
    assert np.all((idx >= 260) & (idx < 520))
    baseline_pred = np.column_stack([q.argmax(axis=1) for q in p])
    baseline = metrics(truth, baseline_pred)
    results=[]
    for strength in STRENGTHS:
        name = f'exp17_risk{int(strength)}'
        pred = decode_probabilities(p, strength)
        m=metrics(truth,pred)
        report=dict(name=name,config=dict(risk_strength=strength,top_k=3,min_relative_probability=.2,cost='min(abs(class-target)/half_screen,1)',objective='-log(p)+risk_strength*expected_cost'),validation=m,baseline=baseline,changed_predictions=int(np.any(pred!=baseline_pred,axis=1).sum()),exact_delta=m['exact']-baseline['exact'],far_half_delta=m['far_half_screen']-baseline['far_half_screen'])
        (ROOT/f'{name}-result.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(ROOT/f'{name}-validation.npz',px=p[0],py=p[1],truth=truth,idx=idx,pred=pred,baseline_pred=baseline_pred)
        results.append(report)
    chosen=max(results,key=lambda r:r['validation']['score'])
    summary=dict(baseline=baseline,results=results,chosen=chosen['name'],note='Validation-only decoder comparison. Probabilities remain baseline probabilities; pred contains actual cost-sensitive decisions. No test labels accessed.')
    (ROOT/'exp17-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
