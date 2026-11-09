"""Two fixed diverse probability ensembles; no fitting and no test data access."""
import json,re
import numpy as np
from common import ROOT,decode,metrics

# Methods using the unchanged M/F bank share one family, regardless of classifier,
# training subset, feature selection, decoder, or loss. Other families add their
# own sensor representation (even when the shared bank is also included).
BASE_BANK={13,14,16,18,19,20,21,22,23,24,25,26}

def feature_family(agent):
    return 'shared_motion_lid_optical_bank' if agent in BASE_BANK else f'sensor_representation_{agent:02d}'

def joint_marginals(joint):
    """Row-major 13x20 cell mass; marginal MAP can differ from joint MAP."""
    a=np.asarray(joint,float).reshape(-1,13,20)
    a=a/a.sum(axis=(1,2),keepdims=True)
    return a.sum(1),a.sum(2)

def ensemble_probabilities(probability_pairs,weights):
    """Pure weighted arithmetic pool of normalized per-axis distributions."""
    weights=np.asarray(weights,float)
    if len(weights)!=len(probability_pairs) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid weights')
    result=[]
    for axis in (0,1):
        ps=[np.asarray(pair[axis],float) for pair in probability_pairs]
        if any(p.shape!=ps[0].shape or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(1)<=0) for p in ps):
            raise ValueError('Invalid probabilities')
        result.append(sum(w*p/p.sum(1,keepdims=True) for w,p in zip(weights,ps))/weights.sum())
    return result

def main():
    base=np.load(ROOT/'baseline-current-validation.npz')
    candidates=[];audit=[]
    for path in sorted(ROOT.glob('*-result.json')):
        report=json.loads(path.read_text());name=report['name']
        if name.startswith('exp30'):continue
        match=re.match(r'exp(\d+)',name)
        agent=int(match.group(1)) if match else 0
        z=np.load(ROOT/f'{name}-validation.npz')
        assert np.array_equal(z['idx'],base['idx']) and np.array_equal(z['truth'],base['truth'])
        pair=(z['px'],z['py']);pred=decode(pair)
        record=dict(name=name,agent=agent,reported_score=report['validation']['score'],marginal_score=metrics(z['truth'],pred)['score'])
        if agent in (0,17,28,29):
            record['excluded']='baseline separately weighted' if agent==0 else {17:'risk decoder predictions not represented by saved px/py',28:'confidence-only',29:'nested guarded baseline/template ensemble duplicates members'}[agent]
        else:
            if agent==22:
                pair=joint_marginals(z['joint']);record['conversion']='joint mass marginalized; marginal MAP differs from reported joint MAP'
            else:
                for key in ('pred','predictions'):
                    if key in z:assert np.array_equal(pred,z[key]),name
                assert abs(record['marginal_score']-record['reported_score'])<1e-10,name
            record['family']=feature_family(agent)
            candidates.append((record,pair))
        audit.append(record)
    # Exactly one winning configuration per agent, then one agent per family.
    ranked=sorted(candidates,key=lambda x:(-x[0]['reported_score'],x[0]['name']))
    chosen=[];agents=set();families=set()
    for rec,pair in ranked:
        if rec['agent'] in agents:continue
        agents.add(rec['agent'])
        if rec['family'] in families:continue
        families.add(rec['family']);chosen.append((rec,pair))
    for k in (3,5):
        members=[dict(name='baseline-current',weight=2.)]+[dict(name=r['name'],weight=1.,family=r['family']) for r,p in chosen[:k]]
        p=ensemble_probabilities([(base['px'],base['py'])]+[pair for rec,pair in chosen[:k]],[m['weight'] for m in members])
        name=f'exp30-diverse{k}';pred=decode(p)
        report=dict(name=name,config=dict(policy=f'baseline weight 2 plus top {k} distinct feature families weight 1',members=members),validation=metrics(base['truth'],pred),extra=dict(selection='ranked by predeclared shared validation score; no test access',family_rule='unchanged M/F-bank methods share one family; added sensor representations form separate families',joint_caveat='exp22 can be marginalized but joint MAP is not equivalent; excluded here by shared-bank family deduplication',candidate_audit=audit))
        (ROOT/f'{name}-result.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(ROOT/f'{name}-validation.npz',px=p[0],py=p[1],pred=pred,truth=base['truth'],idx=base['idx'])
        print(json.dumps(dict(name=name,members=members,validation=report['validation'])))
if __name__=='__main__':main()
