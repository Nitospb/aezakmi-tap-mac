from pathlib import Path
from collections import Counter
import json,gzip,math,zipfile,hashlib,struct
src=Path('outputs/tap-data/2026-11-07T15-48-03Z-732E01')
out=Path('outputs/tap-dataset')
meta=json.loads((src/'metadata.json').read_text())
events=[json.loads(x) for x in (src/'events.jsonl').read_text().splitlines() if x.strip()]
assert any(e.get('event')=='finish' for e in events), 'Session not yet finalized'
invalid={e['trial_id'] for e in events if e.get('event')=='invalidate_trial'}
counts=Counter(); sources=Counter(); angles=Counter(); errors=[]; excluded=0; total=0; seen=set()
with gzip.open(out/'trials.valid.jsonl.gz','wt',encoding='utf-8') as dst:
    for line in (src/'trials.jsonl').open():
        r=json.loads(line); total+=1
        if r['trial_id'] in invalid: excluded+=1; continue
        assert r['trial_id'] not in seen
        seen.add(r['trial_id'])
        assert 0<=r['zone']<meta['cols']*meta['rows']
        assert 0<r['x_normalized']<1 and 0<r['y_normalized']<1
        for key in ('accel_g','gyro_dps'):
            rows=r[key]
            assert len(rows)>100
            assert all(len(a)==4 and all(math.isfinite(v) for v in a) for a in rows)
            assert all(a[0]<=b[0] for a,b in zip(rows,rows[1:]))
        sources[r.get('capture_source','automatic')]+=1
        counts[r['zone']]+=1; angles[r['lid_angle']]+=1
        dst.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
raw=src/'samples.bin'
assert raw.stat().st_size%40==0
with raw.open('rb') as f:
    raw_count=0
    while block:=f.read(40*32768):
        for kind,t,x,y,z in struct.iter_unpack('<Qdddd',block):
            assert kind in (0,1) and all(math.isfinite(v) for v in (t,x,y,z))
            raw_count+=1
report={'source_session':src.name,'valid_trials':len(seen),'total_written_trials':total,'invalidated_repeats_excluded':excluded,'capture_sources':dict(sources),'grid':[meta['cols'],meta['rows']],'covered_cells':len(counts),'per_cell_min':min(counts.values()),'per_cell_max':max(counts.values()),'cell_counts':dict(sorted(counts.items())),'lid_angles_degrees':dict(sorted(angles.items(),key=lambda kv: kv[0] if kv[0] is not None else -1)),'raw_sample_count':raw_count,'notes':['Only this completed 20x13 session is included; earlier pilot sessions are separate.','Manual confirmations contain the full prompt interval with unknown exact tap timing. Do not use confirmation time as the tap time.','Target centers are instructed labels; finger location was not independently observed.','No predictive accuracy has been measured. Use an independent session to evaluate a future model.']}
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile('outputs/tap-dataset.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for name in ('trials.valid.jsonl.gz','summary.json','metadata.json'):
        z.write(out/name,name)
    z.write(src/'events.jsonl','events.jsonl')
    z.write(raw,'samples.bin')
with zipfile.ZipFile('outputs/tap-dataset.zip') as z:
    assert z.testzip() is None
print(json.dumps({k:v for k,v in report.items() if k not in ('cell_counts','notes')},ensure_ascii=False))
print('Archive verified:',Path('outputs/tap-dataset.zip').stat().st_size,'bytes')
