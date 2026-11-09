import Foundation

struct MotionSample {
    let kind: Int
    let t: Double
    let v: SIMD3<Double>
}

struct Signal {
    let times: [Double]
    let raw: [SIMD3<Double>]
    let high: [SIMD3<Double>]
    init(_ samples: [MotionSample]) {
        times = samples.map{$0.t}; raw = samples.map{$0.v}
        var base = samples.first?.v ?? .zero
        var result: [SIMD3<Double>] = []
        for s in samples { base += 0.038*(s.v-base); result.append(s.v-base) }
        high = result
    }
    func interpolate(_ t: Double, values: [SIMD3<Double>]) -> SIMD3<Double> {
        guard t > times[0] else { return values[0] }
        guard t < times[times.count-1] else { return values[values.count-1] }
        var lo=0, hi=times.count-1
        while hi-lo>1 { let mid=(hi+lo)/2; if times[mid]<=t { lo=mid } else { hi=mid } }
        let w=(t-times[lo])/max(1e-12,times[hi]-times[lo])
        return values[lo]*(1-w)+values[hi]*w
    }
    func magnitudeAt(_ t: Double) -> Double {
        guard t > times[0] else { return magnitude(high[0]) }
        guard t < times[times.count-1] else { return magnitude(high[high.count-1]) }
        var lo=0, hi=times.count-1
        while hi-lo>1 { let mid=(hi+lo)/2; if times[mid]<=t { lo=mid } else { hi=mid } }
        let w=(t-times[lo])/max(1e-12,times[hi]-times[lo])
        return magnitude(high[lo])*(1-w)+magnitude(high[hi])*w
    }
}
func magnitude(_ v: SIMD3<Double>) -> Double { sqrt(v.x*v.x+v.y*v.y+v.z*v.z) }

func motionFeatures(samples: [MotionSample], trigger: Double, lid: Int?) throws -> [Double] {
    let aa=samples.filter{$0.kind==0}, gg=samples.filter{$0.kind==1}
    guard aa.count>100,gg.count>100 else { throw ModelError.invalid("Недостаточно отсчётов") }
    let accel=Signal(aa), gyro=Signal(gg)
    var center: Double?, best = -Double.infinity
    for i in gyro.times.indices where gyro.times[i]>=trigger-0.015 && gyro.times[i]<=trigger+0.12 {
        let t=gyro.times[i], energy=magnitude(gyro.high[i])/2+accel.magnitudeAt(t)/0.04
        if energy>best { best=energy; center=t }
    }
    guard let center else { throw ModelError.invalid("Не найден импульс") }
    var f: [Double]=[], scales: [Double]=[]
    for signal in [gyro,accel] {
        let wave=(0..<32).map { signal.interpolate(center-0.02+Double($0)*0.24/31,values:signal.high) }
        let scale=max(wave.map{magnitude($0)}.max()!,1e-8); scales.append(scale)
        let normalized=wave.map{$0/scale}
        let rms=sqrt(wave.reduce(0.0){$0+$1.x*$1.x+$1.y*$1.y+$1.z*$1.z}/32)
        f += [log1p(scale),log1p(rms)]
        for k in 0..<3 {
            let a=normalized.map{$0[k]}
            let maxIndex=a.indices.max { abs(a[$0])<abs(a[$1]) }!
            // NumPy argmax uses the first occurrence; ties are unusual but kept consistent.
            let firstMax=a.indices.first{abs(a[$0]) == abs(a[maxIndex])}!
            let changes=(1..<32).filter{a[$0]*a[$0-1]<0}.count
            f += [a.max()!,a.min()!,sqrt(a.reduce(0){$0+$1*$1}/32),a.prefix(8).reduce(0,+)/8,a.suffix(24).reduce(0,+)/24,a.reduce(0){$0+abs($1)}/32,Double(firstMax)/31,Double(changes)/31]
        }
        for (i,j) in [(0,1),(0,2),(1,2)] {
            let dot=normalized.reduce(0.0){$0+$1[i]*$1[j]}
            let ni=normalized.reduce(0.0){$0+$1[i]*$1[i]}, nj=normalized.reduce(0.0){$0+$1[j]*$1[j]}
            f.append(dot/max(sqrt(ni*nj),1e-12))
        }
        for v in normalized { f += [v.x,v.y,v.z] }
    }
    for signal in [gyro,accel] {
        let before=signal.times.indices.filter{signal.times[$0]<trigger-0.04}
        let mean=before.isEmpty ? signal.raw[0] : before.reduce(SIMD3<Double>.zero){$0+signal.raw[$1]}/Double(before.count)
        f += [mean.x,mean.y,mean.z]
    }
    f += [Double(lid ?? 120)/180,log((scales[0]+1e-8)/(scales[1]+1e-8))]
    guard f.allSatisfy({$0.isFinite}) else { throw ModelError.invalid("Некорректный сигнал") }
    return f
}

enum ModelError: Error { case invalid(String) }
indirect enum RegressionTree {
    case leaf(Double)
    case branch(Int,Double,RegressionTree,RegressionTree)
    init(_ node: [String:Any]) throws {
        if let value=node["leaf_value"] as? Double { self = .leaf(value); return }
        guard let feature=node["split_feature"] as? Int, let threshold=node["threshold"] as? Double,
              node["decision_type"] as? String == "<=", let left=node["left_child"] as? [String:Any], let right=node["right_child"] as? [String:Any] else { throw ModelError.invalid("Неподдерживаемое дерево") }
        self = .branch(feature,threshold,try RegressionTree(left),try RegressionTree(right))
    }
    func predict(_ x: [Double]) -> Double {
        switch self { case .leaf(let v): return v
        case .branch(let f,let t,let l,let r): return x[f]<=t ? l.predict(x) : r.predict(x) }
    }
}
struct RegionPrediction {
    let nx: Int
    let ny: Int
    let zone: Int
}
struct RegionHead {
    let nx: Int
    let ny: Int
    let classes: [Int]
    let trees: [RegressionTree]
    let coef: [[Double]]
    let intercept: [Double]
    func probabilities(_ f: [Double]) -> [Double] {
        var scores=[Double](repeating:0,count:classes.count)
        if coef.isEmpty {
            for (i,t) in trees.enumerated() {scores[i % scores.count] += t.predict(f)}
        } else {
            for i in coef.indices {scores[i]=zip(coef[i],f).reduce(intercept[i]){$0+$1.0*$1.1}}
        }
        let m=scores.max()!,e=scores.map{exp($0-m)},total=e.reduce(0,+)
        var full=[Double](repeating:0,count:nx*ny)
        for i in classes.indices {full[classes[i]]=e[i]/total};return full
    }
}
final class SpatialModel {
    let featureCount: Int
    var kind="regression", decoder="map"
    var forests:[[RegressionTree]]=[]
    var heads:[(classes:[Int],trees:[RegressionTree])]=[]
    var linearHeads:[(classes:[Int],coef:[[Double]],intercept:[Double])]=[]
    var templates:[[Double]]=[], labels:[[Int]]=[]
    var mean:[Double]=[],scale:[Double]=[]
    var regionHeads: [RegionHead]=[]
    var lastRegions: [RegionPrediction]=[]
    var multiscaleWeight=0.0
    init(url:URL) throws {
        guard let root=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],root["feature_version"] as? Int==1,let count=root["feature_count"] as? Int else {throw ModelError.invalid("Некорректная модель")}
        featureCount=count;kind=root["kind"] as? String ?? "regression";decoder=root["decoder"] as? String ?? "map"
        mean=root["mean"] as? [Double] ?? [];scale=root["scale"] as? [Double] ?? []
        multiscaleWeight=root["multiscale_weight"] as? Double ?? 0
        for head in root["region_heads"] as? [[String:Any]] ?? [] {
            let trees=try (head["trees"] as? [[String:Any]] ?? []).map{try RegressionTree($0["tree_structure"] as! [String:Any])}
            regionHeads.append(RegionHead(nx:head["nx"] as! Int,ny:head["ny"] as! Int,classes:head["classes"] as! [Int],trees:trees,coef:head["coef"] as? [[Double]] ?? [],intercept:head["intercept"] as? [Double] ?? []))
        }
        if kind=="nearest" {
            templates=root["templates"] as! [[Double]];labels=root["labels"] as! [[Int]]
        } else if kind=="linear_axes" {
            guard let h=root["linear_heads"] as? [[String:Any]],h.count==2 else {throw ModelError.invalid("Некорректные линейные модели")}
            for head in h {linearHeads.append((head["classes"] as! [Int],head["coef"] as! [[Double]],head["intercept"] as! [Double]))}
        } else if kind=="axes" || kind=="cells" {
            guard let h=root["heads"] as? [[String:Any]],h.count==(kind=="axes" ? 2:1) else {throw ModelError.invalid("Некорректные классификаторы")}
            for head in h {heads.append((head["classes"] as! [Int],try (head["trees"] as! [[String:Any]]).map{try RegressionTree($0["tree_structure"] as! [String:Any])}))}
        } else {
            guard let models=root["models"] as? [[[String:Any]]],models.count==2 else {throw ModelError.invalid("Некорректная регрессия")}
            forests=try models.map{try $0.map{try RegressionTree($0["tree_structure"] as! [String:Any])}}
        }
    }
    func softmax(_ scores:[Double])->[Double] {
        let m=scores.max()!,e=scores.map{exp($0-m)},total=e.reduce(0,+);return e.map{$0/total}
    }
    func band(_ i:Int,_ n:Int)->Int {Double(i)<Double(n)/3 ? 0 : (Double(i)>=Double(n)*2/3 ? 2:1)}
    func decode(_ px:[Double],_ py:[Double])->SIMD2<Double> {
        if decoder != "hierarchical" {
            var x=0,y=0
            for i in px.indices where px[i]>px[x] {x=i};for i in py.indices where py[i]>py[y] {y=i}
            return SIMD2(Double(x)+0.5,Double(y)+0.5)
        }
        var ax=[Double](repeating:0,count:3),by=[Double](repeating:0,count:3)
        for c in 0..<20 {ax[band(c,20)] += px[c]};for r in 0..<13 {by[band(r,13)] += py[r]}
        let frameProbability=1-(1-px[0]-px[19])*(1-py[0]-py[12])
        var best = -Double.infinity,result=SIMD2<Double>(0.5,0.5)
        for r in 0..<13 {for c in 0..<20 {
            let edge=c==0 || c==19 || r==0 || r==12
            let corner=(c==0 || c==19) && (r==0 || r==12)
            var score=Double(10+(edge ? 8:0)+(corner ? 4:0))*px[c]*py[r]+2*px[c]+2*py[r]
            score += 1.5*ax[band(c,20)]+1.5*by[band(r,13)]
            if band(c,20)==1 && band(r,13)==1 {score -= 4*frameProbability}
            if score>best {best=score;result=SIMD2(Double(c)+0.5,Double(r)+0.5)}
        }}
        return result
    }
    func predict(features:[Double]) throws -> SIMD2<Double> {
        guard features.count==featureCount else {throw ModelError.invalid("Размер признаков")}
        let f=mean.isEmpty ? features : features.indices.map{(features[$0]-mean[$0])/scale[$0]}
        if kind=="nearest" {
            var best=Double.infinity,index=0
            for i in templates.indices {
                var d=0.0;for j in f.indices {let x=f[j]-templates[i][j];d += x*x}
                if d<best {best=d;index=i}
            }
            return SIMD2(Double(labels[index][0])+0.5,Double(labels[index][1])+0.5)
        }
        if kind=="axes" || kind=="cells" || kind=="linear_axes" {
            var probs:[[Double]]=[]
            if kind=="linear_axes" {
                for (k,head) in linearHeads.enumerated() {
                    let scores=head.coef.enumerated().map{(i,row) in zip(row,f).reduce(head.intercept[i]){$0+$1.0*$1.1}}
                    let p=softmax(scores);var full=[Double](repeating:0,count:k==0 ? 20:13)
                    for i in head.classes.indices {full[head.classes[i]]=p[i]};probs.append(full)
                }
            } else {
                for (k,head) in heads.enumerated() {
                    var scores=[Double](repeating:0,count:head.classes.count)
                    for (i,tree) in head.trees.enumerated(){scores[i % scores.count] += tree.predict(f)}
                    let p=softmax(scores)
                    if kind=="cells" {
                        var winner=0;for i in scores.indices where scores[i]>scores[winner]{winner=i}
                        let z=head.classes[winner];return SIMD2(Double(z%20)+0.5,Double(z/20)+0.5)
                    }
                    var full=[Double](repeating:0,count:k==0 ? 20:13)
                    for i in head.classes.indices{full[head.classes[i]]=p[i]};probs.append(full)
                }
            }
            if decoder=="multiscale" {
                let regionProbs=regionHeads.map{$0.probabilities(f)}
                lastRegions=[]
                for (head,p) in zip(regionHeads,regionProbs) {
                    var z=0;for i in p.indices where p[i]>p[z] {z=i}
                    lastRegions.append(RegionPrediction(nx:head.nx,ny:head.ny,zone:z))
                }
                var best = -Double.infinity,result=SIMD2<Double>(0.5,0.5)
                for r in 0..<13 {for c in 0..<20 {
                    var score=log(max(probs[0][c],1e-12))+log(max(probs[1][r],1e-12))
                    for (head,p) in zip(regionHeads,regionProbs) {
                        let x=min(head.nx-1,Int((Double(c)+0.5)*Double(head.nx)/20))
                        let y=min(head.ny-1,Int((Double(r)+0.5)*Double(head.ny)/13))
                        score += multiscaleWeight*log(max(p[y*head.nx+x],1e-12))
                    }
                    if score>best {best=score;result=SIMD2(Double(c)+0.5,Double(r)+0.5)}
                }}
                // Display the fixed regions containing the final selected cell.
                lastRegions=regionHeads.map {head in
                    let x=min(head.nx-1,Int(result.x*Double(head.nx)/20))
                    let y=min(head.ny-1,Int(result.y*Double(head.ny)/13))
                    return RegionPrediction(nx:head.nx,ny:head.ny,zone:y*head.nx+x)
                }
                return result
            }
            return decode(probs[0],probs[1])
        }
        let values=forests.map{$0.reduce(0.0){$0+$1.predict(f)}};return SIMD2(values[0],values[1])
    }
}
