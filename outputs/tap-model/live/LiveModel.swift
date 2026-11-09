import Foundation
struct MotionSample { let kind:Int;let t:Double;let v:SIMD3<Double> }
struct PredictionResult {let point:SIMD2<Double>;let details:[String:Any]}
final class LiveModel {
    let process=Process(),input=Pipe(),output=Pipe(),queue=DispatchQueue(label:"tap.model.worker")
    var ready=false
    var version=0,trainedFeedback=0
    var modelPath:String?
    var lastRegions:[(nx:Int,ny:Int,zone:Int)]=[]
    private var bytes=Data()
    var onStatus:((String?)->Void)?
    func line() throws -> [String:Any] {
        while true {
            if let end=bytes.firstIndex(of:10) {let data=bytes.prefix(upTo:end);bytes.removeSubrange(...end);return try JSONSerialization.jsonObject(with:data) as! [String:Any]}
            let data=output.fileHandleForReading.availableData
            guard !data.isEmpty else{throw NSError(domain:"Worker",code:1,userInfo:[NSLocalizedDescriptionKey:"Процесс модели завершился"])}
            bytes.append(data)
        }
    }
    func start() {
        let root=Bundle.main.bundleURL.deletingLastPathComponent()
        process.executableURL=root.deletingLastPathComponent().appendingPathComponent("work/model-env/bin/python")
        process.arguments=[Bundle.main.resourceURL!.appendingPathComponent("worker.py").path,root.appendingPathComponent("tap-model").path]
        var env=ProcessInfo.processInfo.environment;env["OPENBLAS_NUM_THREADS"]="2";env["OMP_NUM_THREADS"]="2";env["PYTHONUNBUFFERED"]="1";process.environment=env
        process.standardInput=input;process.standardOutput=output
        queue.async { [self] in
            do {try process.run();let status=try line();guard status["ready"] as? Bool==true else{throw NSError(domain:"Worker",code:2)};DispatchQueue.main.async{self.ready=true;self.version=status["version"] as? Int ?? 0;self.modelPath=status["model_path"] as? String;self.trainedFeedback=status["training_feedback"] as? Int ?? 0;self.onStatus?(nil)}}
            catch {DispatchQueue.main.async{self.onStatus?(error.localizedDescription)}}
        }
    }
    func predict(samples:[MotionSample],trigger:Double,lid:Double?,requestID:String,completion:@escaping(Result<PredictionResult,Error>)->Void) {
        queue.async { [self] in
            do {
                let rows:[[Double]]=samples.map { s in [Double(s.kind),s.t,s.v.x,s.v.y,s.v.z] }
                let angle:Any=lid.map{ $0 as Any } ?? NSNull()
                let request:[String:Any] = ["trigger":trigger,"lid":angle,"samples":rows,"request_id":requestID]
                var data=try JSONSerialization.data(withJSONObject:request);data.append(10);try input.fileHandleForWriting.write(contentsOf:data)
                let response=try line()
                if let error=response["error"] as? String{throw NSError(domain:"Model",code:3,userInfo:[NSLocalizedDescriptionKey:error])}
                guard let col=response["col"] as? Int,let row=response["row"] as? Int else{throw NSError(domain:"Model",code:4)}
                DispatchQueue.main.async {
                    self.lastRegions=[]
                    for n in [2,3,4] {
                        let yy=min(n-1,Int((Double(row)+0.5)*Double(n)/13.0))
                        let xx=min(n-1,Int((Double(col)+0.5)*Double(n)/20.0))
                        self.lastRegions.append((nx:n,ny:n,zone:yy*n+xx))
                    }
                    completion(.success(PredictionResult(point:SIMD2(Double(col)+0.5,Double(row)+0.5),details:response)))
                }
            } catch {DispatchQueue.main.async{completion(.failure(error))}}
        }
    }
    func command(_ request:[String:Any],completion:@escaping(Result<[String:Any],Error>)->Void){
        queue.async { [self] in
            do{
                var data=try JSONSerialization.data(withJSONObject:request);data.append(10);try input.fileHandleForWriting.write(contentsOf:data)
                let response=try line();if let error=response["error"] as? String{throw NSError(domain:"Training",code:1,userInfo:[NSLocalizedDescriptionKey:error])}
                DispatchQueue.main.async{self.version=response["version"] as? Int ?? self.version;self.modelPath=response["model_path"] as? String ?? self.modelPath;self.trainedFeedback=response["training_feedback"] as? Int ?? self.trainedFeedback;completion(.success(response))}
            }catch{DispatchQueue.main.async{completion(.failure(error))}}
        }
    }
    func stop(){if process.isRunning{process.terminate()}}
}
