import Foundation

/// Append-only local journal. Sensor file shares the collector's <Qdddd> format.
final class PlayRecorder {
    let folder:URL
    private var raw:FileHandle,events:FileHandle,predictions:FileHandle,labels:FileHandle
    private(set) var predictionCount=0,labelCount=0,block=1
    private var labelIDs=[String:String]()
    private var predictionBlocks=[String:Int]()
    private var labelled=Set<String>()
    private(set) var manualCount=0
    private var closed=false,lastSync=0.0
    init(storageRoot:URL?=nil,sourceModelRoot:URL?=nil,activeModel:URL?=nil) throws {
        let root=Bundle.main.bundleURL.deletingLastPathComponent()
        let name=ISO8601DateFormatter().string(from:Date()).replacingOccurrences(of:":",with:"-")+"-"+UUID().uuidString.prefix(6)
        let targetFolder=(storageRoot ?? root.appendingPathComponent("tap-play-data")).appendingPathComponent(name)
        folder=targetFolder
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        func file(_ name:String)throws->FileHandle {
            let url=targetFolder.appendingPathComponent(name)
            guard FileManager.default.createFile(atPath:url.path,contents:nil) else{throw NSError(domain:"CreateDataFile",code:1)}
            return try FileHandle(forWritingTo:url)
        }
        raw=try file("samples.bin");events=try file("events.jsonl");predictions=try file("predictions.jsonl");labels=try file("labels.jsonl")
        let modelRoot=sourceModelRoot ?? root.appendingPathComponent("tap-model")
        let snapshot=folder.appendingPathComponent("model-snapshot");try FileManager.default.createDirectory(at:snapshot,withIntermediateDirectories:true)
        for path in ["light-dynamics/model.joblib","light-dynamics/train.py","no-microphone/train.py","train_model.py","live/worker.py","live/feedback_training.py"] {
            let dest=snapshot.appendingPathComponent(path);try FileManager.default.createDirectory(at:dest.deletingLastPathComponent(),withIntermediateDirectories:true);try FileManager.default.copyItem(at:modelRoot.appendingPathComponent(path),to:dest)
        }
        if let activeModel,FileManager.default.fileExists(atPath:activeModel.path){try FileManager.default.copyItem(at:activeModel,to:snapshot.appendingPathComponent("active-model.joblib"))}
        let metadata:[String:Any] = ["schema":1,"purpose":"Free-play sensor capture with optional user mouse ground truth; manual supervised replay updates on key1; all versions retained","created_at":ISO8601DateFormatter().string(from:Date()),"clock":"mach_absolute_time seconds, shared with HID timestamps","cols":20,"rows":13,"audio_used":false,"raw_format":"40 bytes little endian <Qdddd>: kind,t,x,y,z; 0 accel g,1 gyro deg/s,2 lid degrees,3 lux,4 raw optical channels0/1/2,5 raw optical channel3","lid_source":"feature report7 preferred; Double, no integer rounding","lid_read_requested_hz":200,"sensor_policy":"Continuous while session exists, including pauses; pause stops prediction, not capture","label_policy":"Only mouse/manual annotations are ground truth. Predictions never become labels. Latest valid label event supersedes earlier annotations for same prediction id.","prediction_window":"trigger-0.9..trigger+0.4 seconds, all sensor streams; extras/ignored onsets recorded as events","snapshot":"model-snapshot contains base model and exact feature sources","os":ProcessInfo.processInfo.operatingSystemVersionString]
        try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys]).write(to:folder.appendingPathComponent("metadata.json"),options:.atomic)
        try event("session_start")
    }
    private func append(_ value:[String:Any],_ file:FileHandle)throws {
        var data=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]);data.append(10);try file.write(contentsOf:data)
    }
    func event(_ type:String,extra:[String:Any]=[:])throws {
        guard !closed else{return};var row=extra;row["event"]=type;row["t"]=MachClock.now();row["block"]=block;try append(row,events)
    }
    func samples(_ batch:[MotionSample])throws {
        guard !closed else{return};var data=Data(capacity:batch.count*40)
        for s in batch {
            var k=UInt64(s.kind).littleEndian;withUnsafeBytes(of:&k){data.append(contentsOf:$0)}
            for v in [s.t,s.v.x,s.v.y,s.v.z]{var bits=v.bitPattern.littleEndian;withUnsafeBytes(of:&bits){data.append(contentsOf:$0)}}
        }
        try raw.write(contentsOf:data)
        if MachClock.now()-lastSync>=1 {try sync()}
    }
    func prediction(id:String,trigger:Double,lid:Double?,samples:[MotionSample],details:[String:Any],onsets:[Double],captureBlock:Int)throws {
        let row:[String:Any] = ["prediction_id":id,"block":captureBlock,"trigger_t":trigger,"lid_angle":lid as Any? ?? NSNull(),"created_t":MachClock.now(),"model":details,"window_start":trigger-0.9,"window_end":trigger+0.4,"samples":samples.map{[Double($0.kind),$0.t,$0.v.x,$0.v.y,$0.v.z]},"detected_onsets":onsets,"multiple_onsets":onsets.count>1,"label_source":"unlabelled_until_explicit_mouse_annotation"]
        try append(row,predictions);try event("prediction_saved",extra:["prediction_id":id]);try sync();predictionCount+=1;predictionBlocks[id]=captureBlock
    }
    func label(id:String,x:Double,y:Double,pixel:[Double],base:[Int],candidates:[[Int]],rawClick:[Double]?=nil)throws {
        let eid=UUID().uuidString;let col=min(19,max(0,Int(x*20))),row=min(12,max(0,Int(y*13)))
        let value:[String:Any] = ["event":"mouse_ground_truth","event_id":eid,"prediction_id":id,"t":MachClock.now(),"block":predictionBlocks[id] ?? block,"annotation_block":block,"x_normalized":x,"y_normalized":y,"col":col,"row":row,"view_xy_top_left":pixel,"raw_click_view_xy_top_left":rawClick ?? pixel,"source":"explicit_user_mouse_annotation_after_prediction","supersedes_event_id":labelIDs[id] as Any? ?? NSNull(),"predicted_col":base[0],"predicted_row":base[1],"shown_candidates":candidates]
        try append(value,labels);try sync();labelIDs[id]=eid;labelled.insert(id);labelCount=labelled.count
    }
    func manualLabel(x:Double,y:Double,control:Double,start:Double,id:String?=nil,samples:[MotionSample]=[],searchStart:Double?=nil,rawClick:[Double]?=nil) throws {
        let eventID=UUID().uuidString,pid=id ?? UUID().uuidString
        let value:[String:Any] = ["event":"missed_tap_ground_truth","event_id":eventID,"prediction_id":pid,"t":MachClock.now(),"block":block,"trigger_t":NSNull(),"tap_timestamp_known":false,"interval_start":start,"interval_end":samples.last?.t ?? control-0.08,"control_t":control,"search_start":searchStart ?? start+0.15,"manual_samples":samples.map{[Double($0.kind),$0.t,$0.v.x,$0.v.y,$0.v.z]},"x_normalized":x,"y_normalized":y,"col":min(19,max(0,Int(x*20))),"row":min(12,max(0,Int(y*13))),"raw_click_view_xy_top_left":rawClick as Any? ?? NSNull(),"supersedes_event_id":labelIDs[pid] as Any? ?? NSNull(),"source":"user_marks_missed_tap; trigger must be estimated from motion, not control time"]
        try append(value,labels);try sync();if !labelled.contains(pid){manualCount+=1};labelIDs[pid]=eventID;labelled.insert(pid);labelCount=labelled.count
    }
    func newBlock(lid:Double?)throws {block+=1;try event("new_conditions",extra:["lid_angle":lid as Any? ?? NSNull()]);try sync()}
    func sync()throws {try raw.synchronize();try events.synchronize();try predictions.synchronize();try labels.synchronize();lastSync=MachClock.now()}
    func finish()throws {
        guard !closed else{return};try event("session_end",extra:["predictions":predictionCount,"manual_taps":manualCount,"labelled_predictions":labelCount]);try sync()
        try raw.close();try events.close();try predictions.close();try labels.close();closed=true
    }
}
