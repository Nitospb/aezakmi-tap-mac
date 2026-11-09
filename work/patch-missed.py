from pathlib import Path
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text()
s=s.replace('else if event.keyCode==29 {controller?.resetModel()}', 'else if event.keyCode==29 {controller?.resetModel()}\n        else if event.keyCode==21 {controller?.missedTap()}')
s=s.replace('else if c.locked,c.prediction != nil {', 'else if c.locked,c.manualID != nil {text=c.confirmed ? "Пропущенный тап сохранён · 1 — дообучить · → — дальше" : "Пропущенный тап · укажи клетку мышью"}\n        else if c.locked,c.prediction != nil {')
s=s.replace('3 — варианты · → — следующий тап','3 — варианты · 4 — не распознал · → — следующий тап')
s=s.replace('var manualControl:Double?', 'var manualControl:Double?\n    var manualID:String?,manualSamples:[MotionSample]=[],manualEarliest=0.0,lastRelease=0.0')
s=s.replace('r.predictionCount) тапов', 'r.predictionCount+r.manualCount) тапов')
s=s.replace('        let x=min(0.999999,max(0,Double(point.x/view.bounds.width))),y=min(0.999999,max(0,Double(point.y/view.bounds.height)))', '''        let rawX=min(0.999999,max(0,Double(point.x/view.bounds.width))),rawY=min(0.999999,max(0,Double(point.y/view.bounds.height)))
        let col=Int(rawX*20),row=Int(rawY*13),x=(Double(col)+0.5)/20,y=(Double(row)+0.5)/13
        let centerPixel=[x*Double(view.bounds.width),y*Double(view.bounds.height)],rawPixel=[Double(point.x),Double(point.y)]''')
s=s.replace('if let control=manualControl {try r.manualLabel(x:x,y:y,control:control,start:control-3);manualControl=nil;armedAfter=MachClock.now()+0.6}', 'if let control=manualControl,let id=manualID {try r.manualLabel(x:x,y:y,control:control,start:manualSamples.first?.t ?? control-8,id:id,samples:manualSamples,searchStart:manualEarliest,rawClick:rawPixel)}')
s=s.replace('pixel:[Double(point.x),Double(point.y)],base:[a.col,a.row],candidates:a.candidates)', 'pixel:centerPixel,base:[a.col,a.row],candidates:a.candidates,rawClick:rawPixel)')
s=s.replace('    func nextTap(){', '''    func missedTap(){
        guard recorder != nil,saveFailure==nil,!training,!predicting,pending==nil else{return}
        guard !locked else{flash("Сначала →, чтобы завершить текущую отметку");return}
        let control=MachClock.now(),end=control-0.08
        manualSamples=ring.filter{$0.t>=max(lastRelease-0.9,control-8) && $0.t<=end}
        guard manualSamples.count>200 else{flash("Сигнала пока мало · подожди после тапа");return}
        manualControl=control;manualEarliest=lastRelease+0.45;manualID=UUID().uuidString;locked=true;confirmed=false;ready=false
        prediction=nil;latestID=nil;candidates=[];truthPoint=nil;notice=nil;error=nil
        log("missed_tap_requested",["manual_id":manualID!,"control_t":control,"search_start":manualEarliest,"window_end":end]);status();view.needsDisplay=true
    }
    func nextTap(){''')
s=s.replace('manualControl=nil;notice=nil;error=nil', 'manualControl=nil;manualID=nil;manualSamples=[];lastRelease=MachClock.now();notice=nil;error=nil')
s=s.replace('pending=nil;if !locked{prediction=nil}', 'pending=nil;if !locked{prediction=nil;lastRelease=MachClock.now()}')
s=s.replace('ring.removeAll{$0.t<now-3}', 'ring.removeAll{$0.t<now-9}')
p.write_text(s)
p=Path('outputs/tap-model/live/PlayRecorder.swift');s=p.read_text().replace('private var manualCount=0','private(set) var manualCount=0')
s=s.replace('candidates:[[Int]])throws', 'candidates:[[Int]],rawClick:[Double]?=nil)throws')
s=s.replace('"view_xy_top_left":pixel,"source"', '"view_xy_top_left":pixel,"raw_click_view_xy_top_left":rawClick ?? pixel,"source"')
s=s.replace('labelCount=labelled.count+manualCount','labelCount=labelled.count')
a=s.index('    func manualLabel(');b=s.index('    func newBlock(',a)
s=s[:a]+'''    func manualLabel(x:Double,y:Double,control:Double,start:Double,id:String?=nil,samples:[MotionSample]=[],searchStart:Double?=nil,rawClick:[Double]?=nil) throws {
        let eventID=UUID().uuidString,pid=id ?? UUID().uuidString
        let value:[String:Any] = ["event":"missed_tap_ground_truth","event_id":eventID,"prediction_id":pid,"t":MachClock.now(),"block":block,"trigger_t":NSNull(),"tap_timestamp_known":false,"interval_start":start,"interval_end":samples.last?.t ?? control-0.08,"control_t":control,"search_start":searchStart ?? start+0.15,"manual_samples":samples.map{[Double($0.kind),$0.t,$0.v.x,$0.v.y,$0.v.z]},"x_normalized":x,"y_normalized":y,"col":min(19,max(0,Int(x*20))),"row":min(12,max(0,Int(y*13))),"raw_click_view_xy_top_left":rawClick as Any? ?? NSNull(),"supersedes_event_id":labelIDs[pid] as Any? ?? NSNull(),"source":"user_marks_missed_tap; trigger must be estimated from motion, not control time"]
        try append(value,labels);try sync();if !labelled.contains(pid){manualCount+=1};labelIDs[pid]=eventID;labelled.insert(pid);labelCount=labelled.count
    }
'''+s[b:]
s=s.replace('"predictions":predictionCount,"labelled_predictions"', '"predictions":predictionCount,"manual_taps":manualCount,"labelled_predictions"')
p.write_text(s)
p=Path('outputs/tap-model/live/TrainingControls.swift');s=p.read_text().replace('self.locked=false;self.confirmed=false;', 'self.locked=false;self.confirmed=false;self.manualID=nil;self.manualControl=nil;self.manualSamples=[];self.lastRelease=MachClock.now();')
p.write_text(s)
