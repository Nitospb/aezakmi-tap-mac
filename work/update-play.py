from pathlib import Path
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text()
s=s.replace('final class PredictionView:NSView', 'final class PredictionView:NSView') # existing spacing preserved
s=s.replace('final class PredictionView: NSView {', '''final class PassThroughLabel:NSTextField {override func hitTest(_ point:NSPoint)->NSView?{nil}}
final class PredictionView: NSView {''')
s=s.replace('else if event.keyCode==53 {controller?.pause()}', '''else if event.keyCode==53 {controller?.pause()}
        else if event.keyCode==124 {controller?.missedTap()}
        else if event.keyCode==45 {controller?.newConditions()}
        else if event.keyCode==20 {controller?.showCandidates.toggle();needsDisplay=true}''')
s=s.replace('    override func draw(_ dirtyRect:NSRect) {', '''    override func mouseDown(with event:NSEvent){controller?.mark(convert(event.locationInWindow,from:nil))}
    override func draw(_ dirtyRect:NSRect) {''')
a=s.index('        if c.prediction != nil {');b=s.index('        if let p=c.prediction {',a)
s=s[:a]+'''        if c.showCandidates {
            for (i,p) in c.candidates.enumerated() where i>0 {
                let r=NSRect(x:floor(CGFloat(p.x))*w,y:floor(CGFloat(p.y))*h,width:w,height:h).insetBy(dx:3,dy:3)
                NSColor.systemOrange.setStroke();let path=NSBezierPath(roundedRect:r,xRadius:5,yRadius:5);path.lineWidth=3;path.stroke()
                ("\\(i+1)" as NSString).draw(at:NSPoint(x:r.minX+6,y:r.minY+5),withAttributes:[.font:NSFont.systemFont(ofSize:14),.foregroundColor:NSColor.systemOrange])
            }
        }
'''+s[b:]
s=s.replace('        let text:String', '''        if let p=c.truthPoint {
            let x=CGFloat(p.x)*bounds.width,y=CGFloat(p.y)*bounds.height
            NSColor.systemCyan.setStroke();let cross=NSBezierPath();cross.lineWidth=3
            cross.move(to:NSPoint(x:x-10,y:y));cross.line(to:NSPoint(x:x+10,y:y));cross.move(to:NSPoint(x:x,y:y-10));cross.line(to:NSPoint(x:x,y:y+10));cross.stroke()
        }
        let text:String''')
s=s.replace('else if !c.running {text="Пробел — распознавание · новая модель со светом"}', '''else if c.manualControl != nil {text="Мышью отметь, где был нераспознанный тап"}
        else if !c.running {text="Пробел — играть · каждый тап сохраняется"}''')
s=s.replace('text="Клетка \\(Int(p.x)+1), \\(Int(p.y)+1) · области: 2×2 синий · 3×3 фиолетовый · 4×4 оранжевый"', 'text="Клетка \\(Int(p.x)+1), \\(Int(p.y)+1) · мышью можно отметить, где был тап"')
s=s.replace('        let attrs:[NSAttributedString.Key:Any]', '''        let hint="Мышь — правильная точка · N — новые условия · 3 — варианты · → — не распознал · ⌘Q — сохранить и выйти"
        (hint as NSString).draw(at:NSPoint(x:18,y:bounds.height-20),withAttributes:[.font:NSFont.systemFont(ofSize:11),.foregroundColor:NSColor.lightGray])
        let attrs:[NSAttributedString.Key:Any]''')
s=s.replace('var error:String?', '''var error:String?
    var recorder:PlayRecorder?,saveFailure:String?,saveStatus:PassThroughLabel!
    var pendingID:String?,latestID:String?,pendingOnsets:[Double]=[]
    var candidates:[SIMD2<Double>]=[],truthPoint:SIMD2<Double>?,showCandidates=false
    var manualControl:Double?
    var answers=[String:(col:Int,row:Int,candidates:[[Int]])]()
    var annotations=[String:(col:Int,row:Int)]()''')
s=s.replace('view.controller=self;window.contentView=view', '''view.controller=self;window.contentView=view
        saveStatus=PassThroughLabel(labelWithString:"Запись начнётся по пробелу");saveStatus.font=NSFont.systemFont(ofSize:12);saveStatus.textColor = .white;saveStatus.drawsBackground=true;saveStatus.backgroundColor=NSColor.black.withAlphaComponent(.85);saveStatus.frame=NSRect(x:14,y:12,width:frame.width-28,height:24);saveStatus.autoresizingMask=[.width];view.addSubview(saveStatus)''')
s=s.replace('    func toggle() {', '''    func log(_ name:String,_ extra:[String:Any]=[:]) {do {try recorder?.event(name,extra:extra)}catch{storageFailed(error)}}
    func storageFailed(_ e:Error){saveFailure=e.localizedDescription;error="Ошибка сохранения: \\(e.localizedDescription)";running=false;ready=false;pending=nil;view?.needsDisplay=true}
    func status(){
        guard let r=recorder else{return}
        var text="Сохранено: \\(r.predictionCount) тапов · отмечено \\(r.labelCount) · условия \\(r.block)"
        if !running{text += " · ПАУЗА: датчики сохраняются"}
        let count=annotations.count
        if count>0 {
            var exact=0,near=0,among=0
            for (id,label) in annotations {if let a=answers[id]{if a.col==label.col && a.row==label.row{exact+=1};if max(abs(a.col-label.col),abs(a.row-label.row))<=1{near+=1};if a.candidates.contains(where:{$0==[label.col,label.row]}){among+=1}}}
            text += " · до отметки: точно \\(exact)/\\(count), рядом \\(near)/\\(count), из3 \\(among)/\\(count)"
        }
        saveStatus.stringValue=saveFailure.map{"ОШИБКА ЗАПИСИ: \\($0)"} ?? text
    }
    func mark(_ point:NSPoint){
        guard let r=recorder,saveFailure==nil,pending==nil,!predicting else{return}
        let x=min(0.999999,max(0,Double(point.x/view.bounds.width))),y=min(0.999999,max(0,Double(point.y/view.bounds.height)))
        do {
            if let control=manualControl {try r.manualLabel(x:x,y:y,control:control,start:control-3);manualControl=nil;armedAfter=MachClock.now()+.6}
            else if let id=latestID,let a=answers[id] {try r.label(id:id,x:x,y:y,pixel:[Double(point.x),Double(point.y)],base:[a.col,a.row],candidates:a.candidates);annotations[id]=(Int(x*20),Int(y*13))}
            else{return}
            truthPoint=SIMD2(x,y);error=nil;status();view.needsDisplay=true
        }catch{storageFailed(error)}
    }
    func missedTap(){
        guard recorder != nil,saveFailure==nil,pending==nil,!predicting else{return};manualControl=MachClock.now();ready=false;latestID=nil;log("manual_label_requested",["interval_start":manualControl!-3,"interval_end":manualControl!]);view.needsDisplay=true
    }
    func newConditions(){
        guard let r=recorder,saveFailure==nil else{return}
        do {try r.newBlock(lid:reader.lidAngle);prediction=nil;latestID=nil;truthPoint=nil;candidates=[];armedAfter=MachClock.now()+.6;ready=false;status();view.needsDisplay=true}catch{storageFailed(error)}
    }
    func toggle() {''')
s=s.replace('        guard model?.ready == true', '        guard saveFailure==nil else{return}\n        guard model?.ready == true')
s=s.replace('error=nil;running=true;ready=false;', '''do {if recorder==nil{recorder=try PlayRecorder()}}catch{storageFailed(error);return}
        error=nil;running=true;log("resume");status();ready=false;''')
s=s.replace('func pause(){running=false;ready=false;pending=nil;view?.needsDisplay=true}', '''func pause(){
        if running{log("pause",["pending_id":pendingID as Any? ?? NSNull()])}
        running=false;ready=false;pending=nil;pendingID=nil;manualControl=nil
        do{try recorder?.sync()}catch{storageFailed(error)};status();view?.needsDisplay=true
    }''')
s=s.replace('func applicationWillTerminate(_ notification:Notification){model?.stop()}', 'func applicationWillTerminate(_ notification:Notification){do{try recorder?.finish()}catch{storageFailed(error)};model?.stop()}')
s=s.replace('let now=MachClock.now(),batch=inbox.drain().sorted{$0.t<$1.t};ring.append', '''let now=MachClock.now(),batch=inbox.drain().sorted{$0.t<$1.t}
        if saveFailure==nil {do{try recorder?.samples(batch)}catch{storageFailed(error)}}
        status();ring.append''')
s=s.replace('if running && ready && pending==nil && s.t-lastInput>0.45 && hit {\n                pending=s.t;pendingLid=reader.lidAngle;ready=false;view.needsDisplay=true\n            }', '''if hit {
                if running && ready && pending==nil && manualControl==nil && s.t-lastInput>0.45 {
                    pending=s.t;pendingID=UUID().uuidString;pendingOnsets=[s.t];latestID=nil;truthPoint=nil;pendingLid=reader.lidAngle;ready=false
                    log("tap_trigger",["prediction_id":pendingID!,"sample_t":s.t,"lid_angle":pendingLid as Any? ?? NSNull()]);view.needsDisplay=true
                }else{
                    if let t=pending,s.t>=t && s.t<=t+.4{pendingOnsets.append(s.t)}
                    log("onset_not_assigned",["sample_t":s.t,"pending_id":pendingID as Any? ?? NSNull(),"reason":!running ? "paused" : (s.t-lastInput<=.45 ? "control_input":"busy_or_settling")])
                }
            }''')
s=s.replace('let samples=ring.filter{$0.t>=t-0.9 && $0.t<=t+0.4};predicting=true', 'let samples=ring.filter{$0.t>=t-0.9 && $0.t<=t+0.4};let id=pendingID!;let angle=pendingLid;let onsets=pendingOnsets;predicting=true')
s=s.replace('model!.predict(samples:samples,trigger:t,lid:pendingLid)', 'model!.predict(samples:samples,trigger:t,lid:angle,requestID:id)')
s=s.replace('guard self.running,self.pending==t else{return}\n                switch result {\n                case .success(let p):self.prediction=p;self.error=nil\n                case .failure(let e):self.error="Не удалось распознать: \\(e.localizedDescription)"\n                }', '''switch result {
                case .success(let result):
                    do {
                        try self.recorder?.prediction(id:id,trigger:t,lid:angle,samples:samples,details:result.details,onsets:onsets)
                        let raw=result.details["candidates"] as? [[String:Any]] ?? []
                        let cs=raw.compactMap{a -> [Int]? in guard let col=a["col"] as? Int,let row=a["row"] as? Int else{return nil};return [col,row]}
                        self.answers[id]=(Int(result.point.x),Int(result.point.y),cs)
                        if self.running,self.pending==t {self.prediction=result.point;self.latestID=id;self.candidates=cs.map{SIMD2(Double($0[0])+.5,Double($0[1])+.5)};self.error=nil}
                        self.status()
                    }catch{self.storageFailed(error)}
                case .failure(let e):self.log("prediction_failed",["prediction_id":id,"trigger_t":t,"error":e.localizedDescription]);self.error="Не удалось распознать · сигналы сохранены"
                }''')
s=s.replace('self.pending=nil;self.armedAfter', 'if self.pending==t{self.pending=nil;self.pendingID=nil};self.armedAfter')
s=s.replace('if pending==nil && !predicting && !ready', 'if pending==nil && !predicting && manualControl==nil && !ready')
p.write_text(s)
