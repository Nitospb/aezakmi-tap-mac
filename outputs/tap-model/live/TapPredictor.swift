import Cocoa
import OSLog

enum Log { static let sensor=Logger(subsystem:"local.tappredictor",category:"sensor") }
final class SampleInbox {
    let lock=NSLock(); var buffer:[MotionSample]=[]
    func add(_ s: MotionSample) { lock.withLock{buffer.append(s)} }
    func add(_ batch:[MotionSample]) {lock.withLock{buffer.append(contentsOf:batch)}}
    func drain()->[MotionSample] { lock.withLock{let b=buffer;buffer.removeAll(keepingCapacity:true);return b} }
}
final class PredictorWindow: NSWindow {
    override var canBecomeKey: Bool {true}
    override var canBecomeMain: Bool {true}
}
final class PassThroughLabel:NSTextField {override func hitTest(_ point:NSPoint)->NSView?{nil}}
final class PredictionView: NSView {
    weak var controller: Predictor?
    override var isFlipped: Bool {true}
    override var acceptsFirstResponder: Bool {true}
    override func keyDown(with event: NSEvent) {
        guard !event.isARepeat else {return}
        if event.modifierFlags.contains(.command),[12,13].contains(event.keyCode) { NSApp.terminate(nil);return }
        if event.keyCode==49 { controller?.toggle() }
        else if event.keyCode==53 {controller?.pause()}
        else if event.keyCode==124 {controller?.nextTap()}
        else if event.keyCode==45 {controller?.newConditions()}
        else if event.keyCode==20 {controller?.showCandidates.toggle();needsDisplay=true}
        else if event.keyCode==18 {controller?.retrain()}
        else if event.keyCode==19 {controller?.newConditions()}
        else if event.keyCode==29 {controller?.resetModel()}
        else if event.keyCode==21 {controller?.missedTap()}
        else {super.keyDown(with:event)}
    }
    override func mouseDown(with event:NSEvent){controller?.mark(convert(event.locationInWindow,from:nil))}
    override func draw(_ dirtyRect:NSRect) {
        NSColor(calibratedWhite:0.045,alpha:1).setFill();bounds.fill()
        guard let c=controller else{return}
        let w=bounds.width/20,h=bounds.height/13
        for row in 0..<13 {for col in 0..<20 {
            let r=NSRect(x:CGFloat(col)*w,y:CGFloat(row)*h,width:w,height:h).insetBy(dx:1,dy:1)
            let actual=c.truthPoint.map{Int($0.x*20)==col && Int($0.y*13)==row} ?? false
            let selected=c.prediction.map{Int(floor($0.x))==col && Int(floor($0.y))==row} ?? false
            (c.confirmed && actual ? (c.confirmedCorrect ? NSColor.systemGreen.withAlphaComponent(0.4) : NSColor.systemBlue.withAlphaComponent(0.75)) : (selected ? NSColor.systemOrange.withAlphaComponent(c.confirmed ? 0.2 : 0.75) : NSColor(calibratedWhite:0.13,alpha:1))).setFill()
            NSBezierPath(roundedRect:r,xRadius:5,yRadius:5).fill()
            (selected ? NSColor.black : NSColor(calibratedWhite:0.36,alpha:1)).setFill()
            NSBezierPath(ovalIn:NSRect(x:r.midX-2.5,y:r.midY-2.5,width:5,height:5)).fill()
        }}
        if c.showCandidates {
            for (i,p) in c.candidates.enumerated() where i>0 {
                let r=NSRect(x:floor(CGFloat(p.x))*w,y:floor(CGFloat(p.y))*h,width:w,height:h).insetBy(dx:3,dy:3)
                NSColor.systemOrange.setStroke();let path=NSBezierPath(roundedRect:r,xRadius:5,yRadius:5);path.lineWidth=3;path.stroke()
                ("\(i+1)" as NSString).draw(at:NSPoint(x:r.minX+6,y:r.minY+5),withAttributes:[.font:NSFont.systemFont(ofSize:14),.foregroundColor:NSColor.systemOrange])
            }
        }
        if let p=c.prediction {
            let selectedRect=NSRect(x:floor(CGFloat(p.x))*w,y:floor(CGFloat(p.y))*h,width:w,height:h).insetBy(dx:1,dy:1)
            (c.confirmed && c.confirmedCorrect ? NSColor.systemGreen : NSColor.systemOrange).setStroke();let outline=NSBezierPath(roundedRect:selectedRect,xRadius:5,yRadius:5);outline.lineWidth=2;outline.stroke()
            let x=CGFloat(p.x)*w,y=CGFloat(p.y)*h
            (c.confirmed && c.confirmedCorrect ? NSColor.systemGreen : NSColor.white).setStroke();let cross=NSBezierPath();cross.lineWidth=3
            cross.move(to:NSPoint(x:x-9,y:y));cross.line(to:NSPoint(x:x+9,y:y))
            cross.move(to:NSPoint(x:x,y:y-9));cross.line(to:NSPoint(x:x,y:y+9));cross.stroke()
        }
        if let p=c.truthPoint,!c.confirmedCorrect {
            let x=CGFloat(p.x)*bounds.width,y=CGFloat(p.y)*bounds.height
            NSColor.systemCyan.setStroke();let cross=NSBezierPath();cross.lineWidth=3
            cross.move(to:NSPoint(x:x-10,y:y));cross.line(to:NSPoint(x:x+10,y:y));cross.move(to:NSPoint(x:x,y:y-10));cross.line(to:NSPoint(x:x,y:y+10));cross.stroke()
        }
        let text:String
        if let error=c.error {text=error}
        else if c.training {text="Дообучение · датчики продолжают сохраняться"}
        else if let notice=c.notice,MachClock.now()<c.noticeUntil {text=notice}
        else if c.locked,c.manualID != nil {text=c.confirmed ? "Пропущенный тап сохранён · 1 — дообучить · → — дальше" : "Пропущенный тап · укажи клетку мышью"}
        else if c.locked,c.prediction != nil {
            text=c.confirmed ? (c.confirmedCorrect ? "✓ Верно · сохранено · → следующий тап" : "Правильная клетка синяя · сохранено · → следующий тап") : "Предсказание заморожено · отметь мышью · → пропустить срабатывание"
        }
        else if !c.running {text="Пробел — играть · каждый тап сохраняется"}
        else if c.pending != nil {text="Тап обнаружен · распознаю…"}
        else if !c.ready {text="Подожди · датчик успокаивается"}
        else if c.prediction != nil {text="→ — следующий тап"}
        else {text="Коснись любой точки один раз"}
        let hint="Мышь — правильная точка · 1 — дообучить · 2/N — условия · 0 — базовая модель · 3 — варианты · 4 — не распознал · → — следующий тап · ⌘Q — выход"
        (hint as NSString).draw(at:NSPoint(x:18,y:bounds.height-20),withAttributes:[.font:NSFont.systemFont(ofSize:11),.foregroundColor:NSColor.lightGray])
        let attrs:[NSAttributedString.Key:Any]=[.font:NSFont.systemFont(ofSize:c.running ? 16:24,weight:.semibold),.foregroundColor:NSColor.white]
        let size=(text as NSString).size(withAttributes:attrs)
        let nearBottom=c.prediction.map{$0.y>=11} ?? false
        let y:CGFloat = !c.running ? bounds.midY-30 : (nearBottom ? 10:bounds.height-62)
        let box=NSRect(x:bounds.midX-size.width/2-22,y:y,width:size.width+44,height:58)
        NSColor.black.withAlphaComponent(0.9).setFill();NSBezierPath(roundedRect:box,xRadius:10,yRadius:10).fill()
        (text as NSString).draw(at:NSPoint(x:box.midX-size.width/2,y:box.midY-size.height/2),withAttributes:attrs)
    }
}
final class Predictor:NSObject,NSApplicationDelegate,NSWindowDelegate {
    let reader=IMUReader(),inbox=SampleInbox()
    var detector=TapDetector(),model:LiveModel?
    var window:NSWindow!,view:PredictionView!,timer:Timer?
    var running=false,ready=false
    var pending:Double?,pendingLid:Double?,prediction:SIMD2<Double>?
    var ring:[MotionSample]=[]
    var error:String?
    var recorder:PlayRecorder?,saveFailure:String?,saveStatus:PassThroughLabel!
    var training=false,resumeAfterTrain=false,notice:String?,noticeUntil=0.0,pendingBlock=1
    var pendingID:String?,latestID:String?,pendingOnsets:[Double]=[]
    var candidates:[SIMD2<Double>]=[],truthPoint:SIMD2<Double>?,showCandidates=false
    var manualControl:Double?
    var manualID:String?,manualSamples:[MotionSample]=[],manualEarliest=0.0,lastRelease=0.0
    var locked=false,confirmed=false
    var confirmedCorrect:Bool {
        guard confirmed,let p=prediction,let t=truthPoint else{return false}
        return Int(p.x)==Int(t.x*20) && Int(p.y)==Int(t.y*13)
    }
    var answers=[String:(col:Int,row:Int,candidates:[[Int]],version:Int)]()
    var annotations=[String:(col:Int,row:Int)]()
    var predicting=false,lastLight=0.0,lastLid=0.0
    var lastGyro=0.0,lastAccel=0.0,lastInput=0.0,armedAfter=0.0
    func applicationDidFinishLaunching(_ notification:Notification) {
        NSApp.setActivationPolicy(.regular)
        let frame=NSScreen.main!.frame
        window=PredictorWindow(contentRect:frame,styleMask:[.borderless],backing:.buffered,defer:false);window.title="Tap Predictor — области и клетка";window.delegate=self
        view=PredictionView(frame:NSRect(origin:.zero,size:frame.size));view.controller=self;window.contentView=view
        saveStatus=PassThroughLabel(labelWithString:"Запись начнётся по пробелу");saveStatus.font=NSFont.systemFont(ofSize:12);saveStatus.textColor = .white;saveStatus.drawsBackground=true;saveStatus.backgroundColor=NSColor.black.withAlphaComponent(0.85);saveStatus.frame=NSRect(x:14,y:12,width:frame.width-28,height:24);saveStatus.autoresizingMask=[.width];view.addSubview(saveStatus)
        let menu=NSMenu(),sub=NSMenu(),item=NSMenuItem();sub.addItem(withTitle:"Выйти",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q");item.submenu=sub;menu.addItem(item);NSApp.mainMenu=menu
        NSApp.presentationOptions=[.hideDock,.hideMenuBar]
        model=LiveModel();error="Модель загружается…"
        model?.onStatus = {[weak self] problem in self?.error=problem.map{"Ошибка модели: \($0)"};self?.view.needsDisplay=true};model?.start()
        window.makeKeyAndOrderFront(nil);window.makeFirstResponder(view);NSApp.activate(ignoringOtherApps:true)
        reader.onSample = {[weak self] k,t,v in self?.inbox.add(MotionSample(kind:k == .gyro ? 1:0,t:t,v:v))}
        reader.onLidSample = {[weak self] t,a in guard let self else{return};self.inbox.add(MotionSample(kind:2,t:t,v:SIMD3(a,Double(self.reader.lidReportID),self.reader.lidResolution)))}
        reader.onLightSample = {[weak self] t,l in guard let self else{return};let c=self.reader.lightChannels;self.inbox.add([MotionSample(kind:3,t:t,v:SIMD3(l,0,0)),MotionSample(kind:4,t:t,v:SIMD3(Double(c.x),Double(c.y),Double(c.z))),MotionSample(kind:5,t:t,v:SIMD3(Double(c.w),0,0))])}
        reader.lidPollHz=200;reader.start()
        timer=Timer.scheduledTimer(withTimeInterval:0.025,repeats:true){[weak self] _ in self?.tick()}
        NSEvent.addLocalMonitorForEvents(matching:[.keyDown,.leftMouseDown,.rightMouseDown,.scrollWheel]){[weak self] event in self?.lastInput=MachClock.now();return event}
    }
    func log(_ name:String,_ extra:[String:Any]=[:]) {do {try recorder?.event(name,extra:extra)}catch{storageFailed(error)}}
    func storageFailed(_ e:Error){saveFailure=e.localizedDescription;error="Ошибка сохранения: \(e.localizedDescription)";running=false;ready=false;pending=nil;view?.needsDisplay=true}
    func status(){
        guard let r=recorder else{return}
        var text="v\(model?.version ?? 0) · Сохранено: \(r.predictionCount+r.manualCount) тапов · отметок \(r.labelCount) · условия \(r.block)"
        if !running{text += " · ПАУЗА: датчики сохраняются"}
        if locked{text += confirmed ? " · подтверждено →" : " · ждёт отметку мышью"}
        let count=annotations.count
        if count>0 {
            let versions=Array(Set(answers.values.map{$0.version})).sorted().suffix(2)
            for version in versions {
                var exact=0,near=0,n=0
                for (id,label) in annotations {if let a=answers[id],a.version==version{n+=1;if a.col==label.col && a.row==label.row{exact+=1};if max(abs(a.col-label.col),abs(a.row-label.row))<=1{near+=1}}}
                if n>0{text += " · v\(version): точно \(exact)/\(n), рядом \(near)/\(n)"}
            }
        }
        saveStatus.stringValue=saveFailure.map{"ОШИБКА ЗАПИСИ: \($0)"} ?? text
    }
    func mark(_ point:NSPoint){
        guard let r=recorder,saveFailure==nil,pending==nil,!predicting,!training,locked else{return}
        let rawX=min(0.999999,max(0,Double(point.x/view.bounds.width))),rawY=min(0.999999,max(0,Double(point.y/view.bounds.height)))
        let col=Int(rawX*20),row=Int(rawY*13),x=(Double(col)+0.5)/20,y=(Double(row)+0.5)/13
        let centerPixel=[x*Double(view.bounds.width),y*Double(view.bounds.height)],rawPixel=[Double(point.x),Double(point.y)]
        do {
            if let control=manualControl,let id=manualID {try r.manualLabel(x:x,y:y,control:control,start:manualSamples.first?.t ?? control-8,id:id,samples:manualSamples,searchStart:manualEarliest,rawClick:rawPixel)}
            else if let id=latestID,let a=answers[id] {try r.label(id:id,x:x,y:y,pixel:centerPixel,base:[a.col,a.row],candidates:a.candidates,rawClick:rawPixel);annotations[id]=(Int(x*20),Int(y*13))}
            else{return}
            truthPoint=SIMD2(x,y);confirmed=true;ready=false;error=nil;status();view.needsDisplay=true
        }catch{storageFailed(error)}
    }
    func missedTap(){
        guard recorder != nil,saveFailure==nil,!training,!predicting,pending==nil else{return}
        guard !locked else{flash("Сначала →, чтобы завершить текущую отметку");return}
        let control=MachClock.now(),end=control-0.08
        manualSamples=ring.filter{$0.t>=max(lastRelease-0.9,control-8) && $0.t<=end}
        guard manualSamples.count>200 else{flash("Сигнала пока мало · подожди после тапа");return}
        manualControl=control;manualEarliest=lastRelease+0.45;manualID=UUID().uuidString;locked=true;confirmed=false;ready=false
        prediction=nil;latestID=nil;candidates=[];truthPoint=nil;notice=nil;error=nil
        log("missed_tap_requested",["manual_id":manualID!,"control_t":control,"search_start":manualEarliest,"window_end":end]);status();view.needsDisplay=true
    }
    func nextTap(){
        guard recorder != nil,saveFailure==nil,!training else{return}
        guard pending==nil,!predicting else{flash("Дождись предсказания");return}
        log("feedback_next",["prediction_id":latestID as Any? ?? NSNull(),"label_confirmed":confirmed,"skipped_unlabelled":locked && !confirmed])
        locked=false;confirmed=false;prediction=nil;latestID=nil;truthPoint=nil;candidates=[];manualControl=nil;manualID=nil;manualSamples=[];lastRelease=MachClock.now();notice=nil;error=nil
        detector=TapDetector();ready=false;armedAfter=MachClock.now()+0.65;status();view.needsDisplay=true
    }
    func newConditions(){
        guard let r=recorder,saveFailure==nil else{return}
        do {try r.newBlock(lid:reader.lidAngle);armedAfter=MachClock.now()+0.6;ready=false;status();view.needsDisplay=true}catch{storageFailed(error)}
    }
    func toggle() {
        if training{flash("Дообучение ещё идёт");return}
        if running {pause();return}
        guard saveFailure==nil else{return}
        guard model?.ready == true else{error="Модель загружается · подожди";view.needsDisplay=true;return}
        guard !predicting else{error="Дождись завершения распознавания";view.needsDisplay=true;return}
        guard MachClock.now()-lastLight<1,MachClock.now()-lastLid<0.5 else{error="Жду свет и угол крышки";view.needsDisplay=true;return}
        guard MachClock.now()-lastGyro<0.5,MachClock.now()-lastAccel<0.5 else {error="Нет сигнала датчика · подожди и нажми пробел";view.needsDisplay=true;return}
        do {if recorder==nil{recorder=try PlayRecorder(activeModel:model?.modelPath.map{URL(fileURLWithPath:$0)})}}catch{storageFailed(error);return}
        error=nil;running=true;log("resume");status();ready=false;pending=nil;if !locked{prediction=nil;lastRelease=MachClock.now()};armedAfter=MachClock.now()+0.25;view.needsDisplay=true
    }
    func pause(){
        if running{log("pause",["pending_id":pendingID as Any? ?? NSNull()])}
        if pending != nil{locked=false;confirmed=false;prediction=nil;latestID=nil}
        running=false;ready=false;pending=nil;pendingID=nil;if manualID==nil{manualControl=nil};resumeAfterTrain=false
        do{try recorder?.sync()}catch{storageFailed(error)};status();view?.needsDisplay=true
    }
    func applicationDidResignActive(_ notification:Notification){pause()}
    func applicationWillTerminate(_ notification:Notification){do{try recorder?.finish()}catch{storageFailed(error)};model?.stop()}
    func windowWillClose(_ notification:Notification){NSApp.terminate(nil)}
    func tick(){
        let now=MachClock.now(),batch=inbox.drain().sorted{$0.t<$1.t}
        if saveFailure==nil {do{try recorder?.samples(batch)}catch{storageFailed(error)}}
        status();ring.append(contentsOf:batch);ring.removeAll{$0.t<now-9}
        for s in batch {
            if s.kind==1 {lastGyro=s.t}else if s.kind==0{lastAccel=s.t}else if s.kind==2{lastLid=s.t}else if s.kind==3{lastLight=s.t}
            guard s.kind<2 else{continue}
            let hit=detector.process(kind:s.kind,t:s.t,v:s.v,learnNoise:pending==nil)
            if hit {
                if running && ready && !locked && pending==nil && manualControl==nil && s.t-lastInput>0.45 {
                    locked=true;confirmed=false;pending=s.t;pendingID=UUID().uuidString;pendingBlock=recorder?.block ?? 1;notice=nil;pendingOnsets=[s.t];latestID=nil;truthPoint=nil;pendingLid=reader.lidAngle;ready=false
                    log("tap_trigger",["prediction_id":pendingID!,"sample_t":s.t,"lid_angle":pendingLid as Any? ?? NSNull()]);view.needsDisplay=true
                }else{
                    if let t=pending,s.t>=t && s.t<=t+0.4{pendingOnsets.append(s.t)}
                    log("onset_not_assigned",["sample_t":s.t,"pending_id":pendingID as Any? ?? NSNull(),"reason":!running ? "paused" : (locked ? "feedback_locked" : (s.t-lastInput<=0.45 ? "control_input":"busy_or_settling"))])
                }
            }
        }
        if training{view.needsDisplay=true;return}
        guard running else{return}
        if now-lastLight>1 || now-lastLid>0.7 {pause();error="Прервался свет или угол · пробел — повторить";view.needsDisplay=true;return}
        if now-lastGyro>0.7 || now-lastAccel>0.7 {pause();error="Датчик отключился · пробел — повторить";view.needsDisplay=true;return}
        if let t=pending,now-t>0.55 && !predicting {
            let samples=ring.filter{$0.t>=t-0.9 && $0.t<=t+0.4};let id=pendingID!;let angle=pendingLid;let onsets=pendingOnsets;let block=pendingBlock;predicting=true
            model!.predict(samples:samples,trigger:t,lid:angle,requestID:id){[weak self] result in
                guard let self else{return};self.predicting=false
                switch result {
                case .success(let result):
                    do {
                        try self.recorder?.prediction(id:id,trigger:t,lid:angle,samples:samples,details:result.details,onsets:onsets,captureBlock:block)
                        let raw=result.details["candidates"] as? [[String:Any]] ?? []
                        let cs=raw.compactMap{a -> [Int]? in guard let col=a["col"] as? Int,let row=a["row"] as? Int else{return nil};return [col,row]}
                        self.answers[id]=(Int(result.point.x),Int(result.point.y),cs,result.details["version"] as? Int ?? 0)
                        if self.running,self.pending==t {self.prediction=result.point;self.latestID=id;self.candidates=cs.map{SIMD2(Double($0[0])+0.5,Double($0[1])+0.5)};self.error=nil}
                        self.status()
                    }catch{self.storageFailed(error)}
                case .failure(let e):self.locked=false;self.confirmed=false;self.log("prediction_failed",["prediction_id":id,"trigger_t":t,"error":e.localizedDescription]);self.error="Не удалось распознать · сигналы сохранены"
                }
                if self.pending==t{self.pending=nil;self.pendingID=nil};self.armedAfter=MachClock.now()+0.6;self.view.needsDisplay=true
            }
        }
        if pending==nil && !predicting && !locked && manualControl==nil && !ready && now>=armedAfter && now-lastInput>0.45 && detector.quietDuration(at:now)>=0.22 {ready=true;view.needsDisplay=true}
    }
}
let app=NSApplication.shared,controller=Predictor();app.delegate=controller;app.run()
