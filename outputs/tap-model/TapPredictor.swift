import Cocoa
import OSLog

enum Log { static let sensor=Logger(subsystem:"local.tappredictor",category:"sensor") }
final class SampleInbox {
    let lock=NSLock(); var buffer:[MotionSample]=[]
    func add(_ s: MotionSample) { lock.withLock{buffer.append(s)} }
    func drain()->[MotionSample] { lock.withLock{let b=buffer;buffer.removeAll(keepingCapacity:true);return b} }
}
final class PredictorWindow: NSWindow {
    override var canBecomeKey: Bool {true}
    override var canBecomeMain: Bool {true}
}
final class PredictionView: NSView {
    weak var controller: Predictor?
    override var isFlipped: Bool {true}
    override var acceptsFirstResponder: Bool {true}
    override func keyDown(with event: NSEvent) {
        guard !event.isARepeat else {return}
        if event.modifierFlags.contains(.command),[12,13].contains(event.keyCode) { NSApp.terminate(nil);return }
        if event.keyCode==49 { controller?.toggle() }
        else if event.keyCode==53 {controller?.pause()}
        else {super.keyDown(with:event)}
    }
    override func draw(_ dirtyRect:NSRect) {
        NSColor(calibratedWhite:0.045,alpha:1).setFill();bounds.fill()
        guard let c=controller else{return}
        let w=bounds.width/20,h=bounds.height/13
        for row in 0..<13 {for col in 0..<20 {
            let r=NSRect(x:CGFloat(col)*w,y:CGFloat(row)*h,width:w,height:h).insetBy(dx:1,dy:1)
            let selected=c.prediction.map{Int(floor($0.x))==col && Int(floor($0.y))==row} ?? false
            (selected ? NSColor.systemGreen : NSColor(calibratedWhite:0.13,alpha:1)).setFill()
            NSBezierPath(roundedRect:r,xRadius:5,yRadius:5).fill()
            (selected ? NSColor.black : NSColor(calibratedWhite:0.36,alpha:1)).setFill()
            NSBezierPath(ovalIn:NSRect(x:r.midX-2.5,y:r.midY-2.5,width:5,height:5)).fill()
        }}
        if c.prediction != nil {
            for region in c.model?.lastRegions ?? [] {
                let col=region.zone % region.nx,row=region.zone/region.nx
                let rect=NSRect(x:CGFloat(col)*bounds.width/CGFloat(region.nx),y:CGFloat(row)*bounds.height/CGFloat(region.ny),width:bounds.width/CGFloat(region.nx),height:bounds.height/CGFloat(region.ny)).insetBy(dx:2,dy:2)
                let color: NSColor=region.nx==2 ? .systemBlue : (region.nx==3 ? .systemPurple : .systemOrange)
                color.withAlphaComponent(0.13).setFill();rect.fill()
                color.withAlphaComponent(0.9).setStroke();let outline=NSBezierPath(rect:rect);outline.lineWidth=3;outline.stroke()
            }
        }
        if let p=c.prediction {
            let selectedRect=NSRect(x:floor(CGFloat(p.x))*w,y:floor(CGFloat(p.y))*h,width:w,height:h).insetBy(dx:1,dy:1)
            NSColor.systemGreen.setFill();NSBezierPath(roundedRect:selectedRect,xRadius:5,yRadius:5).fill()
            let x=CGFloat(p.x)*w,y=CGFloat(p.y)*h
            NSColor.white.setStroke();let cross=NSBezierPath();cross.lineWidth=2
            cross.move(to:NSPoint(x:x-9,y:y));cross.line(to:NSPoint(x:x+9,y:y))
            cross.move(to:NSPoint(x:x,y:y-9));cross.line(to:NSPoint(x:x,y:y+9));cross.stroke()
        }
        let text:String
        if let error=c.error {text=error}
        else if !c.running {text="Пробел — распознавание"}
        else if c.pending != nil {text="Тап обнаружен · распознаю…"}
        else if !c.ready {text="Подожди · датчик успокаивается"}
        else if let p=c.prediction {text="Клетка \(Int(p.x)+1), \(Int(p.y)+1) · области: 2×2 синий · 3×3 фиолетовый · 4×4 оранжевый"}
        else {text="Коснись любой точки один раз"}
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
    var detector=TapDetector(),model:SpatialModel?
    var window:NSWindow!,view:PredictionView!,timer:Timer?
    var running=false,ready=false
    var pending:Double?,pendingLid:Int?,prediction:SIMD2<Double>?
    var ring:[MotionSample]=[]
    var error:String?
    var lastGyro=0.0,lastAccel=0.0,lastInput=0.0,armedAfter=0.0
    func applicationDidFinishLaunching(_ notification:Notification) {
        NSApp.setActivationPolicy(.regular)
        let frame=NSScreen.main!.frame
        window=PredictorWindow(contentRect:frame,styleMask:[.borderless],backing:.buffered,defer:false);window.title="Tap Predictor — области и клетка";window.delegate=self
        view=PredictionView(frame:NSRect(origin:.zero,size:frame.size));view.controller=self;window.contentView=view
        let menu=NSMenu(),sub=NSMenu(),item=NSMenuItem();sub.addItem(withTitle:"Выйти",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q");item.submenu=sub;menu.addItem(item);NSApp.mainMenu=menu
        NSApp.presentationOptions=[.hideDock,.hideMenuBar]
        do {model=try SpatialModel(url:Bundle.main.resourceURL!.appendingPathComponent("model.json"))}
        catch {self.error="Не удалось загрузить модель: \(error)"}
        window.makeKeyAndOrderFront(nil);window.makeFirstResponder(view);NSApp.activate(ignoringOtherApps:true)
        reader.onSample = {[weak self] k,t,v in self?.inbox.add(MotionSample(kind:k == .gyro ? 1:0,t:t,v:v))};reader.start()
        timer=Timer.scheduledTimer(withTimeInterval:0.025,repeats:true){[weak self] _ in self?.tick()}
        NSEvent.addLocalMonitorForEvents(matching:[.keyDown,.leftMouseDown,.rightMouseDown,.scrollWheel]){[weak self] event in self?.lastInput=MachClock.now();return event}
    }
    func toggle() {
        if running {pause();return}
        guard model != nil else{return}
        guard MachClock.now()-lastGyro<0.5,MachClock.now()-lastAccel<0.5 else {error="Нет сигнала датчика · подожди и нажми пробел";view.needsDisplay=true;return}
        error=nil;running=true;ready=false;pending=nil;prediction=nil;armedAfter=MachClock.now()+0.25;view.needsDisplay=true
    }
    func pause(){running=false;ready=false;pending=nil;view?.needsDisplay=true}
    func applicationDidResignActive(_ notification:Notification){pause()}
    func windowWillClose(_ notification:Notification){NSApp.terminate(nil)}
    func tick(){
        let now=MachClock.now(),batch=inbox.drain();ring.append(contentsOf:batch);ring.removeAll{$0.t<now-3}
        for s in batch {
            if s.kind==1 {lastGyro=s.t}else{lastAccel=s.t}
            let hit=detector.process(kind:s.kind,t:s.t,v:s.v,learnNoise:pending==nil)
            if running && ready && pending==nil && s.t-lastInput>0.45 && hit {
                pending=s.t;pendingLid=reader.lidAngle;ready=false;view.needsDisplay=true
            }
        }
        guard running else{return}
        if now-lastGyro>0.7 || now-lastAccel>0.7 {pause();error="Датчик отключился · пробел — повторить";view.needsDisplay=true;return}
        if let t=pending,now-t>0.46 {
            let samples=ring.filter{$0.t>=t-0.15 && $0.t<=t+0.4}
            do {
                guard let g=samples.last(where:{$0.kind==1}),let a=samples.last(where:{$0.kind==0}),g.t>=t+0.38,a.t>=t+0.38 else {throw ModelError.invalid("Неполная запись")}
                let f=try motionFeatures(samples:samples,trigger:t,lid:pendingLid)
                let p=try model!.predict(features:f)
                prediction=SIMD2(min(19.999,max(0,p.x)),min(12.999,max(0,p.y)));error=nil
            } catch {self.error="Не удалось распознать · повтори лёгкий тап"}
            pending=nil;armedAfter=now+0.6;view.needsDisplay=true
        }
        if pending==nil && !ready && now>=armedAfter && now-lastInput>0.45 && detector.quietDuration(at:now)>=0.22 {ready=true;view.needsDisplay=true}
    }
}
let app=NSApplication.shared,controller=Predictor();app.delegate=controller;app.run()
