import Cocoa
import OSLog

enum Log { static let sensor=Logger(subsystem:"local.sensorscope",category:"sensor") }
struct Reading {
    let kind:Int
    let t:Double
    let raw:SIMD3<Double>
    let dynamic:SIMD3<Double>
}
final class ReadingInbox {
    private let lock=NSLock();private var entries:[(Int,Double,SIMD3<Double>)]=[]
    func add(_ k:Int,_ t:Double,_ v:SIMD3<Double>){lock.withLock{entries.append((k,t,v))}}
    func drain()->[(Int,Double,SIMD3<Double>)]{lock.withLock{let e=entries;entries.removeAll(keepingCapacity:true);return e}}
}
final class ScopeView:NSView {
    weak var controller:ScopeController?
    override var isFlipped:Bool{true}
    override var acceptsFirstResponder:Bool{true}
    override func keyDown(with event:NSEvent){
        if event.keyCode==49 {controller?.togglePause()}
        else if event.keyCode==123 {controller?.previous()}
        else if event.keyCode==124 {controller?.next()}
        else {super.keyDown(with:event)}
    }
    func text(_ s:String,_ point:NSPoint,_ size:CGFloat,_ color:NSColor = .white,_ weight:NSFont.Weight = .regular){
        (s as NSString).draw(at:point,withAttributes:[.font:NSFont.systemFont(ofSize:size,weight:weight),.foregroundColor:color])
    }
    override func draw(_ dirtyRect:NSRect){
        NSColor(calibratedWhite:0.055,alpha:1).setFill();bounds.fill()
        guard let c=controller else{return}
        let kind=c.selected, channels=kind>=2 ? 1:3, now=c.pauseTime ?? MachClock.now()
        let samples=c.history.filter{$0.kind==kind && $0.t>=now-c.windowSeconds && $0.t<=now}
        let start:CGFloat=198, bottom:CGFloat=82, gap:CGFloat=13
        let height=max(80,(bounds.height-start-bottom-CGFloat(channels-1)*gap)/CGFloat(channels))
        for axis in 0..<channels {
            let box=NSRect(x:24,y:start+CGFloat(axis)*(height+gap),width:bounds.width-48,height:height)
            NSColor(calibratedWhite:0.105,alpha:1).setFill();NSBezierPath(roundedRect:box,xRadius:10,yRadius:10).fill()
            let color:NSColor=kind==3 ? .systemYellow : (kind==2 ? .systemOrange : [.systemRed,.systemGreen,.systemCyan][axis])
            let name=kind==3 ? "Освещённость" : (kind==2 ? "Угол крышки" : ["X","Y","Z"][axis])
            let unit=kind==0 ? "град/с" : (kind==1 ? "g":(kind==3 ? "лк":"°"))
            func value(_ s:Reading)->Double {c.useDynamic && kind != 2 ? s.dynamic[axis]:s.raw[axis]}
            let values=samples.map(value),latest=values.last
            let vlabel=latest.map{String(format:kind==2 ? "%.2f %@":"%+.5f %@",$0,unit)} ?? "Нет данных"
            text("\(name)     \(vlabel)",NSPoint(x:box.minX+14,y:box.minY+10),17,color,.semibold)
            let graph=NSRect(x:box.minX+70,y:box.minY+44,width:box.width-90,height:box.height-65)
            var lo=values.min() ?? -1,hi=values.max() ?? 1
            let minimumSpan=kind==2 ? max(c.reader.lidResolution*4,0.04) : (kind==0 ? 0.1 : (kind==3 ? 0.01 : 0.002))
            if hi-lo<minimumSpan {let mid=(lo+hi)/2;lo=mid-minimumSpan/2;hi=mid+minimumSpan/2}
            let margin=(hi-lo)*0.1;lo-=margin;hi+=margin
            NSColor(calibratedWhite:0.22,alpha:1).setStroke()
            for i in 0...4 {
                let yy=graph.minY+CGFloat(i)*graph.height/4
                let p=NSBezierPath();p.move(to:NSPoint(x:graph.minX,y:yy));p.line(to:NSPoint(x:graph.maxX,y:yy));p.stroke()
                let label=hi-Double(i)*(hi-lo)/4
                text(String(format:kind==2 ? "%.2f":"%.3f",label),NSPoint(x:box.minX+8,y:yy-6),10,.secondaryLabelColor)
            }
            for i in 0...8 {
                let xx=graph.minX+CGFloat(i)*graph.width/8
                let p=NSBezierPath();p.move(to:NSPoint(x:xx,y:graph.minY));p.line(to:NSPoint(x:xx,y:graph.maxY));p.stroke()
            }
            if !samples.isEmpty {
                color.setStroke();let p=NSBezierPath();p.lineWidth=1.3
                for (i,s) in samples.enumerated(){
                    let point=NSPoint(x:graph.maxX-CGFloat((now-s.t)/c.windowSeconds)*graph.width,y:graph.maxY-CGFloat((value(s)-lo)/(hi-lo))*graph.height)
                    if i==0{p.move(to:point)}else{p.line(to:point)}
                };p.stroke()
            }
            text("−\(Int(c.windowSeconds)) с",NSPoint(x:graph.minX,y:graph.maxY+4),10,.secondaryLabelColor)
            text(c.paused ? "стоп-кадр":"сейчас",NSPoint(x:graph.maxX-55,y:graph.maxY+4),10,.secondaryLabelColor)
            if kind>=2, let lo=values.min(),let hi=values.max() {
                text(String(format:"Размах %.4f %@",hi-lo,unit),NSPoint(x:box.maxX-210,y:box.minY+13),12,.secondaryLabelColor)
            } else if let peak=values.map({abs($0)}).max(){text(String(format:"|пик| %.5f %@",peak,unit),NSPoint(x:box.maxX-210,y:box.minY+13),12,.secondaryLabelColor)}
        }
        let note=kind==3 ? "Поток HID, запрос интервала 1 мс. Предел диапазона в люксах неизвестен; максимум выше — измеренный за сеанс." : (kind==2 ? String(format:"Отчёт №%d · шаг данных %.2f°; это не паспортная точность. Частота чтения ≠ частота обновления железа.",c.reader.lidReportID,c.reader.lidResolution) : (c.useDynamic ? "Быстрые изменения: плавный фон вычитается. Масштаб каждой оси автоматический; единицы указаны слева." : "Исходные отсчёты. Акселерометр также измеряет гравитацию (~1 g в покое). Масштаб каждой оси автоматический."))
        text(note,NSPoint(x:24,y:bounds.height-62),12,.secondaryLabelColor)
        let ch=c.reader.lightChannels
        let footer = kind==3 ? "Сырые каналы: \(ch.x) / \(ch.y) / \(ch.z) / \(ch.w) · колебания могут быть от тени руки, света экрана и мерцания ламп." : "← / → — следующий датчик · пробел — стоп-кадр · все четыре источника читаются одновременно"
        text(footer,NSPoint(x:24,y:bounds.height-40),12,.secondaryLabelColor)
        text("Based on LidTap by Yemcube · @yem.cube",NSPoint(x:24,y:bounds.height-22),10,.tertiaryLabelColor)
    }
}
final class ScopeController:NSObject,NSApplicationDelegate,NSWindowDelegate {
    let reader=IMUReader(),inbox=ReadingInbox()
    var window:NSWindow!,view:ScopeView!,timer:Timer?
    var picker:NSSegmentedControl!,mode:NSSegmentedControl!,ratePicker:NSPopUpButton!,pauseButton:NSButton!
    var title:NSTextField!,stats:NSTextField!,detail:NSTextField!
    var previousButton:NSButton!,nextButton:NSButton!,resetButton:NSButton!
    var lightMin=Double.infinity,lightMax=0.0,peakLightRate=0.0
    var lightFirstTime=0.0,lightLastTime=0.0,lightCount=0
    var history:[Reading]=[]
    var baseline=[SIMD3<Double>?](repeating:nil,count:4)
    var lastValues=[SIMD3<Double>?](repeating:nil,count:4)
    var arrival=[Double](repeating:0,count:4),counts=[Int](repeating:0,count:4),changes=[Int](repeating:0,count:4)
    var rates=[Double](repeating:0,count:4),changeRates=[Double](repeating:0,count:4)
    var selected=3,useDynamic=false,paused=false,pauseTime:Double?
    var windowSeconds=8.0,rateStart=0.0
    func label(_ value:String,_ size:CGFloat)->NSTextField {
        let l=NSTextField(labelWithString:value);l.font=NSFont.systemFont(ofSize:size,weight:size>=20 ? .semibold:.regular);l.textColor = .white;view.addSubview(l);return l
    }
    func applicationDidFinishLaunching(_ notification:Notification){
        NSApp.setActivationPolicy(.regular)
        let frame=NSScreen.main!.visibleFrame
        window=NSWindow(contentRect:frame,styleMask:[.titled,.closable,.miniaturizable,.resizable],backing:.buffered,defer:false)
        window.title="Sensor Scope — датчики MacBook";window.minSize=NSSize(width:1000,height:700);window.delegate=self
        view=ScopeView(frame:NSRect(origin:.zero,size:frame.size));view.controller=self;window.contentView=view
        title=label("Гироскоп · три оси",24)
        picker=NSSegmentedControl(labels:["1. Гироскоп","2. Акселерометр","3. Угол крышки","4. Освещённость"],trackingMode:.selectOne,target:self,action:#selector(selectSensor));picker.selectedSegment=selected;view.addSubview(picker)
        previousButton=NSButton(title:"←",target:self,action:#selector(previous));view.addSubview(previousButton)
        nextButton=NSButton(title:"→",target:self,action:#selector(next));view.addSubview(nextButton)
        mode=NSSegmentedControl(labels:["Исходный сигнал","Быстрые изменения"],trackingMode:.selectOne,target:self,action:#selector(changeMode));mode.selectedSegment=useDynamic ? 1:0;view.addSubview(mode)
        pauseButton=NSButton(title:"Стоп-кадр (пробел)",target:self,action:#selector(togglePause));view.addSubview(pauseButton)
        ratePicker=NSPopUpButton(frame:.zero,pullsDown:false);ratePicker.addItems(withTitles:["Угол: 10 чтений/с","Угол: 20 чтений/с","Угол: 50 чтений/с","Угол: 100 чтений/с","Угол: 200 чтений/с"]);ratePicker.selectItem(at:0);ratePicker.target=self;ratePicker.action=#selector(changeRate);view.addSubview(ratePicker)
        resetButton=NSButton(title:"Сбросить максимум света",target:self,action:#selector(resetLight));view.addSubview(resetButton)
        stats=label("Жду показаний датчика…",17);detail=label("",12)
        layout()
        let menu=NSMenu(),sub=NSMenu(),item=NSMenuItem();sub.addItem(withTitle:"Выйти",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q");item.submenu=sub;menu.addItem(item);NSApp.mainMenu=menu
        window.makeKeyAndOrderFront(nil);window.makeFirstResponder(view);NSApp.activate(ignoringOtherApps:true)
        reader.onSample={[weak self] kind,t,v in self?.inbox.add(kind == .gyro ? 0:1,t,v)}
        reader.onLidSample={[weak self] t,a in self?.inbox.add(2,t,SIMD3(Double(a),0,0))}
        reader.onLightSample={[weak self] t,lux in self?.inbox.add(3,t,SIMD3(lux,0,0))}
        reader.lidPollHz=10;reader.start();timer=Timer.scheduledTimer(withTimeInterval:1/30,repeats:true){[weak self] _ in self?.tick()}
    }
    func layout(){
        let width=view.bounds.width
        title.frame=NSRect(x:24,y:16,width:width-48,height:32)
        picker.frame=NSRect(x:24,y:58,width:620,height:28)
        previousButton.frame=NSRect(x:658,y:58,width:40,height:28);nextButton.frame=NSRect(x:704,y:58,width:40,height:28)
        pauseButton.frame=NSRect(x:width-234,y:58,width:210,height:28)
        mode.frame=NSRect(x:24,y:98,width:370,height:28)
        ratePicker.frame=NSRect(x:410,y:98,width:220,height:28)
        resetButton.frame=NSRect(x:650,y:98,width:240,height:28)
        stats.frame=NSRect(x:24,y:140,width:width-48,height:25)
        detail.frame=NSRect(x:24,y:167,width:width-48,height:23)
    }
    func windowDidResize(_ notification:Notification){layout();view.needsDisplay=true}
    @objc func selectSensor(){selected=picker.selectedSegment;updateLabels();window.makeFirstResponder(view)}
    @objc func previous(){selected=(selected+3)%4;picker.selectedSegment=selected;updateLabels();window.makeFirstResponder(view)}
    @objc func next(){selected=(selected+1)%4;picker.selectedSegment=selected;updateLabels();window.makeFirstResponder(view)}
    @objc func changeMode(){useDynamic=mode.selectedSegment==1;view.needsDisplay=true;window.makeFirstResponder(view)}
    @objc func changeRate(){reader.lidPollHz=[10.0,20,50,100,200][ratePicker.indexOfSelectedItem];window.makeFirstResponder(view)}
    @objc func resetLight(){lightMin = .infinity;lightMax=0;peakLightRate=0;updateLabels();window.makeFirstResponder(view)}
    @objc func togglePause(){paused.toggle();pauseTime=paused ? MachClock.now():nil;pauseButton.title=paused ? "Продолжить (пробел)":"Стоп-кадр (пробел)";updateLabels();window.makeFirstResponder(view)}
    func updateLabels(){
        title.stringValue=["Гироскоп · три оси","Акселерометр · три оси","Угол крышки · один канал","Освещённость · люксы и быстрые изменения"][selected]
        resetButton.isHidden=selected != 3
        mode.isEnabled=selected != 2;ratePicker.isEnabled=selected==2
        let n=selected,now=MachClock.now()
        if arrival[n]==0 || now-arrival[n]>1 {
            stats.stringValue="Нет свежих показаний"+(reader.error.map{" · \($0)"} ?? "")
        } else {
            let ms=rates[n]>0 ? 1000/rates[n]:0
            stats.stringValue=String(format:"Чтений: %.1f /с · средний интервал: %.2f мс · возраст последнего отчёта: %.0f мс",rates[n],ms,max(0,now-arrival[n])*1000)
        }
        if selected==3 {
            let lux=lastValues[3]?.x ?? 0
            detail.stringValue=String(format:"Сейчас %.5f лк · мин %.5f / макс %.5f лк за сеанс · пик потока %.1f отч./с · изменений %.1f /с",lux,lightMin.isFinite ? lightMin : 0,lightMax,peakLightRate,changeRates[3])
        } else if selected==2 {
            detail.stringValue=String(format:"Изменений значения: %.1f /с · запрошен опрос %.0f /с · текущее значение: %@",changeRates[2],reader.lidPollHz,reader.lidAngle.map{String(format:"%.2f°",$0)} ?? "—")
        } else {
            let v=lastValues[n] ?? .zero
            detail.stringValue=String(format:"Сырые значения: X %+.5f    Y %+.5f    Z %+.5f · график %@",v.x,v.y,v.z,paused ? "заморожен":"обновляется 30 кадров/с")
        }
        view.needsDisplay=true
    }
    func tick(){
        let now=MachClock.now()
        if rateStart==0{rateStart=now}
        for (k,t,v) in inbox.drain(){
            var dynamic=v
            if k != 2 {
                if baseline[k]==nil{baseline[k]=v}
                baseline[k]! += (v-baseline[k]!)*(k==3 ? 0.25 : 0.038);dynamic=v-baseline[k]!
            }
            if let old=lastValues[k],old != v {changes[k]+=1}
            lastValues[k]=v;arrival[k]=t;counts[k]+=1
            if k==3 {lightMin=min(lightMin,v.x);lightMax=max(lightMax,v.x);if lightCount==0{lightFirstTime=t};lightLastTime=t;lightCount+=1}
            if !paused {history.append(Reading(kind:k,t:t,raw:v,dynamic:dynamic))}
        }
        if !paused {history.removeAll{$0.t<now-windowSeconds}}
        let dt=now-rateStart
        if dt>=1 {for i in 0..<4{rates[i]=Double(counts[i])/dt;changeRates[i]=Double(changes[i])/dt;counts[i]=0;changes[i]=0};if lightCount>=2 && lightLastTime>lightFirstTime {rates[3]=Double(lightCount-1)/(lightLastTime-lightFirstTime)};peakLightRate=max(peakLightRate,rates[3]);lightCount=0;rateStart=now}
        updateLabels()
    }
    func windowWillClose(_ notification:Notification){NSApp.terminate(nil)}
}
let app=NSApplication.shared,controller=ScopeController();app.delegate=controller;app.run()
