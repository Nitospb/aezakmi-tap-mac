from pathlib import Path
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text().replace('model:SpatialModel?', 'model:LiveModel?').replace('pendingLid:Int?', 'pendingLid:Double?')
s=s.replace('var error:String?', 'var error:String?\n    var predicting=false,lastLight=0.0,lastLid=0.0')
a=s.index('        do {model=try SpatialModel');b=s.index('        window.makeKeyAndOrderFront',a)
s=s[:a]+'''        model=LiveModel();error="Модель загружается…"
        model?.onStatus = {[weak self] problem in self?.error=problem.map{"Ошибка модели: \\($0)"};self?.view.needsDisplay=true};model?.start()
'''+s[b:]
s=s.replace('func drain()->[MotionSample]', 'func add(_ batch:[MotionSample]) {lock.withLock{buffer.append(contentsOf:batch)}}\n    func drain()->[MotionSample]')
s=s.replace('};reader.start()', '''}
        reader.onLidSample = {[weak self] t,a in self?.inbox.add(MotionSample(kind:2,t:t,v:SIMD3(a,0,0)))}
        reader.onLightSample = {[weak self] t,l in guard let self else{return};let c=self.reader.lightChannels;self.inbox.add([MotionSample(kind:3,t:t,v:SIMD3(l,0,0)),MotionSample(kind:4,t:t,v:SIMD3(Double(c.x),Double(c.y),Double(c.z))),MotionSample(kind:5,t:t,v:SIMD3(Double(c.w),0,0))])}
        reader.lidPollHz=200;reader.start()''')
s=s.replace('guard model != nil else{return}', 'guard model?.ready == true else{error="Модель загружается · подожди";view.needsDisplay=true;return}\n        guard MachClock.now()-lastLight<1,MachClock.now()-lastLid<0.5 else{error="Жду свет и угол крышки";view.needsDisplay=true;return}')
s=s.replace('let now=MachClock.now(),batch=inbox.drain()', 'let now=MachClock.now(),batch=inbox.drain().sorted{$0.t<$1.t}')
s=s.replace('if s.kind==1 {lastGyro=s.t}else{lastAccel=s.t}', 'if s.kind==1 {lastGyro=s.t}else if s.kind==0{lastAccel=s.t}else if s.kind==2{lastLid=s.t}else if s.kind==3{lastLight=s.t}\n            guard s.kind<2 else{continue}')
a=s.index('        if let t=pending,now-t>0.46 {');b=s.index('        if pending==nil && !ready',a)
s=s[:a]+'''        if let t=pending,now-t>0.55 && !predicting {
            let samples=ring.filter{$0.t>=t-0.9 && $0.t<=t+0.4};predicting=true
            model!.predict(samples:samples,trigger:t,lid:pendingLid){[weak self] result in
                guard let self else{return};self.predicting=false
                guard self.running else{self.pending=nil;return}
                switch result {
                case .success(let p):self.prediction=p;self.error=nil
                case .failure(let e):self.error="Не удалось распознать: \\(e.localizedDescription)"
                }
                self.pending=nil;self.armedAfter=MachClock.now()+0.6;self.view.needsDisplay=true
            }
        }
'''+s[b:]
s=s.replace('pending==nil && !ready &&', 'pending==nil && !predicting && !ready &&')
s=s.replace('guard running else{return}\n        if now-lastGyro', 'guard running else{return}\n        if now-lastLight>1 || now-lastLid>0.7 {pause();error="Прервался свет или угол · пробел — повторить";view.needsDisplay=true;return}\n        if now-lastGyro')
s=s.replace('    func windowWillClose', '    func applicationWillTerminate(_ notification:Notification){model?.stop()}\n    func windowWillClose')
s=s.replace('text="Пробел — распознавание"', 'text="Пробел — распознавание · новая модель со светом"')
p.write_text(s)
