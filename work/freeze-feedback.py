from pathlib import Path
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text().replace('controller?.missedTap()','controller?.nextTap()')
s=s.replace('            let selected=c.prediction.map', '            let actual=c.truthPoint.map{Int($0.x*20)==col && Int($0.y*13)==row} ?? false\n            let selected=c.prediction.map')
s=s.replace('(selected ? NSColor.systemGreen : NSColor(calibratedWhite:0.13,alpha:1)).setFill()', '(c.confirmed && actual ? (c.confirmedCorrect ? NSColor.systemGreen.withAlphaComponent(0.4) : NSColor.systemBlue.withAlphaComponent(0.75)) : (selected ? NSColor.systemOrange.withAlphaComponent(c.confirmed ? 0.2 : 0.75) : NSColor(calibratedWhite:0.13,alpha:1))).setFill()')
s=s.replace('            NSColor.systemGreen.setFill();NSBezierPath(roundedRect:selectedRect,xRadius:5,yRadius:5).fill()', '            (c.confirmed && c.confirmedCorrect ? NSColor.systemGreen : NSColor.systemOrange).setStroke();let outline=NSBezierPath(roundedRect:selectedRect,xRadius:5,yRadius:5);outline.lineWidth=2;outline.stroke()')
s=s.replace('NSColor.white.setStroke();let cross=NSBezierPath();cross.lineWidth=2', '(c.confirmed && c.confirmedCorrect ? NSColor.systemGreen : NSColor.white).setStroke();let cross=NSBezierPath();cross.lineWidth=3')
s=s.replace('if let p=c.truthPoint {', 'if let p=c.truthPoint,!c.confirmedCorrect {')
s=s.replace('else if c.manualControl != nil {text="Мышью отметь, где был нераспознанный тап"}', '''else if c.locked,c.prediction != nil {
            text=c.confirmed ? (c.confirmedCorrect ? "✓ Верно · сохранено · → следующий тап" : "Правильная клетка синяя · сохранено · → следующий тап") : "Предсказание заморожено · отметь мышью · → пропустить срабатывание"
        }''')
s=s.replace('else if let p=c.prediction {text="Клетка \\(Int(p.x)+1), \\(Int(p.y)+1) · мышью можно отметить, где был тап"}', 'else if c.prediction != nil {text="→ — следующий тап"}')
s=s.replace('→ — не распознал', '→ — следующий тап')
s=s.replace('    var manualControl:Double?', '''    var manualControl:Double?
    var locked=false,confirmed=false
    var confirmedCorrect:Bool {
        guard confirmed,let p=prediction,let t=truthPoint else{return false}
        return Int(p.x)==Int(t.x*20) && Int(p.y)==Int(t.y*13)
    }''')
s=s.replace('if !running{text += " · ПАУЗА: датчики сохраняются"}', 'if !running{text += " · ПАУЗА: датчики сохраняются"}\n        if locked{text += confirmed ? " · подтверждено →" : " · ждёт отметку мышью"}')
s=s.replace('guard let r=recorder,saveFailure==nil,pending==nil,!predicting,!training else{return}', 'guard let r=recorder,saveFailure==nil,pending==nil,!predicting,!training,locked else{return}')
s=s.replace('truthPoint=SIMD2(x,y);error=nil;', 'truthPoint=SIMD2(x,y);confirmed=true;ready=false;error=nil;')
a=s.index('    func missedTap(){');b=s.index('    func newConditions(){',a)
s=s[:a]+'''    func nextTap(){
        guard recorder != nil,saveFailure==nil,!training else{return}
        guard pending==nil,!predicting else{flash("Дождись предсказания");return}
        log("feedback_next",["prediction_id":latestID as Any? ?? NSNull(),"label_confirmed":confirmed,"skipped_unlabelled":locked && !confirmed])
        locked=false;confirmed=false;prediction=nil;latestID=nil;truthPoint=nil;candidates=[];manualControl=nil;notice=nil;error=nil
        detector=TapDetector();ready=false;armedAfter=MachClock.now()+0.65;status();view.needsDisplay=true
    }
'''+s[b:]
s=s.replace('do {try r.newBlock(lid:reader.lidAngle);prediction=nil;latestID=nil;truthPoint=nil;candidates=[];armedAfter=MachClock.now()+0.6;ready=false;status();view.needsDisplay=true}', 'do {try r.newBlock(lid:reader.lidAngle);armedAfter=MachClock.now()+0.6;ready=false;status();view.needsDisplay=true}')
s=s.replace('status();ready=false;pending=nil;prediction=nil;armedAfter=', 'status();ready=false;pending=nil;if !locked{prediction=nil};armedAfter=')
s=s.replace('running=false;ready=false;pending=nil;pendingID=nil;manualControl=nil;resumeAfterTrain=false', 'if pending != nil{locked=false;confirmed=false;prediction=nil;latestID=nil}\n        running=false;ready=false;pending=nil;pendingID=nil;manualControl=nil;resumeAfterTrain=false')
s=s.replace('if running && ready && pending==nil && manualControl==nil', 'if running && ready && !locked && pending==nil && manualControl==nil')
s=s.replace('pending=s.t;pendingID=UUID().uuidString;', 'locked=true;confirmed=false;pending=s.t;pendingID=UUID().uuidString;')
s=s.replace('(s.t-lastInput<=0.45 ? "control_input":"busy_or_settling")', '(locked ? "feedback_locked" : (s.t-lastInput<=0.45 ? "control_input":"busy_or_settling"))')
s=s.replace('case .failure(let e):self.log("prediction_failed"', 'case .failure(let e):self.locked=false;self.confirmed=false;self.log("prediction_failed"')
s=s.replace('if pending==nil && !predicting && manualControl==nil && !ready', 'if pending==nil && !predicting && !locked && manualControl==nil && !ready')
p.write_text(s)
p=Path('outputs/tap-model/live/TrainingControls.swift');s=p.read_text().replace('self.prediction=nil;self.latestID=nil;self.candidates=[];self.truthPoint=nil;', 'self.locked=false;self.confirmed=false;self.prediction=nil;self.latestID=nil;self.candidates=[];self.truthPoint=nil;');p.write_text(s)
