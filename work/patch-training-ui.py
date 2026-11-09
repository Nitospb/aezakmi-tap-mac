from pathlib import Path
p=Path('outputs/tap-model/live/TapPredictor.swift');s=p.read_text()
s=s.replace('else if event.keyCode==20 {controller?.showCandidates.toggle();needsDisplay=true}', '''else if event.keyCode==20 {controller?.showCandidates.toggle();needsDisplay=true}
        else if event.keyCode==18 {controller?.retrain()}
        else if event.keyCode==19 {controller?.newConditions()}
        else if event.keyCode==29 {controller?.resetModel()}''')
s=s.replace('else if c.manualControl != nil', 'else if c.training {text="Дообучение · датчики продолжают сохраняться"}\n        else if let notice=c.notice,MachClock.now()<c.noticeUntil {text=notice}\n        else if c.manualControl != nil')
s=s.replace('Мышь — правильная точка · N — новые условия · 3 — варианты · → — не распознал · ⌘Q — сохранить и выйти','Мышь — правильная точка · 1 — дообучить · 2/N — условия · 0 — базовая модель · 3 — варианты · → — не распознал · ⌘Q — выход')
s=s.replace('var pendingID:String?', 'var training=false,resumeAfterTrain=false,notice:String?,noticeUntil=0.0,pendingBlock=1\n    var pendingID:String?')
s=s.replace('var answers=[String:(col:Int,row:Int,candidates:[[Int]])]()', 'var answers=[String:(col:Int,row:Int,candidates:[[Int]],version:Int)]()')
s=s.replace('var text="Сохранено:', 'var text="v\\(model?.version ?? 0) · Сохранено:')
s=s.replace(' · отмечено \\(r.labelCount)', ' · отметок \\(r.labelCount)')
a=s.index('        if count>0 {');b=s.index('        saveStatus.stringValue=',a)
s=s[:a]+'''        if count>0 {
            let versions=Array(Set(answers.values.map{$0.version})).sorted().suffix(2)
            for version in versions {
                var exact=0,near=0,n=0
                for (id,label) in annotations {if let a=answers[id],a.version==version{n+=1;if a.col==label.col && a.row==label.row{exact+=1};if max(abs(a.col-label.col),abs(a.row-label.row))<=1{near+=1}}}
                if n>0{text += " · v\\(version): точно \\(exact)/\\(n), рядом \\(near)/\\(n)"}
            }
        }
'''+s[b:]
s=s.replace('guard let r=recorder,saveFailure==nil,pending==nil,!predicting else{return}', 'guard let r=recorder,saveFailure==nil,pending==nil,!predicting,!training else{return}')
s=s.replace('guard recorder != nil,saveFailure==nil,pending==nil,!predicting else{return}', 'guard recorder != nil,saveFailure==nil,pending==nil,!predicting,!training else{return}')
s=s.replace('if running {pause();return}', 'if training{flash("Дообучение ещё идёт");return}\n        if running {pause();return}')
s=s.replace('recorder=try PlayRecorder()', 'recorder=try PlayRecorder(activeModel:model?.modelPath.map{URL(fileURLWithPath:$0)})')
s=s.replace('running=false;ready=false;pending=nil;pendingID=nil;manualControl=nil', 'running=false;ready=false;pending=nil;pendingID=nil;manualControl=nil;resumeAfterTrain=false')
s=s.replace('pending=s.t;pendingID=UUID().uuidString;', 'pending=s.t;pendingID=UUID().uuidString;pendingBlock=recorder?.block ?? 1;notice=nil;')
s=s.replace('let onsets=pendingOnsets;predicting=true', 'let onsets=pendingOnsets;let block=pendingBlock;predicting=true')
s=s.replace('details:result.details,onsets:onsets)', 'details:result.details,onsets:onsets,captureBlock:block)')
s=s.replace('self.answers[id]=(Int(result.point.x),Int(result.point.y),cs)', 'self.answers[id]=(Int(result.point.x),Int(result.point.y),cs,result.details["version"] as? Int ?? 0)')
s=s.replace('        guard running else{return}', '        if training{view.needsDisplay=true;return}\n        guard running else{return}')
p.write_text(s)
p=Path('outputs/tap-model/live/feedback_training.py');s=p.read_text().replace("if label.get('event')=='mouse_ground_truth':labels[label['prediction_id']]=label", "if label.get('event')=='mouse_ground_truth':labels[label['prediction_id']]=label\n   elif label.get('event')=='missed_tap_ground_truth':skipped.append(dict(id=label.get('event_id'),reason='Manual missed tap has no exact trigger; raw interval retained for later processing'))")
p.write_text(s)
