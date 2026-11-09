import Cocoa
extension Predictor {
    func flash(_ text:String){notice=text;noticeUntil=MachClock.now()+5;view.needsDisplay=true}
    func retrain(){
        guard let r=recorder,saveFailure==nil,model?.ready==true,!training else{return}
        guard pending==nil,!predicting else{flash("Дождись результата текущего тапа");return}
        let wasRunning=running;pause();resumeAfterTrain=wasRunning;training=true;notice=nil
        log("training_requested",["version":model?.version ?? 0,"label_events":r.labelCount]);view.needsDisplay=true
        model!.command(["op":"train","session_folder":r.folder.path]){[weak self] result in
            guard let self else{return};self.training=false
            switch result {
            case .success(let info):
                self.log("training_finished",info);do{try self.recorder?.sync()}catch{self.storageFailed(error)}
                if info["trained"] as? Bool==true{self.locked=false;self.confirmed=false;self.manualID=nil;self.manualControl=nil;self.manualSamples=[];self.lastRelease=MachClock.now();self.prediction=nil;self.latestID=nil;self.candidates=[];self.truthPoint=nil;let count=info["training_feedback"] as? Int ?? 0;let skipped=(info["skipped"] as? [[String:Any]])?.count ?? 0;self.flash("v\(self.model!.version) · обучено \(count) отметок · неясных \(skipped) · проверяй новые тапы")}
                else{self.flash(info["reason"] as? String ?? "Нет новых отметок")}
            case .failure(let e):self.log("training_failed",["error":e.localizedDescription]);self.flash("Дообучение не удалось · прежняя модель сохранена")
            }
            if self.resumeAfterTrain && self.saveFailure==nil{self.running=true;self.ready=false;self.armedAfter=MachClock.now()+0.6;self.log("resume_after_training")}
            self.status();self.view.needsDisplay=true
        }
    }
    func resetModel(){
        guard recorder != nil,saveFailure==nil,model?.ready==true,!training,!predicting,pending==nil else{return}
        let wasRunning=running;pause();training=true;resumeAfterTrain=wasRunning
        model!.command(["op":"reset"]){[weak self] result in
            guard let self else{return};self.training=false
            switch result{case .success(let info):self.locked=false;self.confirmed=false;self.manualID=nil;self.manualControl=nil;self.manualSamples=[];self.lastRelease=MachClock.now();self.prediction=nil;self.latestID=nil;self.candidates=[];self.truthPoint=nil;self.log("model_reset",info);self.flash("Базовая модель · записи и дообученные версии сохранены")
            case .failure(let e):self.flash(e.localizedDescription)}
            if self.resumeAfterTrain && self.saveFailure==nil{self.running=true;self.ready=false;self.armedAfter=MachClock.now()+0.6}
            self.status();self.view.needsDisplay=true
        }
    }
}
