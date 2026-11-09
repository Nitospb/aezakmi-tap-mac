from pathlib import Path
p=Path('outputs/TapCollector.swift');s=p.read_text()
s=s.replace('let reader = IMUReader()', 'let reader = IMUReader()\n    let microphone = MicrophoneRecorder()\n    var requestingMicrophone = false\n    var lastLight=0.0,lastLid=0.0')
s=s.replace('var pendingLid: Int?', 'var pendingLid: Double?').replace('var lastLoggedLid: Int?', 'var lastLoggedLid: Double?').replace('var initialLid: Int?', 'var initialLid: Double?')
s=s.replace('reader.start()\n        timer', '''reader.onLidSample = { [weak self] t,a in
            guard let self else{return};self.inbox.add(Sample(kind:2,t:t,v:SIMD3(a,Double(self.reader.lidReportID),self.reader.lidResolution)))
        }
        reader.onLightSample = { [weak self] t,lux in
            guard let self else{return};let ch=self.reader.lightChannels
            self.inbox.add(Sample(kind:3,t:t,v:SIMD3(lux,0,0)))
            self.inbox.add(Sample(kind:4,t:t,v:SIMD3(Double(ch.x),Double(ch.y),Double(ch.z))))
            self.inbox.add(Sample(kind:5,t:t,v:SIMD3(Double(ch.w),0,0)))
        }
        reader.lidPollHz=200
        reader.start()
        timer''')
s=s.replace('session = folder; sessionStarted', 'try microphone.prepare(folder:folder)\n        session = folder; sessionStarted')
s=s.replace('"schema": 3', '"schema": 4')
s=s.replace('uint64 kind (0=accel,1=gyro)', 'uint64 kind (0=accel,1=gyro,2=lid[x=degrees,y=reportID,z=resolution],3=ALS[x=lux],4=ALSraw[x=ch0,y=ch1,z=ch2],5=ALSraw[x=ch3])')
s=s.replace('"lid_angle": "degrees"]', '"lid_angle": "degrees", "light": "lux", "light_raw": "uncalibrated counts"]')
s=s.replace('try JSONSerialization.data(withJSONObject: meta,', '''var metadata=meta
        metadata["lid_source"]=["preferred_report":7,"active_report":reader.lidReportID,"resolution_degrees":reader.lidResolution,"requested_reads_hz":reader.lidPollHz,"precision_note":"Representation step, not physical accuracy. Every successful read is retained, including repeated values."]
        metadata["light_source"]=["usage_page":65280,"usage":4,"report_bytes":122,"requested_interval_us":1000,"measured_report_hz_approx":9.95,"decode":"Float32LE lux@40; UInt32LE raw channels@20,24,28,32"]
        metadata["audio"]=["file":"microphone.caf","index":"audio-chunks.jsonl","sample_rate":microphone.sampleRate,"channels":microphone.channelCount,"device":microphone.deviceName,"format":"lossless Float32 PCM, native input sample rate/channels; no app denoising or normalization","recording_policy":"Only while collection is running, including settling/capture/feedback; paused with Space and on loss of focus","host_clock":"AVAudioTime.hostTime / mach_absolute_time; each chunk has an anchor; no invented samples across pauses","presentation_latency_seconds":microphone.inputLatency,"sync_note":"Common host clock; sensor/audio hardware latency is not independently calibrated"]
        try JSONSerialization.data(withJSONObject: metadata,''')
s=s.replace('guard !finished, failure == nil else { return }\n        guard hz', '''guard !finished, failure == nil, !requestingMicrophone else { return }
        if MicrophoneRecorder.permission != .authorized {
            if MicrophoneRecorder.permission == .notDetermined {
                requestingMicrophone=true;startProblem="Разреши микрофон для записи тапов";redraw()
                MicrophoneRecorder.requestPermission { [weak self] ok in
                    guard let self else{return};self.requestingMicrophone=false
                    self.startProblem=ok ? "Микрофон разрешён · пробел — начать" : "Нужен доступ к микрофону в настройках macOS"
                    self.redraw()
                }
            } else {startProblem="Разреши Tap Collector доступ к микрофону в настройках macOS";redraw()}
            return
        }
        guard MachClock.now()-lastLight<1, MachClock.now()-lastLid<0.5 else {startProblem="Жду свет и дробный угол крышки · пробел — повторить";redraw();return}
        guard reader.lidReportID==7 else {startProblem="Нет дробного отчёта угла №7 · сбор не начат";redraw();return}
        guard hz''')
s=s.replace('running = true; showCue();', 'do {try microphone.resume()} catch {fail(error);return}\n        running = true; showCue();')
s=s.replace('running = false; pending = nil; redraw()', 'running = false; pending = nil; microphone.pause(); redraw()')
s=s.replace('for s in interval {\n            _ = localDetector', 'for s in interval where s.kind<2 {\n            _ = localDetector')
s=s.replace('finished = true; startButton?', 'microphone.close()\n        finished = true; startButton?')
s=s.replace('failure = error.localizedDescription; running = false; pending = nil', 'failure = error.localizedDescription; running = false; pending = nil; microphone.pause()')
s=s.replace('let now = MachClock.now(), batch = inbox.drain()', 'let now = MachClock.now(), batch = inbox.drain().sorted{$0.t<$1.t}\n        if running,let error=microphone.error {fail(NSError(domain:"AudioCapture",code:1,userInfo:[NSLocalizedDescriptionKey:error]));return}')
s=s.replace('if s.kind == 1 { lastGyro = s.t; gyroCounter += 1 } else { lastAccel = s.t }', 'if s.kind == 1 { lastGyro = s.t; gyroCounter += 1 } else if s.kind==0 { lastAccel = s.t } else if s.kind==2 {lastLid=s.t} else if s.kind==3 {lastLight=s.t}\n            guard s.kind<2 else {continue}')
s=s.replace('if session != nil && !finished, let angle', '''if running && (now-lastLight>1 || now-lastLid>0.7 || now-microphone.lastEnd>1.0) && now-cue>1.2 {pause();startProblem="Прервался свет, угол или микрофон · пробел — повторить";redraw()}
        if session != nil && !finished, let angle''')
s=s.replace('now-t >= 0.46', 'now-t >= 0.55 && microphone.lastEnd >= t+0.40')
s=s.replace('else if now-t > 1.4', 'else if now-t > 1.4')
s=s.replace('let angle = reader.lidAngle.map { "\\($0)°" }', 'let angle = reader.lidAngle.map { String(format:"%.2f°",$0) }')
s=s.replace('let grid = gridRect(), cell', '''let lo=manualSamples == nil ? t-0.15 : cue, hi=manualSamples == nil ? t+0.40 : t
        microphone.flush()
        let audioRefs=microphone.references(from:lo,to:hi)
        let audioComplete=microphone.covers(audioRefs,from:lo,to:hi)
        if manualSamples == nil && !audioComplete {reject("Неполный звук · повтори один тап",reason:"incomplete_audio_window");return}
        let lid=samples.filter{$0.kind==2},light=samples.filter{$0.kind==3}
        guard !lid.isEmpty,!light.isEmpty else {reject("Нет угла или света · повтори тап",reason:"incomplete_extra_sensors");return}
        let grid = gridRect(), cell''')
s=s.replace('trial["capture_source"]', '''trial["lid_degrees"] = lid.map{[$0.t,$0.v.x]}
        trial["light_lux"] = light.map{[$0.t,$0.v.x]}
        let fourth=Dictionary(samples.filter{$0.kind==5}.map{($0.t,$0.v.x)},uniquingKeysWith:{$1})
        trial["light_raw_channels"] = samples.filter{$0.kind==4}.map{[$0.t,$0.v.x,$0.v.y,$0.v.z,fourth[$0.t] ?? 0]}
        trial["audio"] = ["file":"microphone.caf","index":"audio-chunks.jsonl","interval_start":lo,"interval_end":hi,"coverage_complete":audioComplete,"frame_ranges":audioRefs]
        trial["capture_source"]''')
# expose stream status as a small overlay on the paused screen, no extra panel.
s=s.replace('text = c.startProblem ?? "Пробел — старт/пауза · ← перезаписать · → я тапнул"', 'text = c.startProblem ?? "Пробел — старт/пауза · 5 источников · ← повторить · → я тапнул"')
p.write_text(s)
