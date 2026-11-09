from pathlib import Path
p=Path('outputs/sensor-scope/IMUReader.swift');s=p.read_text()
s=s.replace('enum Kind { case accel, gyro }','enum Kind { case accel, gyro, light }')
s=s.replace('var onLidSample:', 'var onLightSample: ((Double, Double) -> Void)?\n    private var _lightChannels = SIMD4<UInt32>(repeating: 0)\n    var lightChannels: SIMD4<UInt32> { lock.withLock { _lightChannels } }\n    var onLidSample:')
s=s.replace('private(set) var foundLid = false','private(set) var foundLid = false\n    private(set) var foundLight = false')
s=s.replace('guard page == 0xFF00, usage == 3 || usage == 9, intProperty(service, "MaxInputReportSize") == 22 else { continue }\n            let kind: Kind = usage == 3 ? .accel : .gyro', '''let size = intProperty(service, "MaxInputReportSize")
            guard page == 0xFF00, (usage == 4 && size == 122) || ((usage == 3 || usage == 9) && size == 22) else { continue }
            let kind: Kind = usage == 4 ? .light : (usage == 3 ? .accel : .gyro)''')
s=s.replace('(kind == .gyro && foundGyro) { continue }','(kind == .gyro && foundGyro) || (kind == .light && foundLight) { continue }')
s=s.replace('guard let context, length == 22 else { return }','guard let context else { return }')
s=s.replace('let t = Double(timestamp) * MachClock.tick\n                reader.lastSample', '''let t = Double(timestamp) * MachClock.tick
                if box.kind == .light {
                    guard length == 122 else { return }
                    // Apple SPU ALS: lux Float32 LE @40; four raw UInt32 LE channels @20..32.
                    func u32(_ offset: Int) -> UInt32 { var n: UInt32 = 0; memcpy(&n, report + offset, 4); return UInt32(littleEndian: n) }
                    let lux = Double(Float(bitPattern: u32(40)))
                    guard lux.isFinite, lux >= 0 else { return }
                    reader.lock.withLock { reader._lightChannels = SIMD4(u32(20),u32(24),u32(28),u32(32)) }
                    reader.onLightSample?(t, lux)
                    return
                }
                guard length == 22 else { return }
                reader.lastSample''')
s=s.replace('if kind == .accel { foundAccel = true } else { foundGyro = true }','if kind == .accel { foundAccel = true } else if kind == .gyro { foundGyro = true } else { foundLight = true }')
s=s.replace('foundLid = false\n        lock', 'foundLid = false\n        foundLight = false\n        lock')
p.write_text(s)
p=Path('outputs/sensor-scope/SensorScope.swift');s=p.read_text()
s=s.replace('channels=kind==2 ? 1:3','channels=kind>=2 ? 1:3')
s=s.replace('kind==2 ? .systemOrange :','kind==3 ? .systemYellow : (kind==2 ? .systemOrange :').replace('[.systemRed,.systemGreen,.systemCyan][axis]','[.systemRed,.systemGreen,.systemCyan][axis])')
s=s.replace('let name=kind==2 ? "Угол крышки" :','let name=kind==3 ? "Освещённость" : (kind==2 ? "Угол крышки" :').replace('["X","Y","Z"][axis]','["X","Y","Z"][axis])')
s=s.replace('(kind==1 ? "g":"°")','(kind==1 ? "g":(kind==3 ? "лк":"°"))')
s=s.replace('(kind==0 ? 0.1 : 0.002)','(kind==0 ? 0.1 : (kind==3 ? 0.01 : 0.002))')
s=s.replace('if kind==2, let lo=values.min(),let hi=values.max()', 'if kind>=2, let lo=values.min(),let hi=values.max()')
s=s.replace('String(format:"Размах %.2f°",hi-lo)','String(format:"Размах %.4f %@",hi-lo,unit)')
s=s.replace('let note=kind==2 ?', '''let note=kind==3 ? "Поток HID, запрос интервала 1 мс. Предел диапазона в люксах неизвестен; максимум выше — измеренный за сеанс." : (kind==2 ?''')
s=s.replace('Масштаб каждой оси автоматический.")','Масштаб каждой оси автоматический."))')
s=s.replace('text("← / → — следующий датчик · пробел — стоп-кадр · все три источника читаются одновременно",NSPoint(x:24,y:bounds.height-40),12,.secondaryLabelColor)', '''let footer = kind==3 ? "Сырые каналы: \\(c.reader.lightChannels) · колебания могут быть от тени руки, света экрана и мерцания ламп." : "← / → — следующий датчик · пробел — стоп-кадр · все четыре источника читаются одновременно"
        text(footer,NSPoint(x:24,y:bounds.height-40),12,.secondaryLabelColor)''')
s=s.replace('var previousButton:NSButton!,nextButton:NSButton!', 'var previousButton:NSButton!,nextButton:NSButton!,resetButton:NSButton!\n    var lightMin=Double.infinity,lightMax=0.0,peakLightRate=0.0')
s=s.replace('count:2)', 'count:4)').replace('count:3)', 'count:4)')
s=s.replace('var selected=2,','var selected=3,')
s=s.replace('"3. Угол крышки"]','"3. Угол крышки","4. Освещённость"]')
s=s.replace('stats=label("Жду показаний датчика…",17);detail=label("",12)', '''resetButton=NSButton(title:"Сбросить максимум света",target:self,action:#selector(resetLight));view.addSubview(resetButton)
        stats=label("Жду показаний датчика…",17);detail=label("",12)''')
s=s.replace('reader.lidPollHz=10;reader.start();','reader.onLightSample={[weak self] t,lux in self?.inbox.add(3,t,SIMD3(lux,0,0))}\n        reader.lidPollHz=10;reader.start();')
s=s.replace('width:520,height:28','width:620,height:28').replace('x:558,y:58','x:658,y:58').replace('x:604,y:58','x:704,y:58')
s=s.replace('stats.frame=NSRect', 'resetButton.frame=NSRect(x:650,y:98,width:240,height:28)\n        stats.frame=NSRect')
s=s.replace('selected=(selected+2)%3','selected=(selected+3)%4').replace('selected=(selected+1)%3','selected=(selected+1)%4')
s=s.replace('@objc func togglePause()', '@objc func resetLight(){lightMin = .infinity;lightMax=0;peakLightRate=0;updateLabels();window.makeFirstResponder(view)}\n    @objc func togglePause()')
s=s.replace('"Угол крышки · один канал"][selected]', '"Угол крышки · один канал","Освещённость · люксы и быстрые изменения"][selected]')
s=s.replace('mode.isEnabled=selected != 2;', 'resetButton.isHidden=selected != 3\n        mode.isEnabled=selected != 2;')
s=s.replace('if selected==2 {', '''if selected==3 {
            let lux=lastValues[3]?.x ?? 0
            detail.stringValue=String(format:"Сейчас %.5f лк · мин %.5f / макс %.5f лк за сеанс · пик потока %.1f отч./с · изменений %.1f /с",lux,lightMin.isFinite ? lightMin : 0,lightMax,peakLightRate,changeRates[3])
        } else if selected==2 {''')
s=s.replace('if k<2 {', 'if k != 2 {')
s=s.replace('(v-baseline[k]!)*0.038','(v-baseline[k]!)*(k==3 ? 0.25 : 0.038)')
s=s.replace('lastValues[k]=v;arrival[k]=t;counts[k]+=1','''lastValues[k]=v;arrival[k]=t;counts[k]+=1
            if k==3 {lightMin=min(lightMin,v.x);lightMax=max(lightMax,v.x)}''')
s=s.replace('for i in 0..<3', 'for i in 0..<4')
s=s.replace('};rateStart=now}', '};peakLightRate=max(peakLightRate,rates[3]);rateStart=now}')
p.write_text(s)
