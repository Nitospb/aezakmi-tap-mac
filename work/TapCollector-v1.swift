import Cocoa
import OSLog

enum Log { static let sensor = Logger(subsystem: "local.tapcollector", category: "sensor") }

struct Sample {
    let kind: Int
    let t: Double
    let v: SIMD3<Double>
    var json: [Double] { [t, v.x, v.y, v.z] }
}

final class CollectorWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class Inbox {
    private let lock = NSLock()
    private var samples: [Sample] = []
    func add(_ s: Sample) { lock.withLock { samples.append(s) } }
    func drain() -> [Sample] { lock.withLock { let s = samples; samples.removeAll(keepingCapacity: true); return s } }
}

final class GridView: NSView {
    weak var controller: Collector?
    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command), [12,13].contains(event.keyCode) {
            controller?.finish(); NSApp.terminate(nil); return
        }
        switch event.keyCode {
        case 49: controller?.toggle()
        case 51: controller?.undo()
        case 53: controller?.pause()
        default: super.keyDown(with: event)
        }
    }
    override func draw(_ dirtyRect: NSRect) {
        NSColor(calibratedWhite: 0.055, alpha: 1).setFill(); bounds.fill()
        guard let c = controller else { return }
        let grid = c.gridRect()
        let size = grid.width / CGFloat(c.cols)
        let height = grid.height / CGFloat(c.rows)
        for zone in 0..<(c.cols*c.rows) {
            let r = NSRect(x: grid.minX + CGFloat(zone%c.cols)*size, y: grid.minY + CGFloat(zone/c.cols)*height, width: size, height: height).insetBy(dx: 1, dy: 1)
            let active = zone == c.target && c.running
            (active ? (c.pending != nil ? NSColor.systemGreen : NSColor.systemOrange) : NSColor(calibratedWhite: 0.13, alpha: 1)).setFill()
            NSBezierPath(roundedRect: r, xRadius: 6, yRadius: 6).fill()
            let label = active ? "●" : ""
            let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedDigitSystemFont(ofSize: active ? 32 : 12, weight: .medium), .foregroundColor: active ? NSColor.black : NSColor.gray]
            let textSize = (label as NSString).size(withAttributes: attrs)
            (label as NSString).draw(at: NSPoint(x: r.midX-textSize.width/2, y: r.midY-textSize.height/2), withAttributes: attrs)
        }
        if !c.running {
            let text = c.failure.map { "Ошибка записи: \($0)" } ?? (c.finished ? "Данные сохранены · ⌘Q — выход" : (c.startProblem ?? "Пробел — начать"))
            let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 24, weight: .semibold), .foregroundColor: NSColor.white]
            let s = (text as NSString).size(withAttributes: attrs)
            let box = NSRect(x: bounds.midX-s.width/2-24, y: bounds.midY-30, width: s.width+48, height: 60)
            NSColor.black.withAlphaComponent(0.9).setFill(); NSBezierPath(roundedRect: box, xRadius: 10, yRadius: 10).fill()
            (text as NSString).draw(at: NSPoint(x: box.minX+24, y: box.minY+15), withAttributes: attrs)
        }
    }
}

final class Collector: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let reader = IMUReader()
    let inbox = Inbox()
    var window: NSWindow!
    var view: GridView!
    var status: NSTextField!
    var subtitle: NSTextField!
    var startButton: NSButton!
    var timer: Timer?
    var cols = 20, rows = 13, repetitions = 3
    var sequence: [Int] = []
    var index = 0
    var target: Int { sequence.isEmpty || index >= sequence.count ? -1 : sequence[index] }
    var running = false, finished = false
    var ring: [Sample] = []
    var pending: Double?
    var pendingLid: Int?
    var cue = 0.0, readyAt = 0.0
    var baseline = SIMD3<Double>(repeating: 0)
    var noise = 0.03
    var lastGyro = 0.0, rateStart = 0.0, gyroCounter = 0, hz = 0
    var lastInput = 0.0
    var session: URL?
    var raw: FileHandle?, events: FileHandle?, trials: FileHandle?
    var trialIDs: [String] = []
    var failure: String?
    var startProblem: String?
    var sessionStarted = 0.0
    var initialLid: Int?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        let screen = NSScreen.main!
        window = CollectorWindow(contentRect: screen.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.title = "Tap Collector — запись касаний"
        window.delegate = self
        view = GridView(frame: NSRect(origin: .zero, size: screen.frame.size)); view.controller = self
        window.contentView = view
        status = label("Проверяю датчик…", y: 14, size: 20)
        subtitle = label("Один лёгкий тап пальцем в центр оранжевой клетки. Зелёный — записано. Esc — пауза, ⌫ — отменить последний.", y: 46, size: 13)
        startButton = button("Начать / пауза", x: 20, action: #selector(toggle))
        _ = button("Отменить тап", x: 178, action: #selector(undo))
        _ = button("Клетки крупнее", x: 324, action: #selector(coarser))
        _ = button("Клетки мельче", x: 480, action: #selector(finer))
        _ = button("Завершить и сохранить", x: 635, action: #selector(finish))
        _ = label("Только сбор данных · клики мышью не выполняются · держи угол крышки и опору неизменными · касайся легко", y: 105, size: 12)
        for element in view.subviews { element.isHidden = true }
        NSApp.presentationOptions = [.hideDock, .hideMenuBar]
        window.makeKeyAndOrderFront(nil); window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
        reader.onSample = { [weak self] kind, t, v in self?.inbox.add(Sample(kind: kind == .gyro ? 1 : 0, t: t, v: v)) }
        reader.start()
        timer = Timer.scheduledTimer(withTimeInterval: 0.025, repeats: true) { [weak self] _ in self?.tick() }
        NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown, .scrollWheel]) { [weak self] event in
            self?.lastInput = MachClock.now(); return event
        }
    }
    func label(_ text: String, y: CGFloat, size: CGFloat) -> NSTextField {
        let l = NSTextField(labelWithString: text); l.font = .systemFont(ofSize: size)
        l.textColor = .white; l.frame = NSRect(x: 20, y: y, width: max(700, view.bounds.width-40), height: 25)
        l.autoresizingMask = [.width]; view.addSubview(l); return l
    }
    func button(_ title: String, x: CGFloat, action: Selector) -> NSButton {
        let b = NSButton(title: title, target: self, action: action)
        b.frame = NSRect(x: x, y: 75, width: title.count > 20 ? 210 : 150, height: 28)
        b.bezelStyle = .rounded; view.addSubview(b); return b
    }
    func gridRect() -> NSRect {
        return view.bounds
    }
    func windowDidResize(_ notification: Notification) {
        if running { pause() }; view?.needsDisplay = true
    }
    func applicationDidResignActive(_ notification: Notification) { pause() }
    @objc func coarser() { resizeGrid(-2) }
    @objc func finer() { resizeGrid(2) }
    func resizeGrid(_ delta: Int) {
        guard session == nil else { subtitle.stringValue = "Размер сетки фиксируется на сеанс. Для другой сетки заверши запись и открой приложение снова."; return }
        cols = min(24, max(4, cols+delta)); rows = max(3, Int((Double(cols)*2/3).rounded())); view.needsDisplay = true
    }
    func createSession() throws {
        let root = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("tap-data")
        let folder = root.appendingPathComponent(ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-") + "-" + UUID().uuidString.prefix(6))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        func file(_ name: String) throws -> FileHandle {
            let url = folder.appendingPathComponent(name)
            guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw NSError(domain: "CreateFile", code: 1) }
            return try FileHandle(forWritingTo: url)
        }
        raw = try file("samples.bin"); events = try file("events.jsonl"); trials = try file("trials.jsonl")
        session = folder; sessionStarted = MachClock.now(); initialLid = reader.lidAngle
        sequence = (0..<repetitions).flatMap { _ in Array(0..<(cols*rows)).shuffled() }
        let meta: [String: Any] = ["schema": 1, "created_at": ISO8601DateFormatter().string(from: Date()), "cols": cols, "rows": rows, "repetitions": repetitions,
            "sequence": sequence, "clock": "mach_absolute_time in seconds; shared by samples and events", "raw_format": "40 bytes little-endian per record: uint64 kind (0=accel,1=gyro), float64 timestamp_seconds, float64 x,y,z", "units": ["accel": "g", "gyro": "degrees/second", "lid_angle": "degrees"],
            "labels": "Prompted target center; user compliance assumed. Not independently observed finger coordinates.", "initial_lid_angle": reader.lidAngle as Any? ?? NSNull(),
            "os": ProcessInfo.processInfo.operatingSystemVersionString, "sensor_code_credit": "Based on LidTap by Yemcube https://github.com/yemcube/lidtap", "coordinate_origin": "top-left of grid; x_normalized/y_normalized cover grid rectangle", "session_start_monotonic": sessionStarted]
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: folder.appendingPathComponent("metadata.json"), options: .atomic)
    }
    func writeJSON(_ value: [String: Any], to handle: FileHandle?) throws {
        guard let handle else { throw NSError(domain: "MissingFileHandle", code: 1) }
        var d = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]); d.append(10); try handle.write(contentsOf: d)
    }
    func event(_ type: String, extra: [String: Any] = [:]) {
        guard session != nil else { return }
        var d = extra; d["event"] = type; d["t"] = MachClock.now()
        do { try writeJSON(d, to: events) } catch { fail(error) }
    }
    @objc func toggle() {
        if running { pause(); return }
        guard !finished, failure == nil else { return }
        guard hz > 100, MachClock.now()-lastGyro < 0.4 else {
            startProblem = "Нет потока датчика · подожди и нажми пробел"; view.needsDisplay = true; return
        }
        startProblem = nil
        do { if session == nil { try createSession() } } catch { fail(error); return }
        running = true; showCue(); event("resume"); window.makeFirstResponder(view)
    }
    func showCue() {
        cue = MachClock.now(); readyAt = cue+1.1; pending = nil
        event("cue", extra: ["trial_index": index, "zone": target, "row": target/cols, "col": target%cols, "lid_angle": reader.lidAngle as Any? ?? NSNull()])
        subtitle.stringValue = "Один лёгкий тап в центр оранжевой клетки → дождись следующей. Не нажимай мышью. ⌫ — отмена последнего."
        view.needsDisplay = true
    }
    @objc func pause() {
        guard running else { return }; event("pause", extra: ["discarded_pending": pending != nil]); running = false; pending = nil; view.needsDisplay = true
        try? raw?.synchronize(); try? events?.synchronize(); try? trials?.synchronize()
    }
    @objc func undo() {
        guard !trialIDs.isEmpty, !finished else { return }
        pause(); let id = trialIDs.removeLast(); index -= 1
        event("invalidate_trial", extra: ["trial_id": id, "reason": "user_undo"])
        subtitle.stringValue = "Последняя запись исключена. Пробел — повторить её."
    }
    @objc func finish() {
        guard !finished else { return }; pause(); event("finish", extra: ["accepted_trials": index, "planned_trials": sequence.count])
        do { try raw?.synchronize(); try events?.synchronize(); try trials?.synchronize() } catch { fail(error); return }
        finished = true; startButton.isEnabled = false; view.needsDisplay = true
        subtitle.stringValue = session.map { "Сохранено: \($0.path)" } ?? "Сеанс не начинался; постукивания не записаны."
    }
    func fail(_ error: Error) {
        failure = error.localizedDescription; running = false; pending = nil
        subtitle?.stringValue = "Ошибка записи: \(error.localizedDescription). Сбор остановлен."; view?.needsDisplay = true
    }
    func tick() {
        let now = MachClock.now(), batch = inbox.drain()
        ring.append(contentsOf: batch); ring.removeAll { $0.t < now-2 }
        if session != nil && !finished && failure == nil {
            var data = Data(capacity: batch.count*40)
            for s in batch {
                var kind = UInt64(s.kind).littleEndian; withUnsafeBytes(of: &kind) { data.append(contentsOf: $0) }
                for value in [s.t, s.v.x, s.v.y, s.v.z] { var bits = value.bitPattern.littleEndian; withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) } }
            }
            do { try raw?.write(contentsOf: data) } catch { fail(error) }
        }
        for s in batch where s.kind == 1 {
            lastGyro = s.t; gyroCounter += 1
            baseline += (s.v-baseline)*0.038
            let v = s.v-baseline
            let mag = sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
            let threshold = max(0.65, noise*7)
            if !running || (pending == nil && mag < threshold) { noise = 0.999*noise+0.001*min(mag, 0.5) }
            if running && pending == nil && s.t > readyAt && s.t-lastInput > 0.6 && mag > threshold {
                pending = s.t; pendingLid = reader.lidAngle
                event("tap_trigger", extra: ["sample_t": s.t, "threshold_dps": threshold, "peak_trigger_dps": mag]); view.needsDisplay = true
            }
        }
        if rateStart == 0 { rateStart = now }
        if now-rateStart >= 1 {
            hz = Int(Double(gyroCounter)/(now-rateStart)); gyroCounter = 0; rateStart = now
            if session != nil && !finished { event("context", extra: ["lid_angle": reader.lidAngle as Any? ?? NSNull(), "gyro_hz": hz, "running": running]) }
        }
        if running && now-lastGyro > 0.7 { pause(); startProblem = "Датчик отключился · пробел — повторить"; view.needsDisplay = true }
        if running, let a = reader.lidAngle, let b = initialLid, abs(a-b) > 4 {
            pause(); startProblem = "Верни угол крышки \(b)° · пробел — продолжить"; view.needsDisplay = true }
        if running, let t = pending, now > t+0.4 { completeTap(t) }
        let angle = reader.lidAngle.map { "\($0)°" } ?? "—"
        status.stringValue = "\(cols)×\(rows) = \(cols*rows) клеток · \(index) / \(cols*rows*repetitions) · гироскоп \(hz) Гц · крышка \(angle)" + (running ? " · ЗАПИСЬ" : (finished ? " · СОХРАНЕНО" : " · ПАУЗА"))
    }
    func completeTap(_ t: Double) {
        let samples = ring.filter { $0.t >= t-0.15 && $0.t <= t+0.35 }
        let gyro = samples.filter { $0.kind == 1 }, accel = samples.filter { $0.kind == 0 }
        guard gyro.count > 100, accel.count > 100, let end = gyro.last, end.t >= t+0.32 else {
            pending = nil; readyAt = MachClock.now()+1.2; event("rejected", extra: ["reason": "incomplete_sensor_window"]); return
        }
        let grid = gridRect(), cell = grid.width/CGFloat(cols), height = grid.height/CGFloat(rows), col = target%cols, row = target/cols
        let center = NSPoint(x: grid.minX+(CGFloat(col)+0.5)*cell, y: grid.minY+(CGFloat(row)+0.5)*height)
        let global = window.convertPoint(toScreen: view.convert(center, to: nil))
        let id = UUID().uuidString
        let trial: [String: Any] = ["trial_id": id, "trial_index": index, "zone": target, "row": row, "col": col,
            "x_normalized": (Double(col)+0.5)/Double(cols), "y_normalized": (Double(row)+0.5)/Double(rows),
            "target_view_xy_top_left": [center.x, center.y], "grid_view_rect_top_left": [grid.minX, grid.minY, grid.width, grid.height],
            "target_global_appkit_xy_bottom_left": [global.x, global.y], "screen_backing_scale": window.backingScaleFactor,
            "cue_t": cue, "trigger_t": t, "lid_angle": pendingLid as Any? ?? NSNull(), "label_source": "instructed_target_center",
            "accel_g": accel.map { $0.json }, "gyro_dps": gyro.map { $0.json }]
        do { try writeJSON(trial, to: trials); try trials?.synchronize() } catch { fail(error); return }
        trialIDs.append(id); index += 1; pending = nil
        if index >= sequence.count { finish() } else { showCue() }
    }
    func windowWillClose(_ notification: Notification) { finish(); NSApp.terminate(nil) }
    func applicationWillTerminate(_ notification: Notification) { finish() }
}

if CommandLine.arguments.contains("--probe") {
    let r = IMUReader(), inbox = Inbox(); r.onSample = { k,t,v in inbox.add(Sample(kind:k == .gyro ? 1:0,t:t,v:v)) }; r.start()
    Thread.sleep(forTimeInterval: 3)
    let s = inbox.drain()
    print("gyro=\(s.filter{$0.kind == 1}.count) accel=\(s.filter{$0.kind == 0}.count) lid=\(r.lidAngle.map(String.init) ?? "nil") error=\(r.error ?? "none")")
} else {
    let app = NSApplication.shared
    let delegate = Collector(); app.delegate = delegate; app.run()
}
