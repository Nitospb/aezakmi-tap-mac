import Foundation
import IOKit
import IOKit.hid

/// Seconds on the same clock as IOHID report timestamps (mach_absolute_time).
enum MachClock {
    static let tick: Double = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return Double(tb.numer) / Double(tb.denom) / 1e9
    }()

    static func now() -> Double { Double(mach_absolute_time()) * tick }
}

/// Reads the undocumented Bosch IMU and lid-angle sensor on Apple Silicon MacBooks.
///
/// The IMU sits on the logic board (in the base) and appears as `AppleSPUHIDDevice` services:
/// usage page 0xFF00, usage 3 = accelerometer, 9 = gyroscope. Each 22-byte report carries
/// int32 Q16.16 x/y/z at offsets 6/10/14 (g and deg/s). The sensors only stream after their
/// `AppleSPUHIDDriver` services are woken. The lid sensor is usage page 0x20 / usage 0x8A.
final class IMUReader {
    enum Kind { case accel, gyro }

    /// Called on the sensor thread for every report.
    var onSample: ((Kind, Double, SIMD3<Double>) -> Void)?

    private let lock = NSLock()
    private var _lidAngle: Int?
    private var _error: String?
    private(set) var foundAccel = false
    private(set) var foundGyro = false
    private(set) var foundLid = false

    var lidAngle: Int? { lock.withLock { _lidAngle } }
    var error: String? { lock.withLock { _error } }
    /// How many times the watchdog (or a request) reopened the sensors.
    var reopenCount: Int { lock.withLock { _reopens } }

    private var _reopens = 0
    private var _reopenRequested = false
    private var _stallRequested = false
    private var lastSample = 0.0  // sensor thread only
    private var devices: [IOHIDDevice] = []

    /// Ask the sensor thread to close and reopen the devices (used by tests; the watchdog does it on stalls).
    func requestReopen() { lock.withLock { _reopenRequested = true } }

    /// Test hook: make the watchdog believe no data has arrived for 10 s (it should then reopen by itself).
    func simulateStall() { lock.withLock { _stallRequested = true } }

    private var thread: Thread?
    private var keep: [AnyObject] = []
    private var reportBuffers: [UnsafeMutablePointer<UInt8>] = []
    private var lidDevice: IOHIDDevice?

    func start() {
        guard thread == nil else { return }
        let t = Thread { [weak self] in self?.run() }
        t.name = "LidTap.IMU"
        t.qualityOfService = .userInteractive
        t.start()
        thread = t
    }

    private func run() {
        do {
            try open()
        } catch {
            lock.withLock { _error = "\(error)" }
            Log.sensor.error("\(String(describing: error), privacy: .public)")
        }
        var buf = [UInt8](repeating: 0, count: 8)
        var lastReopen = MachClock.now()
        lastSample = MachClock.now()
        while true {
            // With no devices scheduled the run loop returns at once, so pace the loop ourselves.
            if devices.isEmpty { Thread.sleep(forTimeInterval: 0.05) } else { CFRunLoopRunInMode(.defaultMode, 0.05, false) }
            // Watchdog: reports can stop after sleep/wake or if another process powers the sensor down.
            let now = MachClock.now()
            if lock.withLock({ () -> Bool in defer { _stallRequested = false }; return _stallRequested }) {
                lastSample = now - 10
                lastReopen = now - 10
            }
            let requested = lock.withLock { () -> Bool in defer { _reopenRequested = false }; return _reopenRequested }
            if requested || (now - lastSample > 2.5 && now - lastReopen > 5) {
                let reason = requested ? "requested" : "no data for \(Int(now - lastSample)) s"
                Log.sensor.notice("Reopening motion sensor (\(reason, privacy: .public))")
                lastReopen = now
                reopen()
            }
            if let lid = lidDevice {
                var len: CFIndex = 8
                if IOHIDDeviceGetReport(lid, kIOHIDReportTypeFeature, 1, &buf, &len) == kIOReturnSuccess, len >= 3 {
                    let angle = Int(buf[1]) | Int(buf[2]) << 8
                    lock.withLock { _lidAngle = angle }
                }
            }
        }
    }

    private final class CallbackBox {
        weak var reader: IMUReader?
        let kind: Kind
        init(reader: IMUReader, kind: Kind) { self.reader = reader; self.kind = kind }
    }

    private struct OpenError: Error, CustomStringConvertible {
        let description: String
    }

    private func open() throws {
        for driver in services("AppleSPUHIDDriver") {
            for (key, value) in [("SensorPropertyReportingState", 1), ("SensorPropertyPowerState", 1), ("ReportInterval", 1000)] {
                IORegistryEntrySetCFProperty(driver, key as CFString, NSNumber(value: value))
            }
            IOObjectRelease(driver)
        }
        let loop = CFRunLoopGetCurrent()
        for service in services("AppleSPUHIDDevice") {
            defer { IOObjectRelease(service) }
            let page = intProperty(service, "PrimaryUsagePage"), usage = intProperty(service, "PrimaryUsage")
            if page == 0x20, usage == 0x8A, lidDevice == nil {
                if let dev = IOHIDDeviceCreate(kCFAllocatorDefault, service),
                   IOHIDDeviceOpen(dev, IOOptionBits(kIOHIDOptionsTypeNone)) == kIOReturnSuccess {
                    lidDevice = dev
                    foundLid = true
                }
                continue
            }
            guard page == 0xFF00, usage == 3 || usage == 9, intProperty(service, "MaxInputReportSize") == 22 else { continue }
            let kind: Kind = usage == 3 ? .accel : .gyro
            if (kind == .accel && foundAccel) || (kind == .gyro && foundGyro) { continue }
            guard let dev = IOHIDDeviceCreate(kCFAllocatorDefault, service),
                  IOHIDDeviceOpen(dev, IOOptionBits(kIOHIDOptionsTypeNone)) == kIOReturnSuccess else { continue }
            let report = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
            let box = CallbackBox(reader: self, kind: kind)
            keep += [box]
            devices.append(dev)
            reportBuffers.append(report)
            IOHIDDeviceRegisterInputReportWithTimeStampCallback(dev, report, 4096, { context, _, _, _, _, report, length, timestamp in
                guard let context, length == 22 else { return }
                let box = Unmanaged<CallbackBox>.fromOpaque(context).takeUnretainedValue()
                func q16(_ offset: Int) -> Double {
                    var v: Int32 = 0
                    memcpy(&v, report + offset, 4)
                    return Double(Int32(littleEndian: v)) / 65536.0
                }
                guard let reader = box.reader else { return }
                let t = Double(timestamp) * MachClock.tick
                reader.lastSample = t
                reader.onSample?(box.kind, t, SIMD3(q16(6), q16(10), q16(14)))
            }, Unmanaged.passUnretained(box).toOpaque())
            IOHIDDeviceScheduleWithRunLoop(dev, loop!, CFRunLoopMode.defaultMode.rawValue)
            if kind == .accel { foundAccel = true } else { foundGyro = true }
        }
        if !foundAccel && !foundGyro {
            throw OpenError(description: "No motion sensor found. LidTap needs an M2-or-later MacBook, or an M1 Pro/Max.")
        }
    }

    private func reopen() {
        let loop = CFRunLoopGetCurrent()!
        for dev in devices {
            IOHIDDeviceRegisterInputReportWithTimeStampCallback(dev, reportBuffers[0], 4096, nil, nil)
            IOHIDDeviceUnscheduleFromRunLoop(dev, loop, CFRunLoopMode.defaultMode.rawValue)
            IOHIDDeviceClose(dev, IOOptionBits(kIOHIDOptionsTypeNone))
        }
        if let lid = lidDevice { IOHIDDeviceClose(lid, IOOptionBits(kIOHIDOptionsTypeNone)) }
        devices = []
        keep = []
        lidDevice = nil
        foundAccel = false
        foundGyro = false
        foundLid = false
        lock.withLock { _reopens += 1 }
        // Report buffers stay allocated: the old devices may still reference them until they are released.
        do {
            try open()
            lock.withLock { _error = nil }
        } catch {
            lock.withLock { _error = "\(error)" }
        }
    }

    private func services(_ className: String) -> [io_service_t] {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching(className), &iterator) == KERN_SUCCESS else { return [] }
        defer { IOObjectRelease(iterator) }
        var out: [io_service_t] = []
        while true {
            let s = IOIteratorNext(iterator)
            if s == 0 { return out }
            out.append(s)
        }
    }

    private func intProperty(_ service: io_service_t, _ key: String) -> Int? {
        guard let value = IORegistryEntryCreateCFProperty(service, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() else { return nil }
        return (value as? NSNumber)?.intValue
    }
}
