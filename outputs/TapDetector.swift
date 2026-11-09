import Foundation

/// Detects short motion impulses, with hysteresis and a quiet interval between onsets.
/// Magnitudes describe movement of the laptop, not pressure or force on the display.
struct TapDetector {
    private var gyroBase: SIMD3<Double>?
    private var accelBase: SIMD3<Double>?
    private(set) var gyroMagnitude = 0.0
    private(set) var accelMagnitude = 0.0
    private var gyroNoise = 0.025
    private var accelNoise = 0.0008
    private var quietSince: Double?
    private var latched = false
    private var lastOnset = -Double.infinity
    var gyroThreshold: Double { max(0.4, gyroNoise * 8) }
    var accelThreshold: Double { max(0.010, accelNoise * 8) }
    var ratio: Double { max(gyroMagnitude / gyroThreshold, accelMagnitude / accelThreshold) }
    func quietDuration(at t: Double) -> Double { quietSince.map { max(0, t-$0) } ?? 0 }

    mutating func process(kind: Int, t: Double, v: SIMD3<Double>, learnNoise: Bool) -> Bool {
        if kind == 1 {
            if gyroBase == nil { gyroBase = v }
            gyroBase! += (v-gyroBase!)*0.038
            let d = v-gyroBase!; gyroMagnitude = sqrt(d.x*d.x+d.y*d.y+d.z*d.z)
            if learnNoise && gyroMagnitude < gyroThreshold * 0.5 {
                gyroNoise = 0.999*gyroNoise + 0.001*gyroMagnitude
            }
        } else {
            if accelBase == nil { accelBase = v }
            accelBase! += (v-accelBase!)*0.038
            let d = v-accelBase!; accelMagnitude = sqrt(d.x*d.x+d.y*d.y+d.z*d.z)
            if learnNoise && accelMagnitude < accelThreshold * 0.5 {
                accelNoise = 0.999*accelNoise + 0.001*accelMagnitude
            }
        }
        if ratio < 0.45 {
            if quietSince == nil { quietSince = t }
            if quietDuration(at: t) >= 0.07 { latched = true }
        } else {
            quietSince = nil
            if ratio >= 1 && latched && t-lastOnset >= 0.16 {
                latched = false; lastOnset = t; return true
            }
        }
        return false
    }
}

struct ImpactStrength {
    let gyroPeak: Double
    let accelPeak: Double
    var relativeIndex: Double { max(gyroPeak/2, accelPeak/0.04) }
    var category: String { relativeIndex < 1 ? "weak" : (relativeIndex < 3 ? "medium" : "strong") }
    var text: String { relativeIndex < 1 ? "слабо" : (relativeIndex < 3 ? "средне" : "сильно") }
}
