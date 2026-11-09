import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox

/// PCM capture with per-buffer host-clock anchors; recording is local and pause-gated.
final class MicrophoneRecorder {
    private let engine = AVAudioEngine()
    private let queue = DispatchQueue(label: "local.tapcollector.audio-disk")
    private let lock = NSLock()
    private var enabled = false, installed = false
    private var queued = 0, _error: String?, _lastEnd = 0.0
    private var file: AVAudioFile?, index: FileHandle?, pcmSync: FileHandle?
    private var chunks: [[String: Any]] = [] // queue only
    private(set) var sampleRate = 0.0
    private(set) var channelCount = 0
    private(set) var deviceName = ""
    private(set) var inputLatency = 0.0
    var error: String? { lock.withLock { _error } }
    var lastEnd: Double { lock.withLock { _lastEnd } }
    var isRunning: Bool { engine.isRunning }
    static var permission: AVAuthorizationStatus { AVCaptureDevice.authorizationStatus(for: .audio) }
    static func requestPermission(_ completion: @escaping (Bool)->Void) {
        AVCaptureDevice.requestAccess(for: .audio) { ok in DispatchQueue.main.async { completion(ok) } }
    }
    private func audioError(_ message: String)->NSError { NSError(domain:"Microphone",code:1,userInfo:[NSLocalizedDescriptionKey:message]) }
    private func builtInDevice() throws -> AudioDeviceID {
        var address=AudioObjectPropertyAddress(mSelector:kAudioHardwarePropertyDevices,mScope:kAudioObjectPropertyScopeGlobal,mElement:kAudioObjectPropertyElementMain)
        var size:UInt32=0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),&address,0,nil,&size)==noErr else { throw audioError("Не удалось найти микрофон MacBook") }
        var devices=[AudioDeviceID](repeating:0,count:Int(size)/MemoryLayout<AudioDeviceID>.size)
        let result=devices.withUnsafeMutableBytes { AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),&address,0,nil,&size,$0.baseAddress!) }
        guard result==noErr else { throw audioError("Не удалось перечислить микрофоны") }
        for device in devices {
            var transport:UInt32=0, bytes=UInt32(MemoryLayout<UInt32>.size)
            address=AudioObjectPropertyAddress(mSelector:kAudioDevicePropertyTransportType,mScope:kAudioObjectPropertyScopeGlobal,mElement:kAudioObjectPropertyElementMain)
            guard AudioObjectGetPropertyData(device,&address,0,nil,&bytes,&transport)==noErr, transport==kAudioDeviceTransportTypeBuiltIn else { continue }
            address=AudioObjectPropertyAddress(mSelector:kAudioDevicePropertyStreams,mScope:kAudioDevicePropertyScopeInput,mElement:kAudioObjectPropertyElementMain)
            var streams:UInt32=0
            guard AudioObjectGetPropertyDataSize(device,&address,0,nil,&streams)==noErr, streams>0 else { continue }
            address=AudioObjectPropertyAddress(mSelector:kAudioObjectPropertyName,mScope:kAudioObjectPropertyScopeGlobal,mElement:kAudioObjectPropertyElementMain)
            var name:Unmanaged<CFString>?;bytes=UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            if AudioObjectGetPropertyData(device,&address,0,nil,&bytes,&name)==noErr, let name { deviceName=name.takeRetainedValue() as String }
            return device
        }
        throw audioError("Встроенный микрофон не найден; внешний автоматически не подставляется")
    }
    func prepare(folder:URL) throws {
        guard Self.permission == .authorized else { throw audioError("Нет разрешения на микрофон") }
        let node=engine.inputNode
        var device=try builtInDevice()
        guard let unit=node.audioUnit, AudioUnitSetProperty(unit,kAudioOutputUnitProperty_CurrentDevice,kAudioUnitScope_Global,0,&device,UInt32(MemoryLayout<AudioDeviceID>.size))==noErr else { throw audioError("Не удалось выбрать встроенный микрофон") }
        let format=node.outputFormat(forBus:0)
        guard format.sampleRate>0, format.channelCount>0, format.commonFormat == .pcmFormatFloat32, !format.isInterleaved else { throw audioError("Неподдерживаемый формат микрофона") }
        sampleRate=format.sampleRate;channelCount=Int(format.channelCount);inputLatency=node.presentationLatency
        file=try AVAudioFile(forWriting:folder.appendingPathComponent("microphone.caf"),settings:format.settings,commonFormat:.pcmFormatFloat32,interleaved:false)
        pcmSync=try FileHandle(forWritingTo:folder.appendingPathComponent("microphone.caf"))
        let url=folder.appendingPathComponent("audio-chunks.jsonl")
        guard FileManager.default.createFile(atPath:url.path,contents:nil) else { throw audioError("Не удалось создать индекс аудио") }
        index=try FileHandle(forWritingTo:url)
        node.installTap(onBus:0,bufferSize:512,format:format) { [weak self] buffer,time in self?.receive(buffer,time) }
        installed=true;engine.prepare()
    }
    func resume() throws {
        guard installed else { throw audioError("Микрофон не подготовлен") }
        lock.withLock { enabled=true }
        do { try engine.start() } catch { lock.withLock { enabled=false };throw error }
    }
    func pause() {
        lock.withLock { enabled=false };engine.pause();flush()
    }
    func flush() {
        queue.sync {
            do {try pcmSync?.synchronize();try index?.synchronize()} catch {lock.withLock{_error=error.localizedDescription}}
        }
    }
    func close() { pause();engine.stop();if installed {engine.inputNode.removeTap(onBus:0);installed=false};queue.sync {file=nil;try? pcmSync?.synchronize();try? pcmSync?.close();pcmSync=nil;try? index?.close();index=nil} }
    private func receive(_ source:AVAudioPCMBuffer,_ time:AVAudioTime) {
        let accepted=lock.withLock { ()->Bool in guard enabled,_error==nil else{return false};guard queued<64 else{_error="Запись звука не успевает на диск";return false};queued+=1;return true }
        guard accepted else{return}
        guard time.isHostTimeValid,let src=source.floatChannelData,let copy=AVAudioPCMBuffer(pcmFormat:source.format,frameCapacity:source.frameLength),let dst=copy.floatChannelData else {lock.withLock{queued-=1;_error="У аудиобуфера нет PCM или метки hostTime"};return}
        copy.frameLength=source.frameLength
        for c in 0..<Int(source.format.channelCount) { memcpy(dst[c],src[c],Int(source.frameLength)*MemoryLayout<Float>.size) }
        let t=Double(time.hostTime)*MachClock.tick,rate=source.format.sampleRate
        queue.async { [self] in
            defer{lock.withLock{queued-=1}}
            do {
                guard let file,let index else {throw audioError("Файлы аудио закрыты")}
                let offset=file.framePosition,n=Int(copy.frameLength)
                try file.write(from:copy)
                let row:[String:Any] = ["t":t,"frames":n,"sample_rate":rate,"channels":channelCount,"file_frame_start":offset,"audio_sample_time":time.isSampleTimeValid ? time.sampleTime as Any : NSNull()]
                var data=try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]);data.append(10);try index.write(contentsOf:data)
                chunks.append(row);lock.withLock{_lastEnd=t+Double(n)/rate}
            } catch {lock.withLock{_error=error.localizedDescription}}
        }
    }
    func references(from lo:Double,to hi:Double)->[[String:Any]] {
        queue.sync {
            chunks.compactMap { row in
                let t=row["t"] as! Double, n=row["frames"] as! Int, rate=row["sample_rate"] as! Double
                let a=max(0,Int(ceil((lo-t)*rate))), b=min(n,Int(ceil((hi-t)*rate)))
                guard b>a else{return nil}
                return ["file_frame_start":(row["file_frame_start"] as! Int64)+Int64(a),"frames":b-a,"t":t+Double(a)/rate,"sample_rate":rate]
            }
        }
    }
    func covers(_ refs:[[String:Any]],from lo:Double,to hi:Double)->Bool {
        guard let first=refs.first,let last=refs.last else{return false}
        let tolerance=2.0/sampleRate
        guard (first["t"] as! Double)<=lo+tolerance,(last["t"] as! Double)+Double(last["frames"] as! Int)/sampleRate>=hi-tolerance else{return false}
        var end=lo
        for r in refs {let t=r["t"] as! Double;if t-end>tolerance{return false};end=t+Double(r["frames"] as! Int)/sampleRate}
        return true
    }
}
