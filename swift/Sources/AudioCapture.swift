import ScreenCaptureKit
import CoreMedia
import AVFAudio
import CoreGraphics
import Foundation
import AppKit
import CoreAudio
import AudioToolbox

// MARK: - Mic Capture via AVAudioEngine

class MicCapture {
    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private(set) var startHostTime: UInt64 = 0

    // Mute support — guarded by os_unfair_lock (tap callback is on AVAudioEngine thread)
    private var _isMuted = false
    private var _muteLock = os_unfair_lock_s()

    var isMuted: Bool {
        os_unfair_lock_lock(&_muteLock)
        defer { os_unfair_lock_unlock(&_muteLock) }
        return _isMuted
    }

    func toggleMute() {
        os_unfair_lock_lock(&_muteLock)
        _isMuted.toggle()
        let muted = _isMuted
        os_unfair_lock_unlock(&_muteLock)
        fputs(muted ? "[MIC_MUTED]\n" : "[MIC_UNMUTED]\n", stderr)
    }

    func start(outputPath: String, deviceName: String?) throws {
        let input = engine.inputNode

        // If deviceName specified, find and set the audio device
        if let name = deviceName {
            let deviceID = try findInputDevice(named: name)
            var id = deviceID
            let err = AudioUnitSetProperty(
                input.audioUnit!,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global, 0,
                &id, UInt32(MemoryLayout<AudioDeviceID>.size))
            if err != noErr {
                throw MicError.cannotSetDevice(name)
            }
        }

        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            throw MicError.noInputAvailable
        }

        let url = URL(fileURLWithPath: outputPath)
        audioFile = try AVAudioFile(forWriting: url,
                                     settings: format.settings,
                                     commonFormat: .pcmFormatFloat32,
                                     interleaved: true)

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, time in
            guard let self else { return }
            if self.startHostTime == 0 {
                self.startHostTime = time.hostTime
            }
            if self.isMuted, let channelData = buffer.floatChannelData {
                let channels = Int(buffer.format.channelCount)
                let frames = Int(buffer.frameLength)
                for ch in 0..<channels {
                    memset(channelData[ch], 0, frames * MemoryLayout<Float>.size)
                }
            }
            try? self.audioFile?.write(from: buffer)
        }
        try engine.start()
        fputs("Recording microphone audio to \(outputPath)...\n", stderr)
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioFile = nil
    }

    enum MicError: Error, CustomStringConvertible {
        case deviceNotFound(String)
        case cannotSetDevice(String)
        case noInputAvailable

        var description: String {
            switch self {
            case .deviceNotFound(let n): return "Input device not found: \(n)"
            case .cannotSetDevice(let n): return "Cannot set input device: \(n)"
            case .noInputAvailable: return "No audio input available (sample rate is 0)"
            }
        }
    }
}

/// Find an input audio device by name, returning its AudioDeviceID.
func findInputDevice(named name: String) throws -> AudioDeviceID {
    var propAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)

    var dataSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                    &propAddress, 0, nil, &dataSize)
    let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var devices = [AudioDeviceID](repeating: 0, count: deviceCount)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &propAddress, 0, nil, &dataSize, &devices)

    for deviceID in devices {
        // Get device name
        var nameAddr = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cfNameRef: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddr, 0, nil, &nameSize, &cfNameRef)
        let deviceName = cfNameRef?.takeRetainedValue() as String? ?? "(unknown)"

        // Check input channel count
        var inputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var bufSize: UInt32 = 0
        AudioObjectGetPropertyDataSize(deviceID, &inputAddr, 0, nil, &bufSize)
        if bufSize > 0 {
            let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { bufferList.deallocate() }
            AudioObjectGetPropertyData(deviceID, &inputAddr, 0, nil, &bufSize, bufferList)
            let inputChannels = UnsafeMutableAudioBufferListPointer(bufferList)
                .reduce(0) { $0 + Int($1.mNumberChannels) }
            if inputChannels > 0 && deviceName.lowercased().contains(name.lowercased()) {
                return deviceID
            }
        }
    }
    throw MicCapture.MicError.deviceNotFound(name)
}

// MARK: - System Audio Capture via ScreenCaptureKit

class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate, SCContentSharingPickerObserver {
    private var stream: SCStream?
    private var audioFile: AVAudioFile?
    private var audioConverter: AVAudioConverter?
    private let captureQueue = DispatchQueue(label: "com.ownscribe.audioCapture", qos: .userInitiated)

    private let outputPath: String

    // Timestamp for sync alignment
    private(set) var startHostTime: UInt64 = 0

    // Silence detection
    private var peakLevel: Float = 0.0
    private var totalFrames: Int64 = 0
    private var silenceChecked: Bool = false
    private var silenceWarned: Bool = false

    // Picker continuation
    private var startContinuation: CheckedContinuation<Void, Error>?

    init(outputPath: String) {
        self.outputPath = outputPath
        super.init()
    }

    func start() async throws {
        // Configure and show the content sharing picker
        let picker = SCContentSharingPicker.shared
        var pickerConfig = SCContentSharingPickerConfiguration()
        pickerConfig.allowedPickerModes = [.singleWindow, .singleDisplay, .singleApplication]
        picker.defaultConfiguration = pickerConfig
        picker.add(self)
        picker.isActive = true
        picker.present()

        // Suspend until the picker delegate fires
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.startContinuation = continuation
        }
    }

    // MARK: - SCContentSharingPickerObserver

    func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter, for stream: SCStream?) {
        Task {
            do {
                try await self.beginCapture(with: filter)
                self.startContinuation?.resume()
                self.startContinuation = nil
            } catch {
                self.startContinuation?.resume(throwing: error)
                self.startContinuation = nil
            }
        }
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        fputs("Content picker cancelled.\n", stderr)
        exit(0)
    }

    func contentSharingPickerStartDidFailWithError(_ error: Error) {
        self.startContinuation?.resume(throwing: error)
        self.startContinuation = nil
    }

    // MARK: - Begin Capture

    private func beginCapture(with filter: SCContentFilter) async throws {
        // Configure stream (audio only, minimal video)
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false

        // Create AVAudioFile for WAV output (interleaved to avoid CoreAudio warning)
        let fileFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: true)!
        let audioFile = try AVAudioFile(forWriting: URL(fileURLWithPath: outputPath),
                                         settings: fileFormat.settings,
                                         commonFormat: .pcmFormatFloat32,
                                         interleaved: true)
        self.audioFile = audioFile

        // Create and start stream
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: captureQueue)
        try await stream.startCapture()
        self.stream = stream

        fputs("Recording system audio to \(outputPath)... Press Ctrl+C to stop.\n", stderr)
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard let audioFile else { return }
        guard CMSampleBufferGetNumSamples(sampleBuffer) > 0 else { return }

        // Capture start host time from first audio buffer
        if startHostTime == 0 {
            startHostTime = mach_absolute_time()
        }

        // Get format from sample buffer
        guard let formatDesc = sampleBuffer.formatDescription,
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }
        guard let sampleFormat = AVAudioFormat(streamDescription: asbd) else { return }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: sampleFormat, frameCapacity: frameCount) else { return }
        pcmBuffer.frameLength = frameCount

        // Copy audio data into PCM buffer
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )
        guard status == noErr else { return }

        // Convert non-interleaved → interleaved if needed, then write
        do {
            if sampleFormat.isInterleaved {
                try audioFile.write(from: pcmBuffer)
            } else {
                // Lazily create converter matching this source format
                if audioConverter == nil || audioConverter!.inputFormat != sampleFormat {
                    let interleavedFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                                       sampleRate: sampleFormat.sampleRate,
                                                       channels: sampleFormat.channelCount,
                                                       interleaved: true)!
                    audioConverter = AVAudioConverter(from: sampleFormat, to: interleavedFmt)
                }
                if let converter = audioConverter {
                    let outFmt = converter.outputFormat
                    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: frameCount) else { return }
                    try converter.convert(to: outBuffer, from: pcmBuffer)
                    try audioFile.write(from: outBuffer)
                }
            }
        } catch {
            fputs("Write error: \(error)\n", stderr)
        }

        totalFrames += Int64(frameCount)

        // Peak detection on float channel data
        if let channelData = pcmBuffer.floatChannelData {
            let channelCount = Int(sampleFormat.channelCount)
            for ch in 0..<channelCount {
                let samples = channelData[ch]
                for i in 0..<Int(frameCount) {
                    let absVal = abs(samples[i])
                    if absVal > peakLevel { peakLevel = absVal }
                }
            }
        }

        // Check for silence after ~3 seconds of data
        if !silenceChecked && totalFrames > 48000 * 3 {
            silenceChecked = true
            if peakLevel < 1e-6 {
                silenceWarned = true
                fputs("[SILENCE_WARNING] Audio data received but peak level is near zero (\(peakLevel)). Audio may be silent.\n", stderr)
                fputs("Check: System Settings > Privacy & Security > Screen Recording — enable your terminal app.\n", stderr)
            }
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream error: \(error)\n", stderr)
    }

    // MARK: - Stop

    func stop() {
        let sem = DispatchSemaphore(value: 0)
        Task.detached { [stream] in
            try? await stream?.stopCapture()
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 2)

        // AVAudioFile finalizes on close
        audioFile = nil

        let seconds = Double(totalFrames) / 48000.0
        fputs("Saved \(outputPath) (\(String(format: "%.1f", seconds)) seconds, peak=\(String(format: "%.6f", peakLevel)))\n", stderr)
        if totalFrames > 0 && peakLevel < 1e-6 {
            fputs("[SILENCE_WARNING] Recording appears silent. Check Screen Recording permission.\n", stderr)
        }
    }

    enum CaptureError: Error, CustomStringConvertible {
        case cannotOpenFile(String)
        case noDisplay

        var description: String {
            switch self {
            case .cannotOpenFile(let p): return "Cannot open file: \(p)"
            case .noDisplay: return "No display found"
            }
        }
    }
}

// MARK: - List apps

func listAudioApps() {
    print("Running apps:")
    for app in NSWorkspace.shared.runningApplications {
        if app.activationPolicy == .regular, let name = app.localizedName {
            print("  PID \(app.processIdentifier): \(name)")
        }
    }
}

// MARK: - List input devices

func listInputDevices() {
    // Get default input device
    var defaultAddr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var defaultDevice: AudioDeviceID = 0
    var defaultSize = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &defaultAddr, 0, nil, &defaultSize, &defaultDevice)

    // Enumerate all devices
    var propAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)

    var dataSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                    &propAddress, 0, nil, &dataSize)
    let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var devices = [AudioDeviceID](repeating: 0, count: deviceCount)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                &propAddress, 0, nil, &dataSize, &devices)

    print("Input devices:")
    for deviceID in devices {
        // Get device name
        var nameAddr = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cfNameRef: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddr, 0, nil, &nameSize, &cfNameRef)
        let name = cfNameRef?.takeRetainedValue() as String? ?? "(unknown)"

        // Check input channel count
        var inputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var bufSize: UInt32 = 0
        AudioObjectGetPropertyDataSize(deviceID, &inputAddr, 0, nil, &bufSize)
        guard bufSize > 0 else { continue }

        let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        defer { bufferList.deallocate() }
        AudioObjectGetPropertyData(deviceID, &inputAddr, 0, nil, &bufSize, bufferList)
        let inputChannels = UnsafeMutableAudioBufferListPointer(bufferList)
            .reduce(0) { $0 + Int($1.mNumberChannels) }

        if inputChannels > 0 {
            let suffix = (deviceID == defaultDevice) ? " (default)" : ""
            print("  \(name)\(suffix)")
        }
    }
}

// MARK: - Merge audio files with timestamp alignment

func mergeAudioFiles(systemPath: String, micPath: String,
                     systemStartHostTime: UInt64, micStartHostTime: UInt64,
                     outputPath: String) throws {
    // A standard WAV file header (RIFF + fmt + data chunk header) is 44 bytes.
    // Files at or below this size contain no audio frames.
    let wavHeaderSize = 44
    let fm = FileManager.default
    let systemFileSize = (try? fm.attributesOfItem(atPath: systemPath)[.size] as? Int) ?? 0
    let micFileSize = (try? fm.attributesOfItem(atPath: micPath)[.size] as? Int) ?? 0

    // Both empty — clean up temp files and let the caller handle it
    if systemFileSize <= wavHeaderSize && micFileSize <= wavHeaderSize {
        try? fm.removeItem(atPath: systemPath)
        try? fm.removeItem(atPath: micPath)
        return
    }

    // Mic empty but system has data — just rename system file
    if micFileSize <= wavHeaderSize && systemFileSize > wavHeaderSize {
        try? fm.removeItem(atPath: micPath)
        try fm.moveItem(atPath: systemPath, toPath: outputPath)
        fputs("Merged audio saved to \(outputPath) (system only, no mic audio)\n", stderr)
        return
    }

    // Open files — system is optional (may be empty when only mic captured audio)
    let systemFile: AVAudioFile? = systemFileSize > wavHeaderSize
        ? try AVAudioFile(forReading: URL(fileURLWithPath: systemPath)) : nil
    let micFile = try AVAudioFile(forReading: URL(fileURLWithPath: micPath))

    let outputSampleRate: Double = 48000
    let outputChannels: AVAudioChannelCount = 2

    // Compute offset in seconds between the two start times using mach_timebase_info
    let offsetFrames: Int64
    if systemFile != nil {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let ticksToNanos = Double(timebase.numer) / Double(timebase.denom)
        let systemStartNanos = Double(systemStartHostTime) * ticksToNanos
        let micStartNanos = Double(micStartHostTime) * ticksToNanos
        let offsetSeconds = (micStartNanos - systemStartNanos) / 1_000_000_000.0
        offsetFrames = Int64(offsetSeconds * outputSampleRate)
    } else {
        offsetFrames = 0
    }
    let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: outputSampleRate,
                                      channels: outputChannels,
                                      interleaved: true)!
    let outputFile = try AVAudioFile(forWriting: URL(fileURLWithPath: outputPath),
                                      settings: outputFormat.settings,
                                      commonFormat: .pcmFormatFloat32,
                                      interleaved: true)

    // Mic converter using callback API (handles sample rate + channel conversion)
    let micFormat = micFile.processingFormat
    let micConverter = AVAudioConverter(from: micFormat, to: outputFormat)!
    let micRate = micFormat.sampleRate

    let chunkSize: AVAudioFrameCount = 8192
    let systemLength = systemFile.map { Int64($0.length) } ?? 0
    // Mic file length converted to output-rate frames
    let micLengthOutput = Int64(Double(micFile.length) * outputSampleRate / micRate)

    // Calculate total output length accounting for offset
    let systemEndFrame = (offsetFrames >= 0) ? systemLength : systemLength + (-offsetFrames)
    let micEndFrame = (offsetFrames >= 0) ? micLengthOutput + offsetFrames : micLengthOutput
    let totalOutputFrames = max(systemEndFrame, micEndFrame)

    // Mic region in output timeline
    let micOutputStart: Int64 = (offsetFrames >= 0) ? offsetFrames : 0
    let micOutputEnd: Int64 = micOutputStart + micLengthOutput
    var micDone = false

    var outputFrame: Int64 = 0

    while outputFrame < totalOutputFrames {
        let framesToProcess = AVAudioFrameCount(min(Int64(chunkSize), totalOutputFrames - outputFrame))
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: framesToProcess) else { break }
        outBuffer.frameLength = framesToProcess

        // Zero the output buffer
        let outPtr = outBuffer.floatChannelData![0]
        for i in 0..<Int(framesToProcess * outputChannels) {
            outPtr[i] = 0
        }

        // Read and mix system audio (manual interleave from non-interleaved processingFormat)
        if let systemFile = systemFile {
            let sysFrameInFile = (offsetFrames >= 0) ? outputFrame : outputFrame + offsetFrames
            if sysFrameInFile >= 0 && sysFrameInFile < systemLength {
                let sysReadCount = AVAudioFrameCount(min(Int64(framesToProcess), systemLength - sysFrameInFile))
                if sysReadCount > 0 {
                    systemFile.framePosition = AVAudioFramePosition(sysFrameInFile)
                    if let sysBuf = AVAudioPCMBuffer(pcmFormat: systemFile.processingFormat, frameCapacity: sysReadCount) {
                        try systemFile.read(into: sysBuf, frameCount: sysReadCount)
                        let sysData = sysBuf.floatChannelData!
                        for i in 0..<Int(sysBuf.frameLength) {
                            outPtr[i * 2] += sysData[0][i]
                            outPtr[i * 2 + 1] += sysData[1][i]
                        }
                    }
                }
            }
        }

        // Read and mix mic audio using callback-based converter
        let chunkEnd = outputFrame + Int64(framesToProcess)
        if !micDone && chunkEnd > micOutputStart && outputFrame < micOutputEnd {
            let overlapStart = max(outputFrame, micOutputStart)
            let overlapEnd = min(chunkEnd, micOutputEnd)
            let micFramesNeeded = AVAudioFrameCount(overlapEnd - overlapStart)

            if micFramesNeeded > 0 {
                guard let micOutBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: micFramesNeeded) else { break }

                var convError: NSError?
                let status = micConverter.convert(to: micOutBuf, error: &convError) { inNumberOfPackets, outStatus in
                    let remaining = AVAudioFrameCount(micFile.length - micFile.framePosition)
                    if remaining == 0 {
                        outStatus.pointee = .endOfStream
                        return nil
                    }
                    let toRead = min(inNumberOfPackets, remaining)
                    guard let buf = AVAudioPCMBuffer(pcmFormat: micFormat, frameCapacity: toRead) else {
                        outStatus.pointee = .endOfStream
                        return nil
                    }
                    do {
                        try micFile.read(into: buf, frameCount: toRead)
                        outStatus.pointee = .haveData
                        return buf
                    } catch {
                        outStatus.pointee = .endOfStream
                        return nil
                    }
                }

                if status == .endOfStream {
                    micDone = true
                }

                // Mix converted mic audio into output at correct position
                if micOutBuf.frameLength > 0 {
                    let offsetInChunk = Int(overlapStart - outputFrame)
                    let srcPtr = micOutBuf.floatChannelData![0]
                    let count = Int(micOutBuf.frameLength * outputChannels)
                    for i in 0..<count {
                        outPtr[offsetInChunk * Int(outputChannels) + i] += srcPtr[i]
                    }
                }
            }
        }

        // Clamp to [-1, 1]
        let totalSamples = Int(framesToProcess * outputChannels)
        for i in 0..<totalSamples {
            outPtr[i] = max(-1.0, min(1.0, outPtr[i]))
        }

        try outputFile.write(from: outBuffer)
        outputFrame += Int64(framesToProcess)
    }

    // Delete temp files
    try? FileManager.default.removeItem(atPath: systemPath)
    try? FileManager.default.removeItem(atPath: micPath)

    let detail = systemFile == nil ? " (mic only, no system audio)" : ""
    fputs("Merged audio saved to \(outputPath)\(detail)\n", stderr)
}

// MARK: - Main

func printUsage() {
    fputs("""
    ownscribe-audio — system audio capture helper

    USAGE:
        ownscribe-audio capture --output FILE [--mic] [--mic-device NAME]
        ownscribe-audio list-apps
        ownscribe-audio list-devices

    OPTIONS:
        --output, -o FILE    Output WAV file path (required for capture)
        --mic                Also capture microphone input
        --mic-device NAME    Use specific mic input device (implies --mic)
        --help, -h           Show this help

    SUBCOMMANDS:
        capture              Record audio to a WAV file
        list-apps            Show running applications
        list-devices         Show available audio input devices

    """, stderr)
}

func main() {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        printUsage()
        exit(1)
    }

    let command = args[1]

    switch command {
    case "list-apps":
        listAudioApps()

    case "list-devices":
        listInputDevices()

    case "capture":
        // Initialize NSApplication so the picker GUI can render
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)

        var outputPath: String?
        var enableMic = false
        var micDeviceName: String?

        var i = 2
        while i < args.count {
            switch args[i] {
            case "--output", "-o":
                i += 1
                guard i < args.count else {
                    fputs("Error: --output requires a file path\n", stderr)
                    exit(1)
                }
                outputPath = args[i]
            case "--mic":
                enableMic = true
            case "--mic-device":
                i += 1
                guard i < args.count else {
                    fputs("Error: --mic-device requires a device name\n", stderr)
                    exit(1)
                }
                micDeviceName = args[i]
                enableMic = true  // --mic-device implies --mic
            default:
                fputs("Unknown option: \(args[i])\n", stderr)
                printUsage()
                exit(1)
            }
            i += 1
        }

        guard let output = outputPath else {
            fputs("Error: --output is required\n", stderr)
            printUsage()
            exit(1)
        }

        // Determine paths: if mic enabled, use temp files then merge
        let systemPath = enableMic ? output + ".sys.tmp.wav" : output
        let micPath = output + ".mic.tmp.wav"

        let capture = SystemAudioCapture(outputPath: systemPath)
        var micCapture: MicCapture?

        if enableMic {
            let mic = MicCapture()
            do {
                try mic.start(outputPath: micPath, deviceName: micDeviceName)
            } catch {
                fputs("Error starting mic capture: \(error)\n", stderr)
                exit(1)
            }
            micCapture = mic
        }

        // Toggle mic mute on SIGUSR1 (sent by Python wrapper)
        var _sigusr1Source: DispatchSourceSignal?  // retained to keep source alive
        if let mic = micCapture {
            signal(SIGUSR1, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
            src.setEventHandler { mic.toggleMute() }
            src.resume()
            _sigusr1Source = src
        }
        _ = _sigusr1Source

        // Handle Ctrl+C gracefully
        let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        signal(SIGINT, SIG_IGN)
        sigintSource.setEventHandler {
            capture.stop()
            if let mic = micCapture {
                mic.stop()
                // Merge the two files
                do {
                    try mergeAudioFiles(
                        systemPath: systemPath,
                        micPath: micPath,
                        systemStartHostTime: capture.startHostTime,
                        micStartHostTime: mic.startHostTime,
                        outputPath: output)
                } catch {
                    fputs("Error merging audio: \(error)\n", stderr)
                }
            }
            exit(0)
        }
        sigintSource.resume()

        let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        signal(SIGTERM, SIG_IGN)
        sigtermSource.setEventHandler {
            capture.stop()
            if let mic = micCapture {
                mic.stop()
                do {
                    try mergeAudioFiles(
                        systemPath: systemPath,
                        micPath: micPath,
                        systemStartHostTime: capture.startHostTime,
                        micStartHostTime: mic.startHostTime,
                        outputPath: output)
                } catch {
                    fputs("Error merging audio: \(error)\n", stderr)
                }
            }
            exit(0)
        }
        sigtermSource.resume()

        Task {
            do {
                try await capture.start()
            } catch {
                fputs("Error: \(error)\n", stderr)
                exit(1)
            }
        }

        app.run()

    case "--help", "-h":
        printUsage()

    default:
        fputs("Unknown command: \(command)\n", stderr)
        printUsage()
        exit(1)
    }
}

main()
