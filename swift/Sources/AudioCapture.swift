import ScreenCaptureKit
import CoreMedia
import AVFAudio
import CoreGraphics
import Foundation
import AppKit
import CoreAudio
import AudioToolbox
import IOKit.pwr_mgt

// MARK: - Constants

/// Minimum peak amplitude to consider microphone audio "loud" (silence timeout).
private let kMicLoudThreshold: Float = 1e-2
/// Minimum peak amplitude to consider system audio "loud" (silence timeout).
private let kSystemLoudThreshold: Float = 1e-4
/// System audio capture and merge output sample rate (mono float32).
private let kSystemAudioSampleRate: Double = 24000

/// Compute peak amplitude across all channels of float audio data.
func computePeakLevel(in channelData: UnsafePointer<UnsafeMutablePointer<Float>>,
               channels: Int, frames: Int) -> Float {
    var peak: Float = 0
    for ch in 0..<channels {
        for i in 0..<frames {
            let v = abs(channelData[ch][i])
            if v > peak { peak = v }
        }
    }
    return peak
}

// MARK: - Mic Capture via AVAudioEngine

class MicCapture {
    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private(set) var startHostTime: UInt64 = 0

    // Fixed processing format of the output file. Every tap buffer is normalised to
    // this before writing, so a mid-stream format change cannot break the writes the
    // way a fixed AVAudioFile did before (Meet flipping the shared input into
    // 3-channel voice-processing mode made every write fail with -50, killing the mic).
    private var fileFormat: AVAudioFormat?
    private var audioConverter: AVAudioConverter?

    // Set when a write/convert failure has been logged; reset on the next successful write
    // so a persistent failure logs once instead of once per tap buffer (~10+/sec). Touched
    // only from writeBuffer on the tap (render) thread, so it needs no locking.
    private var loggedWriteError = false

    // Remembered so an explicitly named device can be re-applied after a route change.
    private var micDeviceName: String?

    // Re-bind the input + reinstall the tap when the audio route changes. The token is
    // kept so stop() can deregister it. isStopping guards a notification already queued
    // on .main behind stop(): the observer is delivered on .main and stop() runs on
    // .main (the SIGINT/SIGTERM sources), so they serialise and the flag alone is enough.
    private var configObserver: NSObjectProtocol?
    private var isStopping = false

    // Mute support — guarded by os_unfair_lock (tap callback is on AVAudioEngine thread)
    private var _isMuted = false
    private var _muteLock = os_unfair_lock_s()

    // Level tracking for silence timeout (mirrors SystemAudioCapture pattern)
    private var _lastLoudTime: UInt64 = DispatchTime.now().uptimeNanoseconds
    private var _lastLoudTimeLock = os_unfair_lock_s()

    var lastLoudTime: UInt64 {
        os_unfair_lock_lock(&_lastLoudTimeLock)
        defer { os_unfair_lock_unlock(&_lastLoudTimeLock) }
        return _lastLoudTime
    }

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
        micDeviceName = deviceName
        try applyInputDevice()

        let format = engine.inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            throw MicError.noInputAvailable
        }

        let url = URL(fileURLWithPath: outputPath)
        let file = try AVAudioFile(forWriting: url,
                                   settings: format.settings,
                                   commonFormat: .pcmFormatFloat32,
                                   interleaved: true)
        audioFile = file
        fileFormat = file.processingFormat

        guard installMicTap() else {
            throw MicError.noInputAvailable
        }

        // Earbuds plugged in mid-recording, or a conferencing app reconfiguring the
        // shared input format, posts this notification (the engine stops itself first).
        // Rebind + restart so the mic keeps recording instead of silently dying.
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: .main
        ) { [weak self] _ in self?.handleConfigChange() }

        try engine.start()
        fputs("Recording microphone audio to \(outputPath)...\n", stderr)
    }

    /// Re-apply the explicitly named input device, if one was requested. When no device
    /// was named the engine follows the system default input — which is exactly what
    /// makes "switch to earbuds mid-recording" work, so we deliberately do nothing.
    private func applyInputDevice() throws {
        guard let name = micDeviceName else { return }
        // audioUnit can be nil while the node is in flux during a route change; treat
        // that as "cannot set device" rather than force-unwrapping into a crash.
        guard let audioUnit = engine.inputNode.audioUnit else {
            throw MicError.cannotSetDevice(name)
        }
        let deviceID = try findInputDevice(named: name)
        var id = deviceID
        let err = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0,
            &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        if err != noErr {
            throw MicError.cannotSetDevice(name)
        }
    }

    /// Install the capture tap using the input node's current format. Returns false if
    /// no input is available yet (sample rate 0), e.g. mid-route-change.
    @discardableResult
    private func installMicTap() -> Bool {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { return false }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, time in
            guard let self else { return }
            if self.startHostTime == 0 {
                self.startHostTime = time.hostTime
            }
            let muted = self.isMuted
            if muted, let channelData = buffer.floatChannelData {
                let channels = Int(buffer.format.channelCount)
                let frames = Int(buffer.frameLength)
                for ch in 0..<channels {
                    memset(channelData[ch], 0, frames * MemoryLayout<Float>.size)
                }
            }
            // Track peak level for silence timeout (muted mic = silence)
            if !muted, let channelData = buffer.floatChannelData {
                let peak = computePeakLevel(in: channelData,
                                     channels: Int(buffer.format.channelCount),
                                     frames: Int(buffer.frameLength))
                if peak > kMicLoudThreshold {
                    os_unfair_lock_lock(&self._lastLoudTimeLock)
                    self._lastLoudTime = DispatchTime.now().uptimeNanoseconds
                    os_unfair_lock_unlock(&self._lastLoudTimeLock)
                }
            }
            self.writeBuffer(buffer)
        }
        return true
    }

    /// Write a tap buffer to the file, converting to the file's fixed format whenever
    /// the incoming format differs (Meet voice-processing mode, or a new device with a
    /// different sample rate or channel count). The fast path — unchanged format —
    /// writes directly, exactly as the original code did. On a sample-rate change the
    /// converter is rebuilt; any sub-buffer of tail samples left inside the old
    /// converter is dropped (sub-10ms, inaudible for transcription).
    private func writeBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let audioFile, let fileFormat else { return }
        do {
            if buffer.format == fileFormat {
                try audioFile.write(from: buffer)
                loggedWriteError = false
                return
            }
            if audioConverter?.inputFormat != buffer.format {
                audioConverter = AVAudioConverter(from: buffer.format, to: fileFormat)
            }
            guard let converter = audioConverter else {
                logWriteErrorOnce("Mic convert error: converter unavailable for format \(buffer.format)")
                return
            }
            let ratio = fileFormat.sampleRate / buffer.format.sampleRate
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
            guard let outBuffer = AVAudioPCMBuffer(pcmFormat: fileFormat,
                                                   frameCapacity: capacity) else { return }
            var fed = false
            var convError: NSError?
            let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
                if fed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return buffer
            }
            if status == .error || convError != nil {
                logWriteErrorOnce("Mic convert error: \(convError?.description ?? "unknown")")
                return
            }
            if outBuffer.frameLength > 0 {
                try audioFile.write(from: outBuffer)
                loggedWriteError = false
            }
        } catch {
            logWriteErrorOnce("Mic write error: \(error)")
        }
    }

    /// Log a recurring write/convert failure only once until the next successful write, so a
    /// persistent bad state (e.g. an unconvertible format) doesn't flood stderr at the tap
    /// rate. Reset by writeBuffer on every successful write. Render-thread only — no locking.
    private func logWriteErrorOnce(_ message: String) {
        guard !loggedWriteError else { return }
        loggedWriteError = true
        fputs(message + "\n", stderr)
    }

    /// Rebind the input and reinstall the tap after an audio route change. Runs on
    /// .main, so it never overlaps stop(); removeTap quiesces the render thread before
    /// we touch the tap, and the file/converter stay valid across the swap.
    ///
    /// Known limitation: the mic file is written continuously with no silence padding,
    /// so if the input drops out for a moment during a device switch (e.g. plugging in
    /// earbuds), the post-switch mic audio is appended immediately and the mic track
    /// drifts slightly earlier relative to the system track. Acceptable for
    /// transcription/diarization, and far better than the previous behaviour (losing
    /// all post-switch mic audio). The common conferencing-app case reconfigures only
    /// the *format*, not the device, so the converter absorbs it with no gap at all.
    private func handleConfigChange() {
        guard !isStopping else { return }
        fputs("Mic reconfigure: audio route changed; rebinding microphone input.\n", stderr)
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        do {
            try applyInputDevice()  // re-pin a named device; no-op when following default
        } catch {
            fputs("Mic reconfigure: \(error) — falling back to default input.\n", stderr)
        }
        reinstallAndStart()
    }

    /// Reinstall the tap and restart the engine. If the input is not ready yet
    /// (sampleRate 0 while a route is still settling), retry on .main a bounded number
    /// of times rather than relying on another notification arriving — a single
    /// transient zero-rate change must not kill the mic for the rest of the session.
    private func reinstallAndStart(attempt: Int = 0) {
        guard !isStopping else { return }
        guard installMicTap() else {
            let maxAttempts = 20  // ~5s at 0.25s spacing
            if attempt < maxAttempts {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                    self?.reinstallAndStart(attempt: attempt + 1)
                }
            } else {
                fputs("Mic reconfigure: input did not return after \(maxAttempts) retries; mic stopped.\n", stderr)
            }
            return
        }
        do {
            try engine.start()
        } catch {
            fputs("Mic reconfigure: failed to restart engine: \(error)\n", stderr)
        }
    }

    func stop() {
        isStopping = true
        if let token = configObserver {
            NotificationCenter.default.removeObserver(token)
            configObserver = nil
        }
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

    // Silence timeout auto-stop
    var silenceTimeout: TimeInterval = 0  // seconds; 0 = disabled
    var onSilenceTimeout: (() -> Void)?
    var micCapture: MicCapture?  // checked by silence timer
    private var lastLoudTime: UInt64 = DispatchTime.now().uptimeNanoseconds
    private var lastLoudTimeLock = os_unfair_lock_s()
    private var silenceTimer: DispatchSourceTimer?

    // Power assertion to prevent display sleep during capture
    private var powerAssertionID: IOPMAssertionID = IOPMAssertionID(kIOPMNullAssertionID)

    // Picker continuation
    private var startContinuation: CheckedContinuation<Void, Error>?

    init(outputPath: String) {
        self.outputPath = outputPath
        super.init()
    }

    var captureModeAll: Bool = false

    func start() async throws {
        if captureModeAll {
            let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                throw CaptureError.noDisplay
            }
            let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
            try await self.beginCapture(with: filter)
        } else {
            let picker = SCContentSharingPicker.shared
            var pickerConfig = SCContentSharingPickerConfiguration()
            pickerConfig.allowedPickerModes = [.singleWindow, .singleDisplay, .singleApplication]
            picker.defaultConfiguration = pickerConfig
            picker.add(self)
            picker.isActive = true
            picker.present()

            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                self.startContinuation = continuation
            }
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
        config.sampleRate = Int(kSystemAudioSampleRate)
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false

        // Create AVAudioFile for WAV output (interleaved to avoid CoreAudio warning)
        let fileFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: kSystemAudioSampleRate, channels: 1, interleaved: true)!
        let audioFile = try AVAudioFile(forWriting: URL(fileURLWithPath: outputPath),
                                         settings: fileFormat.settings,
                                         commonFormat: .pcmFormatFloat32,
                                         interleaved: true)
        self.audioFile = audioFile

        // Create and start stream
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: captureQueue)

        // Initialize last-loud time before starting capture (no lock needed — callbacks haven't started)
        if silenceTimeout > 0 {
            lastLoudTime = DispatchTime.now().uptimeNanoseconds
        }

        try await stream.startCapture()
        self.stream = stream

        IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventUserIdleDisplaySleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "ownscribe is recording audio" as CFString,
            &powerAssertionID)

        fputs("Recording system audio to \(outputPath)... Press Ctrl+C to stop.\n", stderr)

        // Start silence timeout timer if configured.
        // Checks every 1s whether both system audio and mic (if active) have been
        // quiet longer than silenceTimeout. Uses the most recent "loud" timestamp
        // from either source so that activity on either channel prevents auto-stop.
        if silenceTimeout > 0 {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 1, repeating: 1.0)
            timer.setEventHandler { [weak self] in
                guard let self else { return }
                let now = DispatchTime.now().uptimeNanoseconds
                os_unfair_lock_lock(&self.lastLoudTimeLock)
                var effectiveLastLoud = self.lastLoudTime
                os_unfair_lock_unlock(&self.lastLoudTimeLock)
                // If mic is active, use the more recent of the two
                if let mic = self.micCapture {
                    let micLastLoud = mic.lastLoudTime
                    if micLastLoud > effectiveLastLoud {
                        effectiveLastLoud = micLastLoud
                    }
                }
                guard now >= effectiveLastLoud else { return }
                let elapsed = Double(now - effectiveLastLoud) / 1_000_000_000.0
                if elapsed > self.silenceTimeout {
                    fputs("[SILENCE_TIMEOUT]\n", stderr)
                    self.silenceTimer?.cancel()
                    self.silenceTimer = nil
                    self.onSilenceTimeout?()
                }
            }
            timer.resume()
            silenceTimer = timer
        }
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
                if audioConverter?.inputFormat != sampleFormat {
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

        // Peak detection on the pre-conversion buffer — floatChannelData returns nil
        // for interleaved buffers, so we must use the original SCK buffer
        let bufferPeak: Float = pcmBuffer.floatChannelData.map {
            computePeakLevel(in: $0, channels: Int(pcmBuffer.format.channelCount), frames: Int(pcmBuffer.frameLength))
        } ?? 0.0
        if bufferPeak > self.peakLevel { self.peakLevel = bufferPeak }

        // Update last loud time for silence timeout
        if bufferPeak > kSystemLoudThreshold {
            os_unfair_lock_lock(&lastLoudTimeLock)
            lastLoudTime = DispatchTime.now().uptimeNanoseconds
            os_unfair_lock_unlock(&lastLoudTimeLock)
        }

        // Check for silence after ~3 seconds of data
        if !silenceChecked && Double(totalFrames) > kSystemAudioSampleRate * 3 {
            silenceChecked = true
            if peakLevel < 1e-6 {
                silenceWarned = true
                if micCapture != nil {
                    fputs("[SILENCE_WARNING] System audio is silent (mic is still recording). No system audio sources detected.\n", stderr)
                } else {
                    fputs("[SILENCE_WARNING] Audio data received but peak level is near zero (\(peakLevel)). Audio may be silent.\n", stderr)
                    fputs("Check: System Settings > Privacy & Security > Screen Recording — enable your terminal app.\n", stderr)
                }
            }
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream error: \(error)\n", stderr)
    }

    // MARK: - Stop

    func stop() {
        silenceTimer?.cancel()
        silenceTimer = nil

        if powerAssertionID != IOPMAssertionID(kIOPMNullAssertionID) {
            IOPMAssertionRelease(powerAssertionID)
            powerAssertionID = IOPMAssertionID(kIOPMNullAssertionID)
        }

        let sem = DispatchSemaphore(value: 0)
        Task.detached { [stream] in
            try? await stream?.stopCapture()
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 2)

        // AVAudioFile finalizes on close
        audioFile = nil

        let seconds = Double(totalFrames) / kSystemAudioSampleRate
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

    let outputSampleRate: Double = kSystemAudioSampleRate
    let outputChannels: AVAudioChannelCount = 1

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
            let writeOffset = max(0, -sysFrameInFile)
            let readStart = max(0, sysFrameInFile)
            if readStart < systemLength && writeOffset < Int64(framesToProcess) {
                let maxRead = Int64(framesToProcess) - writeOffset
                let sysReadCount = AVAudioFrameCount(min(maxRead, systemLength - readStart))
                if sysReadCount > 0 {
                    systemFile.framePosition = AVAudioFramePosition(readStart)
                    if let sysBuf = AVAudioPCMBuffer(pcmFormat: systemFile.processingFormat, frameCapacity: sysReadCount) {
                        try systemFile.read(into: sysBuf, frameCount: sysReadCount)
                        let sysData = sysBuf.floatChannelData!
                        let sysCh = Int(sysBuf.format.channelCount)
                        let wo = Int(writeOffset)
                        for i in 0..<Int(sysBuf.frameLength) {
                            var mix: Float = 0
                            for ch in 0..<sysCh { mix += sysData[ch][i] }
                            outPtr[wo + i] += mix / Float(sysCh)
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
        ownscribe-audio capture --output FILE [--mic] [--mic-device NAME] [--silence-timeout N]
        ownscribe-audio list-apps
        ownscribe-audio list-devices

    OPTIONS:
        --output, -o FILE    Output WAV file path (required for capture)
        --mic                Also capture microphone input
        --mic-device NAME    Use specific mic input device (implies --mic)
        --capture-mode-all   Capture all system audio without showing the source picker
        --silence-timeout N  Auto-stop after N seconds of silence (0 = disabled)
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
        var captureModeAll = false
        var silenceTimeout: TimeInterval = 0

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
            case "--capture-mode-all":
                captureModeAll = true
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
            case "--silence-timeout":
                i += 1
                guard i < args.count, let val = TimeInterval(args[i]) else {
                    fputs("Error: --silence-timeout requires a number of seconds\n", stderr)
                    exit(1)
                }
                if val < 0 {
                    fputs("Error: --silence-timeout must be zero (disabled) or a positive number of seconds\n", stderr)
                    exit(1)
                }
                silenceTimeout = val
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
        capture.captureModeAll = captureModeAll
        capture.silenceTimeout = silenceTimeout
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
            capture.micCapture = mic
        }

        // Shared shutdown logic for SIGINT, SIGTERM, and silence timeout
        let shutdown: () -> Void = {
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

        capture.onSilenceTimeout = shutdown

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
        sigintSource.setEventHandler { shutdown() }
        sigintSource.resume()

        let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        signal(SIGTERM, SIG_IGN)
        sigtermSource.setEventHandler { shutdown() }
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
