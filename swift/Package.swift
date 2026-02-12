// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "notetaker-audio",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "notetaker-audio",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
            ]
        )
    ]
)
