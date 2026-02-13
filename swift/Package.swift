// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ownscribe-audio",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "ownscribe-audio",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
            ]
        )
    ]
)
