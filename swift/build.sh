#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$(dirname "$SCRIPT_DIR")/bin"

mkdir -p "$BIN_DIR"

echo "Building ownscribe-audio..."
swiftc \
    -O \
    -o "$BIN_DIR/ownscribe-audio" \
    -framework ScreenCaptureKit \
    -framework CoreMedia \
    -framework AVFAudio \
    -framework AppKit \
    -framework CoreGraphics \
    -framework CoreAudio \
    -framework AudioToolbox \
    "$SCRIPT_DIR/Sources/AudioCapture.swift"

echo "Built: $BIN_DIR/ownscribe-audio"
