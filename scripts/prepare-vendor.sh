#!/bin/bash
# Script to prepare vendored C++ sources for the build system
# This script is idempotent and uses rsync to ensure exact synchronization

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENDOR_DIR="$PROJECT_ROOT/fast-gicp-sys/vendor"
VENDOR_TARGET="$VENDOR_DIR/fast_gicp"
FAST_GICP_SOURCE="$PROJECT_ROOT/fast_gicp"

echo "Preparing vendored sources for fast-gicp-sys..."

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "Error: rsync is required but not installed!"
    echo "Please install rsync using your package manager:"
    echo "  Ubuntu/Debian: sudo apt-get install rsync"
    echo "  Fedora/RHEL: sudo dnf install rsync"
    echo "  Arch: sudo pacman -S rsync"
    exit 1
fi

# Check if fast_gicp submodule exists
if [ ! -f "$FAST_GICP_SOURCE/CMakeLists.txt" ]; then
    echo "Error: fast_gicp submodule not found!"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Create vendor directory if it doesn't exist
echo "Creating vendor directory..."
mkdir -p "$VENDOR_TARGET"

# Sync directories using rsync
echo "Synchronizing files..."

# Sync include directory
echo "  - Syncing include files..."
rsync -aq --delete "$FAST_GICP_SOURCE/include/" "$VENDOR_TARGET/include/"

# Sync only the fast_gicp source subdirectory
echo "  - Syncing source files..."
mkdir -p "$VENDOR_TARGET/src"
rsync -aq --delete "$FAST_GICP_SOURCE/src/fast_gicp/" "$VENDOR_TARGET/src/fast_gicp/"

# Sync thirdparty dependencies
echo "  - Syncing thirdparty dependencies..."
rsync -aq --delete \
    --exclude '.git' \
    --exclude '.github' \
    --exclude 'build' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    "$FAST_GICP_SOURCE/thirdparty/" \
    "$VENDOR_TARGET/thirdparty/"

# Copy root-level files
echo "  - Copying build files..."
cp -f "$FAST_GICP_SOURCE/CMakeLists.txt" "$VENDOR_TARGET/"
cp -f "$FAST_GICP_SOURCE/LICENSE" "$VENDOR_TARGET/"

# Clean up any extra files at root level
find "$VENDOR_TARGET" -maxdepth 1 -type f \
    ! -name "CMakeLists.txt" \
    ! -name "LICENSE" \
    ! -name ".vendor_ready" \
    -delete 2>/dev/null || true

# Create vendor metadata
cat > "$VENDOR_DIR/VENDOR_INFO.md" << EOF
# Vendored fast_gicp Sources

This directory contains vendored C++ sources from the fast_gicp library.

## Source Information
- Repository: https://github.com/SMRT-AIST/fast_gicp
- Commit: $(cd "$FAST_GICP_SOURCE" && git rev-parse HEAD)
- Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- License: BSD 3-Clause (see fast_gicp/LICENSE)

## Contents
- fast_gicp/include/: C++ headers
- fast_gicp/src/fast_gicp/: C++ implementation files
- fast_gicp/thirdparty/: Bundled dependencies (Eigen, Sophus, nvbio)
- fast_gicp/CMakeLists.txt: CMake build configuration
- fast_gicp/LICENSE: BSD 3-Clause license

## Notes
These sources are used when building from crates.io or when the git submodule
is not available. The build.rs script automatically detects and uses these
vendored sources when present.
EOF

# Create vendor ready marker
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC")" > "$VENDOR_TARGET/.vendor_ready"

# Report statistics
VENDOR_SIZE=$(du -sh "$VENDOR_DIR" | cut -f1)
FILE_COUNT=$(find "$VENDOR_DIR" -type f | wc -l)

echo ""
echo "Vendor synchronization complete!"
echo "  Total size: $VENDOR_SIZE"
echo "  File count: $FILE_COUNT"
echo "  Location: $VENDOR_DIR"
echo ""
echo "Next steps:"
echo "  - Build the project: cargo build"
echo "  - Run tests: cargo test"