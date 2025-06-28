#!/bin/bash
set -euo pipefail

# Script to generate and save CXX bindings for docs-only mode
# These pre-generated bindings allow building documentation on docs.rs
# without requiring C++ dependencies

echo "Generating CXX bindings for docs-only mode..."

# Change to the project root
cd "$(dirname "$0")/.."

# Ensure we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Must be run from the project root"
    exit 1
fi

# Create the generated directory
mkdir -p fast-gicp-sys/src/generated

# Build with bindgen feature to generate bindings
echo "Building with bindgen feature..."
cd fast-gicp-sys
cargo build --features bindgen

# The bindgen feature in build.rs should have created the bindings
# Check if they were created
if [ ! -f "src/generated/lib.rs.h" ] && [ ! -f "src/generated/lib.rs.cc" ]; then
    echo "Warning: Expected generated files not found. The bindgen feature may need to be implemented."
    echo "Please ensure the build.rs generate_bindings() function creates files in src/generated/"
fi

echo "Bindings generation complete!"
echo ""
echo "Next steps:"
echo "1. Review the generated files in fast-gicp-sys/src/generated/"
echo "2. Commit them to version control with: git add fast-gicp-sys/src/generated"
echo "3. Test docs-only build with: make doc-rs"