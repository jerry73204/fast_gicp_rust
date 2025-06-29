# Development Guide

## Overview

This guide covers building, testing, and developing the Fast GICP Rust wrapper. All common operations are automated through the provided Makefile for consistency and ease of use.

## Prerequisites

### System Requirements

**Operating System**: Linux (Ubuntu 20.04+ recommended)

**Core Dependencies**:
- **Rust**: 1.70 or later (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **CMake**: 3.15 or later (`sudo apt install cmake`)
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **pkg-config**: For dependency detection (`sudo apt install pkg-config`)

**Point Cloud Dependencies**:
- **PCL**: 1.8 or later (`sudo apt install libpcl-dev`)
- **Eigen3**: 3.3 or later (`sudo apt install libeigen3-dev`)

**Optional Dependencies**:
- **OpenMP**: For CPU parallelization (`sudo apt install libomp-dev`)
- **CUDA**: 11.0+ for GPU acceleration (`nvidia/cuda` Docker image or system install)

### Rust Toolchain Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install required components
rustup component add clippy rustfmt
cargo install cargo-nextest  # For enhanced testing
```

### Project Setup

```bash
# Clone repository
git clone https://github.com/jerry73204/fast_gicp_rust
cd fast_gicp_rust

# Initialize submodules
git submodule update --init --recursive

# Prepare vendored C++ sources (required)
./scripts/prepare-vendor.sh
```

## Build System

### Makefile Targets

The project uses a comprehensive Makefile for all development tasks:

```bash
make help          # Show all available targets
```

### Building

**Basic Build (CPU only)**:
```bash
make build         # Build with default features (OpenMP enabled)
```

**CUDA Build**:
```bash
make build-cuda    # Build with CUDA support
```

**Manual Build Commands**:
```bash
# CPU build
cargo build --release

# CUDA build  
cargo build --release --features cuda

# Debug build
cargo build

# Check compilation without building
cargo check
cargo check --features cuda
```

### Feature Flags

| Feature | Description | Dependencies | Command |
|---------|-------------|--------------|---------|
| `default` | OpenMP parallelization | OpenMP | `cargo build` |
| `cuda` | GPU acceleration | CUDA 11.0+ | `cargo build --features cuda` |
| `docs-only` | Documentation builds | None | `cargo build --features docs-only --no-default-features` |

## Testing

### Comprehensive Testing

**Run All Tests**:
```bash
make test          # Runs CPU and CUDA tests (skips docs-only)
```

**Individual Test Categories**:
```bash
# CPU tests only
cargo nextest run --all-targets --no-fail-fast

# CUDA tests only  
cargo nextest run --features cuda --all-targets --no-fail-fast

# Documentation compilation check
make check-docs-only
```

### Specific Test Commands

**Unit Tests**:
```bash
cargo test --lib                    # Library unit tests
cargo test --lib --features cuda    # CUDA unit tests
```

**Integration Tests**:
```bash
cargo test --test integration       # Basic integration tests
cargo test --test cuda_error_handling --features cuda  # CUDA error tests
cargo test --test ndt_cuda --features cuda            # NDT CUDA tests
```

**Documentation Tests**:
```bash
cargo test --doc                    # Run doc tests
cargo test --doc --features cuda    # CUDA doc tests
```

### Performance Testing

**Benchmarks**:
```bash
cargo bench                         # Run all benchmarks
cargo bench --features cuda         # Include CUDA benchmarks
```

**Manual Performance Testing**:
```bash
cargo run --example basic_registration --release
cargo run --example cuda_registration --features cuda --release
```

## Code Quality

### Linting

**Full Linting Suite**:
```bash
make lint          # Runs formatting check, clippy for all configurations
```

**Individual Linting Commands**:
```bash
# Format checking
cargo +nightly fmt --all -- --check

# Clippy (regular build)
cargo clippy --all-targets -- -D warnings

# Clippy (CUDA build)  
cargo clippy --all-targets --features cuda -- -D warnings

# Clippy (docs-only build)
cargo clippy --lib --features docs-only --no-default-features -- -D warnings
```

### Code Formatting

**Apply Formatting**:
```bash
make format        # Format all code
```

**Manual Formatting**:
```bash
cargo +nightly fmt --all
```

## Documentation

### Generate Documentation

**Standard Documentation**:
```bash
make docs          # Generate docs using non-CUDA stub
```

**Manual Documentation Commands**:
```bash
# Local documentation
cargo doc --open

# docs.rs simulation (non-CUDA)
cargo doc --features docs-only --no-default-features --no-deps

# docs.rs simulation (CUDA)  
cargo doc --features "docs-only cuda" --no-default-features --no-deps
```

### Documentation Validation

**Validate docs.rs Compatibility**:
```bash
./validate_docs.sh  # Comprehensive docs.rs simulation testing
```

## Stub System Development

### Stub Generation

The stub system enables docs.rs compatibility by providing pre-generated FFI stubs.

**Regenerate Stubs**:
```bash
make generate-stubs    # Generate both CUDA and non-CUDA stubs
```

**Verify Stub Correctness**:
```bash
make verify-stubs      # Validate stub generation and filtering
```

**Test Stub Compilation**:
```bash
make test-stubs        # Test both stub variants compile
```

**Complete Stub Workflow**:
```bash
make update-stubs      # Generate, verify, and test stubs in one command
```

### Manual Stub Commands

```bash
# Generate stubs manually
cd fast-gicp-sys && cargo build --features bindgen

# Test stub compilation
cargo check --features docs-only --no-default-features --lib
cargo check --features "docs-only cuda" --no-default-features --lib

# Compare stub variants
diff fast-gicp-sys/src/generated/stub.rs fast-gicp-sys/src/generated/stub_cuda.rs
```

## Development Workflow

### Typical Development Cycle

1. **Make Changes**: Edit source code
2. **Quick Check**: `cargo check` for fast compilation feedback
3. **Test**: `make test` to run comprehensive test suite
4. **Lint**: `make lint` to ensure code quality
5. **Build**: `make build` for final verification

### Adding New Features

**For New FFI Functions**:
1. Add function to `#[cxx::bridge]` in `fast-gicp-sys/src/lib.rs`
2. Implement C++ wrapper in `fast-gicp-sys/src/wrapper.cpp`
3. Add Rust API wrapper in `fast-gicp/src/`
4. Update stubs: `make update-stubs`
5. Add tests and documentation
6. Run full test suite: `make test`

**For New Algorithms**:
1. Follow FFI function process above
2. Create builder pattern implementation
3. Add comprehensive tests including accuracy validation
4. Create usage examples
5. Update documentation

### Debugging

**Build Issues**:
```bash
# Verbose build output
RUST_LOG=debug cargo build

# CMake debugging
cd fast-gicp-sys && RUST_LOG=debug cargo build

# Check system dependencies
pkg-config --list-all | grep -E "(pcl|eigen3)"
```

**Runtime Issues**:
```bash
# Run with debug output
RUST_LOG=debug cargo run --example basic_registration

# Debug CUDA issues
CUDA_LAUNCH_BLOCKING=1 cargo run --example cuda_registration --features cuda
```

## Continuous Integration

### Local CI Simulation

**Full CI Pipeline**:
```bash
# Simulate complete CI pipeline
make clean
make build
make build-cuda  
make test
make lint
make docs
./validate_docs.sh
```

### Pre-commit Checks

**Recommended Pre-commit Routine**:
```bash
#!/bin/bash
# Save as .git/hooks/pre-commit and chmod +x

set -e
echo "Running pre-commit checks..."

make lint
make test
make check-docs-only

echo "✅ All pre-commit checks passed"
```

## Environment Variables

### Build Configuration

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_ROOT` | Override CUDA installation path | `/usr/local/cuda-12.0` |
| `PCL_ROOT` | Override PCL installation path | `/usr/local` |
| `EIGEN3_ROOT` | Override Eigen3 path | `/usr/include/eigen3` |
| `RUST_LOG` | Enable debug logging | `debug` |
| `CARGO_FEATURE_DOCS_ONLY` | Force docs-only mode | `1` |

### Runtime Configuration

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU devices | `0,1` |
| `OMP_NUM_THREADS` | OpenMP thread count | `8` |
| `CUDA_LAUNCH_BLOCKING` | Synchronous CUDA calls | `1` |

## Troubleshooting

### Common Build Issues

**CMake Cannot Find PCL**:
```bash
# Install PCL development packages
sudo apt install libpcl-dev

# Or specify custom path
export PCL_ROOT=/usr/local
```

**CUDA Not Detected**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Set CUDA path if needed
export CUDA_ROOT=/usr/local/cuda
```

**Linking Errors**:
```bash
# Update library cache
sudo ldconfig

# Check library dependencies
ldd target/debug/deps/libfast_gicp_sys-*.so
```

### Performance Issues

**Slow Compilation**:
```bash
# Use faster linker
sudo apt install lld
export RUSTFLAGS="-C link-arg=-fuse-ld=lld"

# Parallel compilation
export CARGO_BUILD_JOBS=8
```

**Runtime Performance**:
```bash
# Enable optimizations
cargo build --release

# Use profile-guided optimization
export RUSTFLAGS="-C profile-generate=/tmp/pgo-data"
cargo build --release
# Run representative workload
export RUSTFLAGS="-C profile-use=/tmp/pgo-data"
cargo build --release
```

## Contribution Guidelines

### Code Style

- Follow standard Rust conventions
- Use `cargo fmt` for consistent formatting
- Ensure `cargo clippy` produces no warnings
- Write comprehensive tests for new features
- Include documentation with examples

### Testing Requirements

- All new features must include unit tests
- Integration tests for end-to-end workflows
- Performance tests for algorithms
- Documentation tests for all public APIs
- CUDA features require both CPU and GPU test variants

### Documentation Standards

- Complete rustdoc for all public APIs
- Include working examples in documentation
- Update design documents for significant changes
- Maintain changelog for version history