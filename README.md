# Fast GICP Rust

[![Crates.io](https://img.shields.io/crates/v/fast-gicp.svg)](https://crates.io/crates/fast-gicp)
[![Documentation](https://docs.rs/fast-gicp/badge.svg)](https://docs.rs/fast-gicp)

A Rust wrapper for the fast_gicp library, providing efficient 3D point cloud registration algorithms.

## Features

- **Fast GICP**: Generalized Iterative Closest Point algorithm
- **Fast VGICP**: Voxelized variant for large point clouds
- **CUDA Support**: GPU acceleration for VGICP and NDT algorithms
- **Safe API**: Memory-safe Rust bindings with error handling
- **Builder Pattern**: Fluent configuration interface

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fast-gicp = "0.3"

# For CUDA support
fast-gicp = { version = "0.3", features = ["cuda"] }
```

## Basic Usage

```rust
use fast_gicp::{FastGICP, PointCloudXYZ};

// Create point clouds
let source = PointCloudXYZ::from_points(&[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]);

let target = PointCloudXYZ::from_points(&[
    [0.1, 0.0, 0.0],
    [1.1, 0.0, 0.0],
    [0.1, 1.0, 0.0],
]);

// Create and configure algorithm
let gicp = FastGICP::builder()
    .max_iterations(50)
    .transformation_epsilon(1e-6)
    .build()?;

// Perform registration
let result = gicp.align(&source, &target)?;

println!("Final transformation: {:?}", result.final_transformation);
println!("Converged: {}", result.has_converged);
println!("Fitness score: {}", result.fitness_score);
```

## Algorithms

### Fast GICP
```rust
let gicp = FastGICP::builder()
    .max_iterations(100)
    .regularization_method(RegularizationMethod::Frobenius)
    .build()?;
```

### Fast VGICP (for large point clouds)
```rust
let vgicp = FastVGICP::builder()
    .resolution(0.5)
    .voxel_accumulation_mode(VoxelAccumulationMode::Additive)
    .build()?;
```

### CUDA Acceleration
```rust
// Requires "cuda" feature
let cuda_vgicp = FastVGICPCuda::builder()
    .resolution(1.0)
    .neighbor_search_method(NeighborSearchMethod::Direct27)
    .build()?;
```

## System Requirements

- **Rust**: 1.70 or later
- **CMake**: 3.15 or later
- **C++ Compiler**: C++17 support required
- **OpenMP**: Optional, for CPU parallelization
- **CUDA**: 11.0+ required for GPU features

## Documentation

Full API documentation is available on [docs.rs](https://docs.rs/fast-gicp).

## Building from Source

```bash
git clone https://github.com/jerry73204/fast_gicp_rust
cd fast_gicp_rust
git submodule update --init --recursive

# Prepare vendored C++ sources (required)
./scripts/prepare-vendor.sh

cargo build --release
```

### Development Notes

This crate includes pre-generated FFI stub files to support documentation builds on docs.rs where C++ dependencies are not available. The stub generation system maintains two variants:
- `stub.rs`: For non-CUDA builds (excludes CUDA-specific types and functions)
- `stub_cuda.rs`: For CUDA builds (includes all FFI items)

#### Conditional Documentation Tests

Starting from version 0.3.0, documentation tests are conditionally compiled:
- In regular builds: Doc tests execute normally to ensure examples work correctly
- In docs-only builds: Doc tests are marked as `no_run` to prevent execution with stub implementations

This is achieved using Rust's `cfg_attr` feature:
```rust
#[cfg_attr(feature = "docs-only", doc = "```no_run")]
#[cfg_attr(not(feature = "docs-only"), doc = "```")]
```

When modifying the FFI interface:

```bash
# Regenerate both CUDA and non-CUDA stubs after FFI changes
make update-stubs

# Or run individual steps:
make generate-stubs  # Generate both stub files
make verify-stubs    # Verify correct CUDA item filtering
make test-stubs      # Test compilation with stubs

# Test documentation build
cargo doc --features docs-only --no-default-features --no-deps
cargo doc --features "docs-only cuda" --no-default-features --no-deps

# Run tests (skips docs-only tests automatically)
make test

# Check docs-only compilation
make check-docs-only
```

The stub system ensures that:
- Documentation on docs.rs displays only the APIs available for each feature combination
- The `fast-gicp` crate works seamlessly with stubs without requiring code changes
- CUDA-specific APIs are only visible when the CUDA feature is enabled
- Documentation tests run correctly in development but are safely skipped in docs-only builds

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

The underlying fast_gicp library is also BSD 3-Clause licensed.
