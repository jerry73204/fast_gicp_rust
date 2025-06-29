# fast-gicp

[![Crates.io](https://img.shields.io/crates/v/fast-gicp.svg)](https://crates.io/crates/fast-gicp)
[![Documentation](https://docs.rs/fast-gicp/badge.svg)](https://docs.rs/fast-gicp)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Rust wrapper for the fast_gicp C++ library, providing efficient 3D point cloud registration algorithms.

## Features

- **Fast GICP**: Generalized Iterative Closest Point algorithm with multi-threading support
- **Fast VGICP**: Voxelized GICP for efficient large-scale point cloud registration
- **CUDA Acceleration**: GPU-accelerated variants (FastVGICPCuda, NDTCuda)
- **Builder Pattern**: Type-safe configuration with compile-time validation
- **Zero-Copy**: Efficient point cloud wrappers with minimal overhead
- **Comprehensive Testing**: 90+ tests covering all algorithms and edge cases

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fast-gicp = "0.3"

# For CUDA support
fast-gicp = { version = "0.3", features = ["cuda"] }
```

Basic usage:

```rust
use fast_gicp::{FastGICP, PointCloudXYZ};

// Create point clouds
let source = PointCloudXYZ::from_points(&[
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
]);

let target = PointCloudXYZ::from_points(&[
    [0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0],
]);

// Configure and run registration
let gicp = FastGICP::builder()
    .max_iterations(50)
    .transformation_epsilon(1e-6)
    .build()?;

let result = gicp.align(&source, &target)?;

println!("Converged: {}", result.has_converged);
println!("Final transformation: {:?}", result.final_transformation);
```

## Available Algorithms

### FastGICP
Standard GICP with optional regularization methods:
```rust
use fast_gicp::types::RegularizationMethod;

let gicp = FastGICP::builder()
    .max_iterations(100)
    .num_threads(4)
    .regularization_method(RegularizationMethod::Frobenius)
    .build()?;
```

### FastVGICP
Voxelized GICP for large point clouds:
```rust
use fast_gicp::types::VoxelAccumulationMode;

let vgicp = FastVGICP::builder()
    .resolution(0.5)
    .voxel_accumulation_mode(VoxelAccumulationMode::Additive)
    .build()?;
```

### CUDA Variants
GPU-accelerated algorithms (requires `cuda` feature):
```rust
// Voxelized GICP on GPU
let cuda_vgicp = FastVGICPCuda::builder()
    .resolution(1.0)
    .build()?;

// NDT on GPU
let ndt = NDTCuda::builder()
    .resolution(0.5)
    .distance_mode(NdtDistanceMode::D2D)
    .build()?;
```

## Documentation

Full API documentation with examples is available at [docs.rs/fast-gicp](https://docs.rs/fast-gicp).

## Requirements

- Rust 1.70+
- CMake 3.15+
- C++17 compiler
- PCL 1.8+ (Point Cloud Library)
- Optional: CUDA 11.0+ for GPU features

## License

Licensed under the BSD 3-Clause License. See [LICENSE](../LICENSE) for details.