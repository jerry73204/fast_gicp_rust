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
fast-gicp = "0.1"

# For CUDA support
fast-gicp = { version = "0.1", features = ["cuda"] }
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

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

The underlying fast_gicp library is also BSD 3-Clause licensed.
