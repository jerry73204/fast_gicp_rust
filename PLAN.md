# Fast GICP Rust Wrapper Design Plan

## Overview

This document outlines the design for creating a Rust wrapper around the fast_gicp C++ library. The project will consist of two crates in a Cargo workspace:

1. **fast-gicp-sys**: Low-level FFI bindings using the cxx crate
2. **fast-gicp**: High-level idiomatic Rust API

### Development Philosophy

**No Backward Compatibility Guarantees During Development**: This project is in active development. We prioritize creating the best possible API over maintaining backward compatibility. When refactoring or redesigning the API:

- Breaking changes are expected and encouraged if they improve the design
- All affected tests and examples will be rewritten to match the new API
- Users should pin to specific commits if they need stability
- Version 1.0 will mark the first stable API with compatibility guarantees

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
├─────────────────────────────────────────────────────────────┤
│                    fast-gicp (Rust API)                     │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │ PointCloud  │ Registration │     GICP     │   CUDA   │  │
│  │   Types     │   Traits     │  Algorithms  │ Variants │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 fast-gicp-sys (FFI Layer)                   │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │  cxx Bridge │ C++ Wrappers │ Build System │ Bindings │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  fast_gicp (C++ Library)                    │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │   FastGICP  │  FastVGICP   │ CUDA Kernels │    NDT   │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Workspace Structure

```
fast_gicp_rust/
├── Cargo.toml                 # Workspace root
├── fast_gicp/                 # C++ submodule
├── fast-gicp-sys/            # Low-level bindings
│   ├── Cargo.toml
│   ├── build.rs              # Build script for C++ compilation
│   ├── src/
│   │   ├── lib.rs
│   │   └── ffi.rs            # cxx bridge definitions
│   └── include/              # C++ wrapper headers
│       └── fast_gicp_wrapper.h
└── fast-gicp/                # High-level Rust API
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── point_cloud.rs    # Rust point cloud types
    │   ├── registration.rs   # Registration traits
    │   ├── gicp/
    │   │   ├── mod.rs
    │   │   ├── fast_gicp.rs
    │   │   └── fast_vgicp.rs
    │   └── cuda/
    │       ├── mod.rs
    │       ├── fast_vgicp_cuda.rs
    │       └── ndt_cuda.rs
    └── examples/
        ├── basic_registration.rs
        └── cuda_registration.rs
```

### Component Responsibilities

| Component         | Responsibility                                         |
|-------------------|--------------------------------------------------------|
| **fast-gicp-sys** | C++ interop, memory management, type conversion        |
| **fast-gicp**     | Safe Rust API, high-level abstractions, error handling |
| **C++ Wrappers**  | Template instantiation, PCL type conversion            |
| **Build System**  | Dependency detection, compilation flags, linking       |

## Design

### Dependencies Management

#### System Dependencies

| Dependency | Required | Detection Method | Feature Flag | Notes |
|------------|----------|-----------------|--------------|-------|
| PCL 1.8+   | Yes      | pkg-config      | None         | Point Cloud Library |
| Eigen3     | Yes      | Bundled         | None         | From fast_gicp/thirdparty |
| OpenMP     | Optional | CMake auto      | `openmp` (default on) | Multi-threading |
| CUDA 11+   | Optional | cuda-sys crate  | `cuda` to enable | GPU acceleration |

#### Cargo.toml Examples

```toml
# Using default features (OpenMP auto-detected)
[dependencies]
fast-gicp = "0.1"

# Enable CUDA support
[dependencies]
fast-gicp = { version = "0.1", features = ["cuda"] }

# Disable OpenMP (force single-threaded)
[dependencies]
fast-gicp = { version = "0.1", default-features = false }

# CUDA only (no OpenMP)
[dependencies]
fast-gicp = { version = "0.1", default-features = false, features = ["cuda"] }

# Both CUDA and OpenMP (default)
[dependencies]
fast-gicp = { version = "0.1", features = ["cuda"] }
```

### Point Cloud Representation

**Challenge**: PCL point types are complex C++ templates with memory layouts that don't map directly to Rust.

**Solution**: Use instantiated wrapper types (PointCloudXYZ, PointCloudXYZI) that directly wrap PCL point clouds. This avoids conversion overhead and keeps the API simple.

### cxx Bridge Design

```cpp
// C++ wrapper interface (fast_gicp_wrapper.h)
namespace fast_gicp_rust {

struct PointCloudXYZ {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

class FastGICPWrapper {
public:
    void setInputSource(const PointCloudXYZ& cloud);
    void setInputTarget(const PointCloudXYZ& cloud);
    void align(Eigen::Matrix4f initial_guess);
    Eigen::Matrix4f getFinalTransformation() const;
    // Configuration methods
    void setMaxIterations(int iterations);
    void setTransformationEpsilon(double epsilon);
    void setMaxCorrespondenceDistance(double distance);
private:
    std::unique_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> impl;
};
}
```

### Memory Management Strategy

| Scenario               | Strategy                            |
|------------------------|-------------------------------------|
| Point Cloud Transfer   | Clone data across FFI boundary      |
| Algorithm Objects      | `UniquePtr` ownership in Rust       |
| Transformation Results | Copy 4x4 matrix (small, fixed size) |
| Configuration          | Pass by value (primitive types)     |

### API Design Principles

1. **Zero-cost abstractions**: Rust API should not add overhead
2. **Type safety**: Leverage Rust's type system for compile-time guarantees
3. **Ergonomics**: Builder pattern for configuration, method chaining
4. **Error handling**: Result types with descriptive errors

```rust
// Example API usage
let mut gicp = FastGICP::builder()
    .max_iterations(50)
    .transformation_epsilon(1e-8)
    .num_threads(4)
    .build()?;

gicp.set_input_source(source_cloud);
gicp.set_input_target(target_cloud);

let result = gicp.align(None)?; // None = identity initial guess
println!("Converged: {}, Score: {}", result.converged, result.fitness_score);
```

### Feature Flag Architecture

| Feature   | fast-gicp-sys                     | fast-gicp                      | Dependencies   | CMake Flag            |
|-----------|-----------------------------------|--------------------------------|----------------|-----------------------|
| `default` | PCL, Eigen3, OpenMP (if found)    | Basic algorithms with OpenMP   | None           | None                  |
| `openmp`  | Enable OpenMP (on by default)     | Multi-threaded algorithms      | OpenMP 4.0+    | None (auto-detected)  |
| `cuda`    | + CUDA toolkit, GPU algorithms    | + GPU algorithm wrappers       | CUDA 11+       | BUILD_VGICP_CUDA=ON   |

### Build Configuration

#### fast-gicp-sys Features

```toml
[features]
default = ["openmp"]
openmp = []  # Enables OpenMP if available
cuda = []  # Enables BUILD_VGICP_CUDA in CMake
```

#### fast-gicp Features

```toml
[features]
default = ["openmp"]
openmp = ["fast-gicp-sys/openmp"]  # Multi-threaded algorithms
cuda = ["fast-gicp-sys/cuda"]  # Enables CUDA algorithms
```

### Build Script Strategy

```rust
// fast-gicp-sys/build.rs

fn main() {
    let mut cmake = cmake::Config::new("../fast_gicp");
    
    // Handle CUDA feature
    if cfg!(feature = "cuda") {
        cmake.define("BUILD_VGICP_CUDA", "ON");
        // Verify CUDA is available
        if !cuda_available() {
            panic!("CUDA feature enabled but CUDA toolkit not found");
        }
    } else {
        cmake.define("BUILD_VGICP_CUDA", "OFF");
    }
    
    // Handle OpenMP
    if !cfg!(feature = "openmp") {
        // Disable OpenMP detection when feature is not enabled
        cmake.define("CMAKE_DISABLE_FIND_PACKAGE_OpenMP", "TRUE");
    }
    // Otherwise, let CMake auto-detect OpenMP
    
    // Always disable unnecessary components
    cmake.define("BUILD_apps", "OFF");
    cmake.define("BUILD_test", "OFF");
    cmake.define("BUILD_PYTHON_BINDINGS", "OFF");
    
    let dst = cmake.build();
    
    // Link libraries
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=fast_gicp");
    
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=fast_vgicp_cuda");
        // Link CUDA libraries
        link_cuda_libraries();
    }
}

## Rust API Design

### Module Structure

```rust
// fast-gicp/src/lib.rs
pub mod error;
pub mod point_cloud;
pub mod transform;
pub mod gicp;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;

// Re-exports for convenience
pub use error::{Error, Result};
pub use point_cloud::{PointCloudXYZ, PointCloudXYZI};
pub use transform::Transform3f;
```

### Core Types

#### Point Cloud Types

```rust
// fast-gicp/src/point_cloud.rs

use fast_gicp_sys as ffi;

/// Point cloud wrapper around instantiated PCL type (pcl::PointCloud<pcl::PointXYZ>)
pub struct PointCloudXYZ {
    inner: cxx::UniquePtr<ffi::PointCloudXYZ>,
}

impl PointCloudXYZ {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyz()?,
        })
    }
    
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyz_with_capacity(capacity)?,
        })
    }
    
    pub fn num_points(&self) -> usize {
        self.inner.size()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.empty()
    }
    
    pub fn clear(&mut self) {
        self.inner.pin_mut().clear();
    }
    
    pub fn reserve(&mut self, capacity: usize) {
        self.inner.pin_mut().reserve(capacity);
    }
    
    pub fn push_point(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
        self.inner.pin_mut().push_point(x, y, z)?;
        Ok(())
    }
    
    pub fn get_point(&self, index: usize) -> Option<[f32; 3]> {
        if index < self.num_points() {
            Some(self.inner.get_point(index))
        } else {
            None
        }
    }
    
    pub fn from_points(points: &[[f32; 3]]) -> Result<Self> {
        let mut cloud = Self::with_capacity(points.len())?;
        for p in points {
            cloud.push_point(p[0], p[1], p[2])?;
        }
        Ok(cloud)
    }
    
    // For algorithms to access the internal pointer
    pub(crate) fn as_ptr(&self) -> &ffi::PointCloudXYZ {
        &self.inner
    }
    
    pub(crate) fn as_mut_ptr(&mut self) -> Pin<&mut ffi::PointCloudXYZ> {
        self.inner.pin_mut()
    }
}

impl FromIterator<[f32; 3]> for PointCloudXYZ {
    fn from_iter<T: IntoIterator<Item = [f32; 3]>>(iter: T) -> Self {
        let points: Vec<[f32; 3]> = iter.into_iter().collect();
        Self::from_points(&points).expect("Failed to create point cloud")
    }
}

impl FromIterator<(f32, f32, f32)> for PointCloudXYZ {
    fn from_iter<T: IntoIterator<Item = (f32, f32, f32)>>(iter: T) -> Self {
        let mut cloud = Self::new().expect("Failed to create point cloud");
        for (x, y, z) in iter {
            cloud.push_point(x, y, z).expect("Failed to add point");
        }
        cloud
    }
}

// Try-collect pattern for fallible collection
impl TryFrom<Vec<[f32; 3]>> for PointCloudXYZ {
    type Error = Error;
    
    fn try_from(points: Vec<[f32; 3]>) -> Result<Self> {
        Self::from_points(&points)
    }
}

/// Point cloud with intensity (pcl::PointCloud<pcl::PointXYZI>)
pub struct PointCloudXYZI {
    inner: cxx::UniquePtr<ffi::PointCloudXYZI>,
}

impl PointCloudXYZI {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyzi()?,
        })
    }
    
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyzi_with_capacity(capacity)?,
        })
    }
    
    pub fn num_points(&self) -> usize {
        self.inner.size()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.empty()
    }
    
    pub fn clear(&mut self) {
        self.inner.pin_mut().clear();
    }
    
    pub fn reserve(&mut self, capacity: usize) {
        self.inner.pin_mut().reserve(capacity);
    }
    
    pub fn push_point(&mut self, x: f32, y: f32, z: f32, intensity: f32) -> Result<()> {
        self.inner.pin_mut().push_point(x, y, z, intensity)?;
        Ok(())
    }
    
    pub fn get_point(&self, index: usize) -> Option<[f32; 4]> {
        if index < self.num_points() {
            Some(self.inner.get_point(index))
        } else {
            None
        }
    }
    
    pub fn from_points(points: &[([f32; 3], f32)]) -> Result<Self> {
        let mut cloud = Self::with_capacity(points.len())?;
        for (p, i) in points {
            cloud.push_point(p[0], p[1], p[2], *i)?;
        }
        Ok(cloud)
    }
    
    // Internal access
    pub(crate) fn as_ptr(&self) -> &ffi::PointCloudXYZI {
        &self.inner
    }
    
    pub(crate) fn as_mut_ptr(&mut self) -> Pin<&mut ffi::PointCloudXYZI> {
        self.inner.pin_mut()
    }
}

impl FromIterator<[f32; 4]> for PointCloudXYZI {
    fn from_iter<T: IntoIterator<Item = [f32; 4]>>(iter: T) -> Self {
        let mut cloud = Self::new().expect("Failed to create point cloud");
        for [x, y, z, i] in iter {
            cloud.push_point(x, y, z, i).expect("Failed to add point");
        }
        cloud
    }
}

impl FromIterator<(f32, f32, f32, f32)> for PointCloudXYZI {
    fn from_iter<T: IntoIterator<Item = (f32, f32, f32, f32)>>(iter: T) -> Self {
        let mut cloud = Self::new().expect("Failed to create point cloud");
        for (x, y, z, i) in iter {
            cloud.push_point(x, y, z, i).expect("Failed to add point");
        }
        cloud
    }
}

impl FromIterator<([f32; 3], f32)> for PointCloudXYZI {
    fn from_iter<T: IntoIterator<Item = ([f32; 3], f32)>>(iter: T) -> Self {
        let mut cloud = Self::new().expect("Failed to create point cloud");
        for ([x, y, z], i) in iter {
            cloud.push_point(x, y, z, i).expect("Failed to add point");
        }
        cloud
    }
}

// Try-collect pattern
impl TryFrom<Vec<([f32; 3], f32)>> for PointCloudXYZI {
    type Error = Error;
    
    fn try_from(points: Vec<([f32; 3], f32)>) -> Result<Self> {
        Self::from_points(&points)
    }
}
```

#### Transform Types

```rust
// fast-gicp/src/transform.rs

/// 3D transformation (pure Rust type)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3f {
    /// 4x4 transformation matrix in row-major order
    pub matrix: [[f32; 4]; 4],
}

impl Transform3f {
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn from_flat(data: &[f32; 16]) -> Self {
        let mut matrix = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                matrix[i][j] = data[i * 4 + j];
            }
        }
        Self { matrix }
    }
    
    pub fn to_flat(&self) -> [f32; 16] {
        let mut data = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 4 + j] = self.matrix[i][j];
            }
        }
        data
    }
    
    pub fn translation(&self) -> [f32; 3] {
        [self.matrix[0][3], self.matrix[1][3], self.matrix[2][3]]
    }
    
    pub fn set_translation(&mut self, x: f32, y: f32, z: f32) {
        self.matrix[0][3] = x;
        self.matrix[1][3] = y;
        self.matrix[2][3] = z;
    }
}

// Conversion for nalgebra interop
impl From<nalgebra::Matrix4<f32>> for Transform3f {
    fn from(m: nalgebra::Matrix4<f32>) -> Self {
        Self::from_flat(m.as_slice().try_into().unwrap())
    }
}

impl From<Transform3f> for nalgebra::Matrix4<f32> {
    fn from(t: Transform3f) -> Self {
        nalgebra::Matrix4::from_row_slice(&t.to_flat())
    }
}
```

### Configuration Types

```rust
// fast-gicp/src/config.rs

/// Base configuration for all registration algorithms
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    pub max_iterations: i32,
    pub rotation_epsilon: f64,
    pub transformation_epsilon: f64,
    pub max_correspondence_distance: f64,
    pub optimizer_type: OptimizerType,
    pub initial_lambda_factor: f64,
    pub debug_print: bool,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            rotation_epsilon: 2e-3,
            transformation_epsilon: 5e-4,
            max_correspondence_distance: std::f64::MAX,
            optimizer_type: OptimizerType::GaussNewton,
            initial_lambda_factor: 1e-9,
            debug_print: false,
        }
    }
}

/// Builder for registration configuration
pub struct RegistrationConfigBuilder {
    config: RegistrationConfig,
}

impl RegistrationConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: RegistrationConfig::default(),
        }
    }
    
    pub fn max_iterations(mut self, iterations: i32) -> Self {
        self.config.max_iterations = iterations;
        self
    }
    
    pub fn convergence_criteria(mut self, rotation_eps: f64, translation_eps: f64) -> Self {
        self.config.rotation_epsilon = rotation_eps;
        self.config.transformation_epsilon = translation_eps;
        self
    }
    
    pub fn max_correspondence_distance(mut self, distance: f64) -> Self {
        self.config.max_correspondence_distance = distance;
        self
    }
    
    pub fn optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.config.optimizer_type = optimizer;
        self
    }
    
    pub fn build(self) -> RegistrationConfig {
        self.config
    }
}

/// FastGICP-specific configuration
#[derive(Debug, Clone)]
pub struct FastGicpConfig {
    pub base: RegistrationConfig,
    pub num_threads: i32,
    pub correspondence_randomness: i32,
    pub regularization_method: RegularizationMethod,
}

impl Default for FastGicpConfig {
    fn default() -> Self {
        Self {
            base: RegistrationConfig::default(),
            num_threads: 0, // 0 = use all available
            correspondence_randomness: 20,
            regularization_method: RegularizationMethod::None,
        }
    }
}

/// FastVGICP-specific configuration
#[derive(Debug, Clone)]
pub struct FastVGicpConfig {
    pub base: FastGicpConfig,
    pub resolution: f64,
    pub voxel_accumulation_mode: VoxelAccumulationMode,
    pub neighbor_search_method: NeighborSearchMethod,
}

impl Default for FastVGicpConfig {
    fn default() -> Self {
        Self {
            base: FastGicpConfig::default(),
            resolution: 1.0,
            voxel_accumulation_mode: VoxelAccumulationMode::Additive,
            neighbor_search_method: NeighborSearchMethod::Direct27,
        }
    }
}
```

### Algorithm Types

#### Enums (Pure Rust types matching C++)

```rust
// fast-gicp/src/gicp/mod.rs

/// Optimizer types matching C++ LSQ_OPTIMIZER_TYPE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OptimizerType {
    GaussNewton = 0,
    LevenbergMarquardt = 1,
}

/// Regularization methods matching C++ RegularizationMethod
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RegularizationMethod {
    None = 0,
    MinEig = 1,
    NormalizedMinEig = 2,
    Plane = 3,
    Frobenius = 4,
}

/// Neighbor search methods matching C++ NeighborSearchMethod
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NeighborSearchMethod {
    Direct27 = 0,
    Direct7 = 1,
    Direct1 = 2,
    DirectRadius = 3,
}

/// Voxel accumulation modes matching C++ VoxelAccumulationMode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum VoxelAccumulationMode {
    Additive = 0,
    AdditiveWeighted = 1,
    Multiplicative = 2,
}
```

#### Algorithm Wrappers

```rust
// fast-gicp/src/gicp/fast_gicp.rs

/// FastGICP algorithm for PointXYZ (thin wrapper around FFI type)
pub struct FastGicp {
    inner: cxx::UniquePtr<ffi::FastGicpWrapperXYZ>,
}

impl FastGicp {
    /// Create with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_fast_gicp_xyz()?,
        })
    }
    
    /// Create with custom configuration
    pub fn with_config(config: FastGicpConfig) -> Result<Self> {
        let mut gicp = Self::new()?;
        gicp.apply_config(&config);
        Ok(gicp)
    }
    
    /// Create using builder pattern
    pub fn builder() -> FastGicpBuilder {
        FastGicpBuilder::new()
    }
    
    fn apply_config(&mut self, config: &FastGicpConfig) {
        // Apply base config
        self.set_max_iterations(config.base.max_iterations);
        self.set_rotation_epsilon(config.base.rotation_epsilon);
        self.set_transformation_epsilon(config.base.transformation_epsilon);
        self.set_max_correspondence_distance(config.base.max_correspondence_distance);
        self.set_optimizer_type(config.base.optimizer_type);
        
        // Apply FastGICP-specific config
        self.set_num_threads(config.num_threads);
        self.set_correspondence_randomness(config.correspondence_randomness);
        self.set_regularization_method(config.regularization_method);
    }
    
    // Wrapper methods
    pub fn set_input_source(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        self.inner.pin_mut().set_input_source(cloud.as_ptr())?;
        Ok(())
    }
    
    pub fn set_input_target(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        self.inner.pin_mut().set_input_target(cloud.as_ptr())?;
        Ok(())
    }
    
    pub fn align(&mut self, initial_guess: Option<Transform3f>) -> Result<Transform3f> {
        let guess = initial_guess.unwrap_or_else(Transform3f::identity);
        self.inner.pin_mut().align(&guess.to_flat())?;
        Ok(Transform3f::from_flat(&self.inner.get_final_transformation()))
    }
    
    // Configuration methods
    pub fn set_max_iterations(&mut self, iterations: i32) {
        self.inner.pin_mut().set_max_iterations(iterations);
    }
    
    pub fn set_num_threads(&mut self, threads: i32) {
        self.inner.pin_mut().set_num_threads(threads);
    }
    
    pub fn set_regularization_method(&mut self, method: RegularizationMethod) {
        self.inner.pin_mut().set_regularization_method(method as i32);
    }
    
    // Query methods
    pub fn get_fitness_score(&self, max_range: Option<f64>) -> f64 {
        self.inner.get_fitness_score(max_range.unwrap_or(std::f64::MAX))
    }
    
    pub fn has_converged(&self) -> bool {
        self.inner.has_converged()
    }
}

/// Builder for FastGICP
pub struct FastGicpBuilder {
    config: FastGicpConfig,
}

impl FastGicpBuilder {
    pub fn new() -> Self {
        Self {
            config: FastGicpConfig::default(),
        }
    }
    
    pub fn max_iterations(mut self, iterations: i32) -> Self {
        self.config.base.max_iterations = iterations;
        self
    }
    
    pub fn num_threads(mut self, threads: i32) -> Self {
        self.config.num_threads = threads;
        self
    }
    
    pub fn regularization(mut self, method: RegularizationMethod) -> Self {
        self.config.regularization_method = method;
        self
    }
    
    pub fn build(self) -> Result<FastGicp> {
        FastGicp::with_config(self.config)
    }
}
```

### CUDA Types

```rust
// fast-gicp/src/cuda/mod.rs

#[cfg(feature = "cuda")]
/// Nearest neighbor methods matching C++ NearestNeighborMethod
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NearestNeighborMethod {
    CpuParallelKdtree = 0,
    GpuBruteforce = 1,
    GpuRbfKernel = 2,
}

#[cfg(feature = "cuda")]
/// NDT distance modes matching C++ NDTDistanceMode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NdtDistanceMode {
    P2D = 0,
    D2D = 1,
}

// fast-gicp/src/cuda/fast_vgicp_cuda.rs

#[cfg(feature = "cuda")]
/// Configuration for CUDA-accelerated VGICP
#[derive(Debug, Clone)]
pub struct FastVGicpCudaConfig {
    pub base: FastVGicpConfig,
    pub nn_search_method: NearestNeighborMethod,
    pub kernel_width: f32,
    pub kernel_max_dist: f32,
}

#[cfg(feature = "cuda")]
/// FastVGICPCuda algorithm (thin wrapper)
pub struct FastVGicpCuda {
    inner: cxx::UniquePtr<ffi::FastVGicpCudaWrapper>,
}

#[cfg(feature = "cuda")]
impl FastVGicpCuda {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_fast_vgicp_cuda()?,
        })
    }
    
    pub fn with_config(config: FastVGicpCudaConfig) -> Result<Self> {
        let mut vgicp = Self::new()?;
        // Apply configuration...
        Ok(vgicp)
    }
    
    pub fn builder() -> FastVGicpCudaBuilder {
        FastVGicpCudaBuilder::new()
    }
}
```

### Error Types

```rust
// fast-gicp/src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to create algorithm instance")]
    CreationError,
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Point cloud is empty")]
    EmptyPointCloud,
    
    #[error("C++ exception: {0}")]
    CppException(String),
    
    #[error("CUDA error: {0}")]
    #[cfg(feature = "cuda")]
    CudaError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

### FFI Types (in fast-gicp-sys)

```rust
// fast-gicp-sys/src/ffi.rs

#[cxx::bridge]
mod ffi {
    // Error handling
    type CppException;
    
    unsafe extern "C++" {
        include!("fast_gicp_wrapper.h");
        
        // Point Cloud Types (instantiated PCL templates)
        type PointCloudXYZ;
        fn create_point_cloud_xyz() -> Result<UniquePtr<PointCloudXYZ>>;
        fn create_point_cloud_xyz_with_capacity(capacity: usize) -> Result<UniquePtr<PointCloudXYZ>>;
        fn size(self: &PointCloudXYZ) -> usize;
        fn empty(self: &PointCloudXYZ) -> bool;
        fn clear(self: Pin<&mut PointCloudXYZ>);
        fn reserve(self: Pin<&mut PointCloudXYZ>, capacity: usize);
        fn push_point(self: Pin<&mut PointCloudXYZ>, x: f32, y: f32, z: f32) -> Result<()>;
        fn get_point(self: &PointCloudXYZ, index: usize) -> [f32; 3];
        
        type PointCloudXYZI;
        fn create_point_cloud_xyzi() -> Result<UniquePtr<PointCloudXYZI>>;
        fn create_point_cloud_xyzi_with_capacity(capacity: usize) -> Result<UniquePtr<PointCloudXYZI>>;
        fn size(self: &PointCloudXYZI) -> usize;
        fn empty(self: &PointCloudXYZI) -> bool;
        fn clear(self: Pin<&mut PointCloudXYZI>);
        fn reserve(self: Pin<&mut PointCloudXYZI>, capacity: usize);
        fn push_point(self: Pin<&mut PointCloudXYZI>, x: f32, y: f32, z: f32, intensity: f32) -> Result<()>;
        fn get_point(self: &PointCloudXYZI, index: usize) -> [f32; 4];
        
        // FastGICP for PointXYZ
        type FastGicpWrapperXYZ;
        fn create_fast_gicp_xyz() -> Result<UniquePtr<FastGicpWrapperXYZ>>;
        fn set_input_source(self: Pin<&mut FastGicpWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        fn set_input_target(self: Pin<&mut FastGicpWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        fn set_max_iterations(self: Pin<&mut FastGicpWrapperXYZ>, iterations: i32);
        fn set_rotation_epsilon(self: Pin<&mut FastGicpWrapperXYZ>, epsilon: f64);
        fn set_transformation_epsilon(self: Pin<&mut FastGicpWrapperXYZ>, epsilon: f64);
        fn set_max_correspondence_distance(self: Pin<&mut FastGicpWrapperXYZ>, distance: f64);
        fn set_optimizer_type(self: Pin<&mut FastGicpWrapperXYZ>, optimizer: i32);
        fn set_num_threads(self: Pin<&mut FastGicpWrapperXYZ>, threads: i32);
        fn set_correspondence_randomness(self: Pin<&mut FastGicpWrapperXYZ>, randomness: i32);
        fn set_regularization_method(self: Pin<&mut FastGicpWrapperXYZ>, method: i32);
        fn align(self: Pin<&mut FastGicpWrapperXYZ>, initial_guess: &[f32; 16]) -> Result<()>;
        fn get_final_transformation(self: &FastGicpWrapperXYZ) -> [f32; 16];
        fn get_fitness_score(self: &FastGicpWrapperXYZ, max_range: f64) -> f64;
        fn has_converged(self: &FastGicpWrapperXYZ) -> bool;
        fn get_final_num_iterations(self: &FastGicpWrapperXYZ) -> i32;
        
        // FastVGICP for PointXYZ
        type FastVGicpWrapperXYZ;
        fn create_fast_vgicp_xyz() -> Result<UniquePtr<FastVGicpWrapperXYZ>>;
        fn set_input_source(self: Pin<&mut FastVGicpWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        fn set_input_target(self: Pin<&mut FastVGicpWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        // ... similar methods plus:
        fn set_resolution(self: Pin<&mut FastVGicpWrapperXYZ>, resolution: f64);
        fn set_voxel_accumulation_mode(self: Pin<&mut FastVGicpWrapperXYZ>, mode: i32);
        fn set_neighbor_search_method(self: Pin<&mut FastVGicpWrapperXYZ>, method: i32);
        
        #[cfg(feature = "cuda")]
        // FastVGICPCuda for PointXYZ
        type FastVGicpCudaWrapperXYZ;
        fn create_fast_vgicp_cuda_xyz() -> Result<UniquePtr<FastVGicpCudaWrapperXYZ>>;
        fn set_input_source(self: Pin<&mut FastVGicpCudaWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        fn set_input_target(self: Pin<&mut FastVGicpCudaWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        // ... similar methods plus:
        fn set_nearest_neighbor_search_method(self: Pin<&mut FastVGicpCudaWrapperXYZ>, method: i32);
        fn set_kernel_width(self: Pin<&mut FastVGicpCudaWrapperXYZ>, width: f32, max_dist: f32);
        
        #[cfg(feature = "cuda")]
        // NDTCuda for PointXYZ
        type NdtCudaWrapperXYZ;
        fn create_ndt_cuda_xyz() -> Result<UniquePtr<NdtCudaWrapperXYZ>>;
        fn set_input_source(self: Pin<&mut NdtCudaWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        fn set_input_target(self: Pin<&mut NdtCudaWrapperXYZ>, cloud: &PointCloudXYZ) -> Result<()>;
        // ... similar methods plus:
        fn set_distance_mode(self: Pin<&mut NdtCudaWrapperXYZ>, mode: i32);
        fn set_outlier_ratio(self: Pin<&mut NdtCudaWrapperXYZ>, ratio: f64);
    }
}

// C++ wrapper headers will contain:
// - PointCloudXYZ: wrapper around pcl::PointCloud<pcl::PointXYZ>::Ptr
// - PointCloudXYZI: wrapper around pcl::PointCloud<pcl::PointXYZI>::Ptr
// - FastGicpWrapperXYZ: wrapper around fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>
// - FastVGicpWrapperXYZ: wrapper around fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>
// etc.
```

### C++ Wrapper Design (fast_gicp_wrapper.h)

```cpp
namespace fast_gicp_rust {

// Opaque wrapper around pcl::PointCloud<pcl::PointXYZ>::Ptr
class PointCloudXYZ {
public:
    PointCloudXYZ();
    explicit PointCloudXYZ(size_t capacity);
    
    size_t size() const;
    bool empty() const;
    void clear();
    void reserve(size_t capacity);
    void push_point(float x, float y, float z);
    std::array<float, 3> get_point(size_t index) const;
    
    // Internal use only
    pcl::PointCloud<pcl::PointXYZ>::Ptr get_pcl_cloud() { return cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr get_pcl_cloud() const { return cloud_; }
    
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
};

// Factory functions for cxx
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz();
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz_with_capacity(size_t capacity);

// Instantiated wrapper for FastGICP<pcl::PointXYZ, pcl::PointXYZ>
class FastGicpWrapperXYZ {
public:
    FastGicpWrapperXYZ();
    void set_input_source(const PointCloudXYZ& cloud);
    void set_input_target(const PointCloudXYZ& cloud);
    void set_max_iterations(int iterations);
    void set_rotation_epsilon(double epsilon);
    void set_transformation_epsilon(double epsilon);
    void set_max_correspondence_distance(double distance);
    void set_optimizer_type(int optimizer);
    void set_num_threads(int threads);
    void set_correspondence_randomness(int randomness);
    void set_regularization_method(int method);
    void align(const std::array<float, 16>& initial_guess);
    std::array<float, 16> get_final_transformation() const;
    double get_fitness_score(double max_range) const;
    bool has_converged() const;
    int get_final_num_iterations() const;
    
private:
    std::shared_ptr<fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>> gicp_;
};

} // namespace fast_gicp_rust
```

### Usage Examples

```rust
// Example 1: Using direct API
use fast_gicp::{PointCloudXYZ, FastGicp, RegularizationMethod};

let source = PointCloudXYZ::from_points(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
let target = PointCloudXYZ::from_points(&[[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])?;

let mut gicp = FastGicp::new()?;
gicp.set_max_iterations(50);
gicp.set_num_threads(4);
gicp.set_regularization_method(RegularizationMethod::MinEig);

gicp.set_input_source(&source)?;
gicp.set_input_target(&target)?;

let transform = gicp.align(None)?;
println!("Fitness: {}", gicp.get_fitness_score(Some(1.0)));

// Example 2: Using builder pattern
let mut gicp = FastGicp::builder()
    .max_iterations(50)
    .num_threads(4)
    .regularization(RegularizationMethod::MinEig)
    .build()?;

// Example 3: Using FromIterator to collect from iterators
let points: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let source: PointCloudXYZ = points.into_iter().collect();

// Collect from tuple iterator
let source: PointCloudXYZ = (0..1000)
    .map(|i| {
        let angle = i as f32 * 0.01;
        (angle.cos(), angle.sin(), i as f32 * 0.1)
    })
    .collect();

// Example 4: Building point clouds incrementally
let mut source = PointCloudXYZ::new()?;
for i in 0..1000 {
    let angle = i as f32 * 0.01;
    source.push_point(angle.cos(), angle.sin(), i as f32 * 0.1)?;
}

// Example 5: Point cloud with intensity using FromIterator
let cloud_with_intensity: PointCloudXYZI = vec![
    ([1.0, 2.0, 3.0], 0.8),
    ([4.0, 5.0, 6.0], 0.3),
    ([7.0, 8.0, 9.0], 0.9),
].into_iter().collect();

// Or from [f32; 4] format
let cloud_xyzi: PointCloudXYZI = vec![
    [1.0, 2.0, 3.0, 0.8],
    [4.0, 5.0, 6.0, 0.3],
].into_iter().collect();

// Example 6: Using CUDA features
#[cfg(feature = "cuda")]
{
    use fast_gicp::cuda::FastVGicpCuda;
    
    let mut cuda_gicp = FastVGicpCuda::new()?;
    cuda_gicp.set_input_source(&source)?;
    cuda_gicp.set_input_target(&target)?;
    let transform = cuda_gicp.align(None)?;
}

// Example 7: Controlling threading
let mut gicp = FastGicp::new()?;
#[cfg(feature = "openmp")]
gicp.set_num_threads(4);  // Multi-threaded with OpenMP
#[cfg(not(feature = "openmp"))]
gicp.set_num_threads(1);  // Single-threaded only
```

## Progress Tracking

### Phase 1: Basic Infrastructure

| Task                                    | Status         | Assignee | Notes |
|-----------------------------------------|----------------|----------|-------|
| Set up workspace structure              | ⬜ Not Started |          | Two crates: fast-gicp-sys, fast-gicp |
| Create Cargo.toml with features         | ⬜ Not Started |          | openmp (default), cuda |
| Implement C++ point cloud wrappers      | ⬜ Not Started |          | PointCloudXYZ, PointCloudXYZI |
| Create cxx bridge for point clouds      | ⬜ Not Started |          | Instantiated PCL types |
| Write CMake-based build.rs              | ⬜ Not Started |          | Handle OpenMP/CUDA features |
| Basic point cloud manipulation example  | ⬜ Not Started |          | Test FFI layer |

### Phase 2: Core Algorithms

| Task                             | Status         | Assignee | Notes |
|----------------------------------|----------------|----------|-------|
| FastGicpWrapperXYZ C++ class     | ⬜ Not Started |          | Wrap FastGICP<PointXYZ, PointXYZ> |
| FastGICP cxx bridge              | ⬜ Not Started |          | All methods from C++ API |
| FastGICP Rust wrapper            | ⬜ Not Started |          | Thin wrapper over FFI |
| Configuration structs            | ⬜ Not Started |          | RegistrationConfig, FastGicpConfig |
| FastVGicpWrapperXYZ C++ class    | ⬜ Not Started |          | Wrap FastVGICP<PointXYZ, PointXYZ> |
| FastVGICP Rust wrapper           | ⬜ Not Started |          | Include voxel-specific methods |
| Integration tests                | ⬜ Not Started |          | Test registration accuracy |
| OpenMP feature tests             | ⬜ Not Started |          | Test with/without OpenMP |

### Phase 3: CUDA Support

| Task                              | Status         | Assignee | Notes |
|-----------------------------------|----------------|----------|-------|
| CUDA detection in build.rs        | ⬜ Not Started |          | Use cuda-sys crate |
| FastVGicpCudaWrapperXYZ C++ class | ⬜ Not Started |          | Wrap GPU implementation |
| FastVGICPCuda cxx bridge          | ⬜ Not Started |          | GPU-specific methods |
| NdtCudaWrapperXYZ C++ class       | ⬜ Not Started |          | Wrap NDT CUDA |
| CUDA feature flag testing         | ⬜ Not Started |          | Conditional compilation |
| CUDA examples                     | ⬜ Not Started |          | GPU vs CPU comparison |
| CUDA CI setup                     | ⬜ Not Started |          | GPU runners for CI |

### Phase 4: Polish and Release

| Task                        | Status         | Assignee | Notes |
|-----------------------------|----------------|----------|-------|
| Comprehensive benchmarks    | ⬜ Not Started |          | Compare with C++ performance |
| API documentation           | ⬜ Not Started |          | Full rustdoc coverage |
| Usage guide                 | ⬜ Not Started |          | README with examples |
| CI/CD pipeline              | ⬜ Not Started |          | Multi-platform, feature matrix |
| Crates.io preparation       | ⬜ Not Started |          | Metadata, categories, keywords |
| Version 0.1.0 release       | ⬜ Not Started |          | First development release |

## Phase-by-Phase Action Items

### Phase 1: Basic Infrastructure (Week 1-2)

1. **Day 1-2: Project Setup**
   - [ ] Initialize workspace with root `Cargo.toml`
   - [ ] Create `fast-gicp-sys/` and `fast-gicp/` directories
   - [ ] Set up feature flags (openmp as default, cuda as optional)
   - [ ] Add cxx, cmake, and pkg-config dependencies

2. **Day 3-4: C++ Wrapper Foundation**
   - [ ] Create `fast-gicp-sys/include/fast_gicp_wrapper.h`
   - [ ] Implement PointCloudXYZ wrapper class
   - [ ] Implement PointCloudXYZI wrapper class
   - [ ] Write cxx bridge for point cloud types
   - [ ] Test point cloud creation and manipulation

3. **Day 5-6: Build System**
   - [ ] Write CMake-based build.rs
   - [ ] Handle OpenMP feature flag (`!cfg!(feature = "openmp")`)
   - [ ] Configure PCL detection via pkg-config
   - [ ] Link against bundled Eigen3
   - [ ] Test build on Linux with/without OpenMP

4. **Day 7-8: Point Cloud Rust API**
   - [ ] Implement PointCloudXYZ Rust wrapper
   - [ ] Implement PointCloudXYZI Rust wrapper
   - [ ] Add FromIterator traits for ergonomic collection
   - [ ] Add TryFrom traits for fallible conversion
   - [ ] Create Transform3f type
   - [ ] Write unit tests for point clouds
   - [ ] Test performance of push_point operations

5. **Day 9-10: Basic Example**
   - [ ] Create examples/ directory
   - [ ] Write point cloud creation example
   - [ ] Test FFI layer thoroughly
   - [ ] Document any build issues

### Phase 2: Core Algorithms (Week 3-4)

**Note**: Before starting new tasks, check for `todo!()` markers from Phase 1 and prioritize completing half-finished features.

1. **Week 3, Day 1-2: FastGICP C++ Wrapper**
   - [ ] Complete any half-finished items from Phase 1 (check `todo!()` markers)
   - [ ] Implement FastGicpWrapperXYZ class
   - [ ] Wrap all LsqRegistration methods
   - [ ] Add FastGICP-specific methods
   - [ ] Handle transformation matrices
   - [ ] Test C++ wrapper compilation

2. **Week 3, Day 3-4: FastGICP FFI Bridge**
   - [ ] Add FastGicpWrapperXYZ to cxx bridge
   - [ ] Implement create/destroy functions
   - [ ] Bridge all setter methods
   - [ ] Bridge align and query methods
   - [ ] Test FFI calls

3. **Week 3, Day 5: FastGICP Rust API**
   - [ ] Create gicp module structure
   - [ ] Implement FastGicp struct
   - [ ] Add configuration structs
   - [ ] Implement builder pattern
   - [ ] Write basic registration example

4. **Week 4, Day 1-2: FastVGICP Implementation**
   - [ ] Implement FastVGicpWrapperXYZ class
   - [ ] Add voxel-specific methods
   - [ ] Create cxx bridge
   - [ ] Implement FastVGicp Rust wrapper
   - [ ] Add voxelized example

5. **Week 4, Day 3-5: Testing and Documentation**
   - [ ] Write integration tests for accuracy
   - [ ] Test OpenMP vs no-OpenMP performance
   - [ ] Add rustdoc to all public APIs
   - [ ] Create comparison benchmarks
   - [ ] Update README with usage examples
   - [ ] Add todo!() for any missing features found during testing

### Phase 3: CUDA Support (Week 5-6)

1. **Week 5, Day 1-2: CUDA Build Infrastructure**
   - [ ] Add cuda-sys dependency (optional)
   - [ ] Update build.rs for CUDA detection
   - [ ] Handle BUILD_VGICP_CUDA CMake flag
   - [ ] Link CUDA libraries conditionally
   - [ ] Test on CUDA-enabled system

2. **Week 5, Day 3-4: FastVGICPCuda Wrapper**
   - [ ] Implement FastVGicpCudaWrapperXYZ
   - [ ] Add GPU-specific methods
   - [ ] Create cxx bridge with cfg(feature = "cuda")
   - [ ] Implement Rust wrapper
   - [ ] Handle CUDA errors properly

3. **Week 5, Day 5: NDTCuda Wrapper**
   - [ ] Implement NdtCudaWrapperXYZ
   - [ ] Add NDT-specific methods
   - [ ] Create cxx bridge
   - [ ] Implement Rust wrapper
   - [ ] Add NDT example

4. **Week 6: CUDA Testing and CI**
   - [ ] Write CUDA-specific tests
   - [ ] Create CPU vs GPU benchmarks
   - [ ] Document CUDA requirements
   - [ ] Set up GPU-enabled CI runners
   - [ ] Test on multiple GPU architectures

### Phase 4: Polish and Release (Week 7-8)

1. **Week 7: Quality Assurance**
   - [ ] Run comprehensive benchmarks
   - [ ] Compare performance with C++
   - [ ] Fix any performance regressions
   - [ ] Complete API documentation
   - [ ] Run clippy with pedantic lints

2. **Week 8: Release Preparation**
   - [ ] Set up CI/CD for all platforms
   - [ ] Test feature matrix in CI
   - [ ] Prepare crates.io metadata
   - [ ] Write comprehensive README
   - [ ] Create CHANGELOG.md
   - [ ] Tag version 0.1.0
   - [ ] Publish to crates.io (optional)

## API Evolution Guidelines

### Refactoring Process

When refactoring or redesigning the API:

1. **Identify Pain Points**: Document what's wrong with the current design
2. **Propose New Design**: Create a design document or issue
3. **Update PLAN.md**: Revise this document with the new design
4. **Implement Changes**: Make breaking changes without deprecation warnings
5. **Update All Code**: 
   - Rewrite all tests to use the new API
   - Update all examples
   - Update documentation
   - No need to maintain old API versions
   - Use `todo!()` for features that become missing after refactoring

### Tracking Half-Completed Work

When features are left half-completed with `todo!()` markers:

1. **Document the State**: Update progress tables to mark tasks as "🔄 In Progress" 
2. **Add Completion Tasks**: Break down what's needed to complete the feature
3. **Prioritize**: Half-completed features should be finished before starting new ones
4. **Update Action Items**: Reorganize upcoming phase items to prioritize completion

Example progress table entry:
| Task | Status | Assignee | Notes |
|------|--------|----------|-------|
| FastGICP Rust wrapper | 🔄 In Progress | | Basic structure done, missing covariance methods |

### Areas Open for Redesign

- Point cloud representation (currently wrapping PCL types)
- Algorithm configuration (builder pattern vs config structs)
- Error types and error handling strategy
- Module organization
- Feature flag structure
- FFI boundary design

### Version Strategy

- `0.x.y`: Breaking changes in any version bump
- `1.0.0`: First stable release with compatibility guarantees
- Until 1.0: Users should use `=` version requirements or git commits

Example Cargo.toml during development:
```toml
# Pin to exact version
fast-gicp = "=0.2.3"

# Or use git commit
fast-gicp = { git = "https://github.com/user/fast_gicp_rust", rev = "abc123" }
```

## Risk Mitigation

| Risk                        | Impact | Mitigation Strategy                     |
|-----------------------------|--------|-----------------------------------------|
| PCL version incompatibility | High   | Support PCL 1.8+ with version detection |
| CUDA toolkit variations     | Medium | Support CUDA 11+, runtime detection     |
| Platform differences        | Medium | CI testing on Linux/macOS/Windows       |
| Performance regression      | High   | Automated benchmarks in CI              |
| Complex C++ templates       | High   | Limited point type support initially    |
| API instability            | Low    | Clear communication about breaking changes |

## Testing and Quality Assurance

### Testing Strategy

#### Unit Tests
```bash
# Run all tests
cargo nextest run --no-fail-fast

# Run tests with all features enabled
cargo nextest run --no-fail-fast --all-features

# Run tests for a specific package
cargo nextest run --no-fail-fast -p fast-gicp-sys
cargo nextest run --no-fail-fast -p fast-gicp

# Run tests with output
cargo nextest run --no-fail-fast --nocapture

# Run a specific test
cargo nextest run --no-fail-fast test_name
```

#### Integration Tests
```bash
# Run all tests including examples and integration tests
cargo nextest run --no-fail-fast --all-targets

# Run tests with all features and all targets
cargo nextest run --no-fail-fast --all-features --all-targets

# Run only doc tests
cargo test --doc
```

#### Feature-Specific Testing
```bash
# Test CUDA features specifically
cargo nextest run --no-fail-fast --features cuda --all-targets

# Test without any features
cargo nextest run --no-fail-fast --no-default-features --all-targets

# Test without OpenMP (single-threaded)
cargo nextest run --no-fail-fast --no-default-features --all-targets

# Test all combinations
cargo nextest run --no-fail-fast --all-targets  # Default (with OpenMP)
cargo nextest run --no-fail-fast --features cuda --all-targets  # OpenMP + CUDA
cargo nextest run --no-fail-fast --no-default-features --all-targets  # No OpenMP
cargo nextest run --no-fail-fast --no-default-features --features cuda --all-targets  # CUDA only
```

### Linting and Code Quality

#### Clippy Linting
```bash
# Run clippy on all targets
cargo clippy --all-targets

# Run clippy with all features
cargo clippy --all-features --all-targets

# Run clippy with stricter lints
cargo clippy --all-targets --all-features -- -W clippy::all -W clippy::pedantic

# Run clippy on workspace
cargo clippy --workspace --all-targets --all-features

# Fix clippy warnings automatically (when possible)
cargo clippy --fix --all-targets --all-features
```

#### Code Formatting
```bash
# Check formatting
cargo fmt --check

# Format code
cargo fmt

# Format entire workspace
cargo fmt --all
```

#### Documentation
```bash
# Build documentation
cargo doc --no-deps

# Build documentation with all features
cargo doc --all-features --no-deps

# Build and open documentation
cargo doc --all-features --no-deps --open

# Check documentation examples
cargo test --doc --all-features
```

### Continuous Integration Commands

```yaml
# Example CI workflow commands
- cargo fmt --all -- --check
- cargo clippy --workspace --all-targets --all-features -- -D warnings

# Test different feature combinations
- cargo nextest run --no-fail-fast --workspace --all-targets  # Default (with OpenMP)
- cargo nextest run --no-fail-fast --workspace --all-targets --no-default-features  # No features
- cargo nextest run --no-fail-fast --workspace --all-targets --features cuda  # OpenMP + CUDA (CUDA runners only)
- cargo nextest run --no-fail-fast --workspace --all-targets --no-default-features --features cuda  # CUDA only (CUDA runners only)
- cargo nextest run --no-fail-fast --workspace --all-targets --all-features  # All features (CUDA runners only)

# Documentation
- cargo doc --workspace --all-features --no-deps
```

### CI Matrix Strategy

```yaml
# GitHub Actions example
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        features: ""  # Default (OpenMP enabled)
      - os: ubuntu-latest
        features: "--no-default-features"  # No OpenMP
      - os: ubuntu-latest
        features: "--features cuda"  # OpenMP + CUDA
      - os: ubuntu-latest
        features: "--no-default-features --features cuda"  # CUDA only
      - os: macos-latest
        features: ""  # Default (OpenMP enabled)
      - os: macos-latest
        features: "--no-default-features"  # No OpenMP
```

### Test Organization

#### Test Philosophy

**Use todo!() for Missing and Half-Completed Features**: When writing tests and discovering that required features are not yet implemented or only partially implemented, use `todo!()` macros and TODO comments instead of creating dummy values or skipping assertions. This ensures tests fail explicitly rather than giving false positives.

**Half-Completed Features**: If you start implementing a feature but can't complete it in the current session, leave `todo!()` markers at the incomplete parts. We'll reorganize action items to prioritize completing these half-finished features before starting new ones.

```rust
#[test]
fn test_feature_not_yet_implemented() {
    let mut gicp = FastGicp::new().unwrap();
    
    // TODO: Implement get_source_covariances method
    let covariances = gicp.get_source_covariances();
    todo!("get_source_covariances not yet implemented");
    
    // This ensures the test fails and reminds us what needs to be done
    assert!(covariances.is_some());
}

#[test]
fn test_partial_implementation() {
    let cloud = PointCloudXYZ::new().unwrap();
    
    // Implemented features can be tested normally
    assert!(cloud.is_empty());
    
    // TODO: Add iteration support - partially implemented but Iterator trait missing
    // for point in &cloud { ... }
    todo!("Iterator trait not yet implemented for PointCloudXYZ");
}

// Example of half-completed implementation
impl PointCloudXYZ {
    pub fn transform(&mut self, transform: &Transform3f) {
        // TODO: Complete transform implementation
        // Basic structure is here but matrix multiplication is missing
        todo!("Matrix multiplication not yet implemented in transform method");
    }
}
```

#### Unit Test Structure
```rust
// In src/point_cloud.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let cloud = PointCloudXYZ::new().unwrap();
        assert!(cloud.is_empty());
        assert_eq!(cloud.num_points(), 0);
    }

    #[test]
    fn test_point_insertion() {
        let mut cloud = PointCloudXYZ::new().unwrap();
        cloud.push_point(1.0, 2.0, 3.0).unwrap();
        assert_eq!(cloud.num_points(), 1);
        assert_eq!(cloud.get_point(0), Some([1.0, 2.0, 3.0]));
    }
    
    #[test]
    fn test_from_iterator_arrays() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cloud: PointCloudXYZ = points.into_iter().collect();
        assert_eq!(cloud.num_points(), 2);
        assert_eq!(cloud.get_point(0), Some([1.0, 2.0, 3.0]));
        assert_eq!(cloud.get_point(1), Some([4.0, 5.0, 6.0]));
    }
    
    #[test]
    fn test_from_iterator_tuples() {
        let cloud: PointCloudXYZ = vec![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
            .into_iter().collect();
        assert_eq!(cloud.num_points(), 2);
    }
    
    #[test]
    fn test_xyzi_from_iterator() {
        let cloud: PointCloudXYZI = vec![
            ([1.0, 2.0, 3.0], 0.8),
            ([4.0, 5.0, 6.0], 0.3),
        ].into_iter().collect();
        assert_eq!(cloud.num_points(), 2);
        assert_eq!(cloud.get_point(0), Some([1.0, 2.0, 3.0, 0.8]));
    }
}
```

#### Integration Test Structure
```rust
// In tests/integration_test.rs
use fast_gicp::{PointCloudXYZ, FastGicp, Transform3f};

#[test]
fn test_basic_registration() {
    let source = PointCloudXYZ::from_points(&[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]).unwrap();
    
    let target = PointCloudXYZ::from_points(&[
        [1.1, 0.1, 0.0],
        [0.1, 1.1, 0.0],
        [0.0, 0.1, 1.1],
    ]).unwrap();
    
    let mut gicp = FastGicp::new().unwrap();
    gicp.set_input_source(&source).unwrap();
    gicp.set_input_target(&target).unwrap();
    
    let result = gicp.align(None).unwrap();
    assert!(gicp.has_converged());
    assert!(gicp.get_fitness_score(None) < 0.1);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_registration() {
    use fast_gicp::cuda::{FastVGicpCuda, NearestNeighborMethod};
    
    let source = create_test_cloud();
    let target = create_test_cloud_transformed();
    
    let mut cuda_gicp = FastVGicpCuda::new().unwrap();
    cuda_gicp.set_nearest_neighbor_search_method(NearestNeighborMethod::GpuBruteforce);
    cuda_gicp.set_input_source(&source).unwrap();
    cuda_gicp.set_input_target(&target).unwrap();
    
    let result = cuda_gicp.align(None).unwrap();
    assert!(cuda_gicp.has_converged());
}

#[cfg(not(feature = "openmp"))]
#[test]
fn test_single_threaded() {
    let mut gicp = FastGicp::new().unwrap();
    gicp.set_num_threads(1); // Should only allow single-threaded
    // Verify it runs without OpenMP
}

#[cfg(feature = "openmp")]
#[test]
fn test_multi_threaded() {
    let mut gicp = FastGicp::new().unwrap();
    gicp.set_num_threads(4); // Should use 4 threads with OpenMP
    // Verify multi-threading works
}
```

#### Benchmark Structure
```rust
// In benches/registration_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fast_gicp::{PointCloudXYZ, FastGicp};

fn benchmark_registration(c: &mut Criterion) {
    let source = create_large_point_cloud(10000);
    let target = create_transformed_cloud(&source);
    
    c.bench_function("fast_gicp_10k_points", |b| {
        b.iter(|| {
            let mut gicp = FastGicp::new().unwrap();
            gicp.set_input_source(&source).unwrap();
            gicp.set_input_target(&target).unwrap();
            black_box(gicp.align(None).unwrap());
        });
    });
}

criterion_group!(benches, benchmark_registration);
criterion_main!(benches);
```

### Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Format check
cargo fmt --all -- --check || exit 1

# Lint check
cargo clippy --all-targets --all-features -- -D warnings || exit 1

# Test
cargo nextest run --no-fail-fast --all-targets || exit 1
```

### Development Workflow

1. **Before committing**:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo nextest run --no-fail-fast --all-targets --all-features
   ```

2. **When making breaking changes**:
   ```bash
   # Update all code to new API
   cargo nextest run --no-fail-fast --all-targets --all-features  # Fix all test failures
   cargo build --examples --all-features    # Fix all examples
   cargo doc --all-features --no-deps       # Update documentation
   ```

   **Note**: Tests with `todo!()` macros are expected to fail. Don't suppress or skip these - they serve as reminders of missing or half-completed features. Half-completed features should be prioritized in the next development session.

3. **Before releasing**:
   ```bash
   cargo nextest run --no-fail-fast --all-features --all-targets --release
   cargo doc --all-features --no-deps
   cargo package --dry-run
   # Update version number (can be breaking)
   ```

4. **For benchmarking**:
   ```bash
   cargo bench
   cargo bench --features cuda
   ```

5. **Discovering todo!() items**:
   ```bash
   # Find all todo!() markers in the codebase
   grep -r "todo!" src/ --include="*.rs"
   
   # Find TODO comments
   grep -r "TODO:" src/ --include="*.rs"
   
   # Run tests to see which ones fail due to todo!()
   cargo nextest run --no-fail-fast 2>&1 | grep "todo"
   ```

## Success Metrics

- [ ] Performance within 10% of C++ implementation
- [ ] Zero unsafe code in high-level crate (all unsafe in sys crate)
- [ ] All implemented tests passing with `cargo nextest run --no-fail-fast --all-features --all-targets`
- [ ] No clippy warnings with `cargo clippy --all-features --all-targets`
- [ ] 90%+ test coverage
- [ ] Examples for all major use cases
- [ ] Builds on Linux, macOS, Windows
- [ ] Documentation for all public APIs
- [ ] Feature flags working correctly (OpenMP, CUDA)
- [ ] Tests with `todo!()` clearly indicate missing features (no false positives)
- [ ] Half-completed features tracked and prioritized for completion

## Implementation Summary

This plan outlines a Rust wrapper for fast_gicp that:

1. **Uses instantiated types** instead of C++ templates to simplify the FFI boundary
2. **Provides thin wrappers** around C++ objects with minimal overhead
3. **Directly wraps PCL point clouds** to avoid conversion costs (no intermediate Rust types)
4. **Ergonomic collection APIs** with FromIterator traits for building point clouds
5. **Supports optional features** for OpenMP (default) and CUDA
6. **Maintains flexibility** to refactor the API during development
7. **Focuses on safety** by keeping all unsafe code in the sys crate

The implementation is divided into four phases, with clear progress tracking and action items for each phase. The design prioritizes correctness and performance over backward compatibility during the initial development phase.
