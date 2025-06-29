# Fast GICP Rust Wrapper Architecture

## Overview

The Fast GICP Rust wrapper provides type-safe, idiomatic Rust bindings for the fast_gicp C++ library. The architecture consists of two main crates in a Cargo workspace, designed for safety, performance, and ease of use.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
├─────────────────────────────────────────────────────────────┤
│                    fast-gicp (Rust API)                     │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │ PointCloud  │ Registration │   Builder    │   CUDA   │  │
│  │  XYZ/XYZI   │  Algorithms  │   Pattern    │ Support  │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 fast-gicp-sys (FFI Layer)                   │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │  CXX Bridge │ C++ Wrappers │ Build System │   Types  │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  fast_gicp (C++ Library)                    │
│  ┌─────────────┬──────────────┬──────────────┬──────────┐  │
│  │   FastGICP  │  FastVGICP   │ CUDA Kernels │  NDTCuda │  │
│  └─────────────┴──────────────┴──────────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### High-Level Crate: `fast-gicp`

**Purpose**: Provides safe, idiomatic Rust API for point cloud registration

**Components**:
- **Point Cloud Types**: Zero-copy wrappers around PCL structures
- **Registration Algorithms**: Safe interfaces to GICP variants
- **Builder Pattern**: Fluent configuration API
- **Error Handling**: Rust-idiomatic error types and Result patterns
- **CUDA Support**: Optional GPU-accelerated algorithms

### Low-Level Crate: `fast-gicp-sys`

**Purpose**: FFI bindings and C++ interoperability layer

**Components**:
- **CXX Bridge**: Type-safe FFI using the cxx crate
- **C++ Wrappers**: Thin C++ wrapper functions for Rust compatibility
- **Build System**: CMake integration and dependency management
- **Stub System**: Pre-generated stubs for docs.rs compatibility

## Directory Structure

```
fast_gicp_rust/
├── Cargo.toml                 # Workspace configuration
├── LICENSE                    # BSD 3-Clause License
├── README.md                  # Project documentation
├── Makefile                   # Build and test automation
├── doc/                       # Design and architecture docs
│   ├── ARCH.md               # This file - Architecture overview
│   ├── DESIGN.md             # Design decisions and principles
│   ├── DESIGN_CODEGEN.md     # Code generation system design
│   ├── PROGRESS.md           # Implementation progress tracking
│   └── DEV.md                # Development and build instructions
├── fast_gicp/                 # C++ submodule (upstream library)
├── fast-gicp-sys/            # Low-level FFI bindings
│   ├── Cargo.toml
│   ├── build.rs              # CXX build configuration
│   ├── src/
│   │   ├── lib.rs            # CXX bridge definitions
│   │   ├── bridge.rs         # FFI bridge implementation
│   │   ├── wrapper.cpp       # C++ wrapper implementations
│   │   ├── test.rs           # Low-level tests
│   │   └── generated/        # Generated stub files
│   │       ├── stub.rs       # Non-CUDA stub
│   │       └── stub_cuda.rs  # CUDA-enabled stub
│   ├── include/
│   │   └── wrapper.h         # C++ wrapper headers
│   └── vendor/               # Vendored dependencies
└── fast-gicp/                # High-level Rust API
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs            # Library root
    │   ├── error.rs          # Error types
    │   ├── point_cloud.rs    # Point cloud types
    │   ├── transform.rs      # 3D transformations
    │   ├── types.rs          # Common enums and types
    │   ├── registration/     # Registration algorithms
    │   │   ├── mod.rs
    │   │   ├── validation.rs
    │   │   ├── fast_gicp/
    │   │   │   ├── mod.rs
    │   │   │   └── builder.rs
    │   │   └── fast_vgicp/
    │   │       ├── mod.rs
    │   │       └── builder.rs
    │   ├── fast_vgicp_cuda/  # CUDA algorithms
    │   │   ├── mod.rs
    │   │   └── builder.rs
    │   └── ndt_cuda/
    │       ├── mod.rs
    │       └── builder.rs
    ├── examples/
    │   ├── basic_registration.rs
    │   └── cuda_registration.rs
    └── tests/
        ├── integration.rs
        ├── registration_accuracy.rs
        ├── enum_validation.rs
        ├── cuda_error_handling.rs
        └── ndt_cuda.rs
```

## Data Flow Architecture

### Registration Pipeline

```
User Input → Builder → Algorithm Instance → Registration → Result
     ↓           ↓            ↓              ↓            ↓
Point Clouds → Config → Fast(V)GICP → C++ Engine → Transform + Metrics
```

### Memory Management

- **Point Clouds**: Zero-copy wrappers with automatic cleanup
- **Algorithms**: RAII pattern with Drop trait implementation  
- **Transforms**: Lightweight value types (64 bytes)
- **Thread Safety**: Not Send/Sync (matches C++ library semantics)

## Feature Architecture

### Conditional Compilation

| Feature     | Description                           | Dependencies    | Build Behavior              |
|-------------|---------------------------------------|-----------------|-----------------------------|
| `default`   | Basic algorithms with OpenMP          | OpenMP (auto)   | Full C++ compilation        |
| `openmp`    | Multi-threaded CPU algorithms         | OpenMP 4.0+     | OpenMP-enabled build        |
| `cuda`      | GPU-accelerated algorithms            | CUDA 11.0+      | CUDA compilation required   |
| `docs-only` | Documentation builds (no C++ code)    | None            | Uses pre-generated stubs    |

### Stub System Architecture

```
Build Type    → Feature Detection → Stub Selection → API Exposure
─────────────────────────────────────────────────────────────────
docs-only     → No CUDA           → stub.rs        → Core APIs only
docs-only+cuda → CUDA enabled     → stub_cuda.rs   → Core + CUDA APIs  
normal        → Runtime detection → Real FFI       → All available APIs
```

## Security Architecture

### Memory Safety

- **FFI Boundary**: All unsafe code contained in `fast-gicp-sys`
- **Type Safety**: cxx crate provides compile-time safety guarantees
- **Resource Management**: RAII ensures proper cleanup of C++ resources
- **Input Validation**: Rust-side validation before FFI calls

### Error Handling

- **Recoverable Errors**: Rust Result types for fallible operations
- **Panic Safety**: C++ exceptions converted to Rust panics safely
- **Resource Cleanup**: Drop trait ensures cleanup even on panic
- **Input Sanitization**: Parameter validation at API boundaries

## Performance Architecture

### Zero-Copy Design

- **Point Clouds**: Direct memory mapping to PCL structures
- **Transforms**: Stack-allocated 4x4 matrices
- **Algorithm State**: Minimal Rust wrapper overhead

### Parallel Processing

- **OpenMP**: CPU parallelization in C++ layer
- **CUDA**: GPU acceleration for supported algorithms
- **Thread Safety**: Algorithms are !Send/!Sync for safety

## Integration Architecture

### Build System Integration

```
Cargo Build → build.rs → CMake → C++ Compilation → Linking
     ↓            ↓         ↓          ↓             ↓
  Feature    → Detection → Config → Object Files → Final Binary
  Flags        Logic      Gen.                     
```

### Dependency Management

- **System Dependencies**: PCL, Eigen3, OpenMP detected via CMake
- **Optional Dependencies**: CUDA runtime for GPU features
- **Vendored Dependencies**: fast_gicp C++ library included as submodule
- **Rust Dependencies**: Minimal set focused on FFI and utilities

## Extensibility Architecture

### Algorithm Addition

1. **C++ Integration**: Add algorithm to upstream fast_gicp
2. **FFI Binding**: Extend CXX bridge in `fast-gicp-sys`
3. **Rust API**: Create safe wrapper in `fast-gicp`
4. **Builder Pattern**: Implement configuration builder
5. **Testing**: Add integration and accuracy tests

### Platform Support

- **Current**: Linux with system package dependencies
- **Future**: Cross-platform support with vendored dependencies
- **Containers**: Docker support for reproducible builds