# fast-gicp-sys

Low-level FFI bindings for the fast_gicp C++ library.

## Build Requirements

### Normal Build
- PCL (Point Cloud Library) >= 1.8
- Eigen3 >= 3.0
- C++ compiler with C++17 support
- CMake

### Documentation Build (docs.rs)
No external dependencies required when using the `docs-only` feature.

## Features

- `openmp` (default): Enable OpenMP parallelization
- `cuda`: Enable CUDA GPU acceleration
- `bindgen`: Regenerate bindings from C++ headers (requires all dependencies)
- `docs-only`: Skip C++ compilation, use pre-generated bindings only

## Pre-generated Bindings

This crate includes pre-generated CXX bindings in `src/generated/` to support documentation builds on docs.rs where C++ dependencies are not available.

### Updating Pre-generated Bindings

When making changes to the FFI interface in `src/lib.rs`, you must regenerate the bindings:

```bash
# From the project root
make generate-bindings

# Or manually
cd fast-gicp-sys
cargo build --features bindgen
```

**Important**: Always commit the updated files in `src/generated/` to ensure docs.rs builds continue to work.

## Development Workflow

1. Make changes to the FFI interface in `src/lib.rs`
2. Test the normal build: `cargo build`
3. Regenerate bindings: `make generate-bindings`
4. Test docs-only build: `cargo build --no-default-features --features docs-only`
5. Commit both the source changes and updated `src/generated/` files
