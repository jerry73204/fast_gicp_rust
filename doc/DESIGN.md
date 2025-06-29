# Fast GICP Rust Wrapper Design Decisions

## Overview

This document explains the key design decisions made in the Fast GICP Rust wrapper, including rationale, trade-offs, and alternatives considered.

## Core Design Principles

### 1. Safety First

**Decision**: All unsafe code is contained within the `fast-gicp-sys` crate.

**Rationale**: 
- Rust's primary value proposition is memory safety
- Users should not need to write unsafe code to use point cloud registration
- FFI boundary is the natural place for unsafe operations

**Implementation**:
- High-level `fast-gicp` crate contains zero unsafe blocks
- All FFI interactions go through type-safe cxx bridge
- C++ exceptions are properly handled and converted to Rust Results

### 2. Zero-Copy Performance

**Decision**: Point clouds use zero-copy wrappers around PCL structures.

**Rationale**:
- Point clouds can be extremely large (millions of points)
- Copying data would create unacceptable performance overhead
- PCL is the standard for point cloud processing

**Trade-offs**:
- ✅ Excellent performance for large datasets
- ✅ Direct compatibility with PCL ecosystem
- ❌ More complex memory management
- ❌ Requires careful lifetime management

**Alternative Considered**: Native Rust point cloud types
- Rejected due to performance concerns and PCL ecosystem compatibility

### 3. Builder Pattern for Configuration

**Decision**: All algorithms use builders for configuration instead of constructor parameters.

**Rationale**:
- Point cloud registration algorithms have many optional parameters
- Builder pattern provides discoverable, self-documenting API
- Allows for future parameter additions without breaking changes
- Common Rust idiom that users expect

**Example**:
```rust
let gicp = FastGICP::builder()
    .max_iterations(50)
    .transformation_epsilon(1e-6)
    .regularization_method(RegularizationMethod::Frobenius)
    .build()?;
```

**Alternative Considered**: Configuration structs
- Rejected because less discoverable and harder to extend

### 4. Direct Registration API

**Decision**: The `align()` method takes source and target clouds directly.

**Rationale**:
- Simplifies the most common use case
- Avoids stateful "set input" pattern from C++ API
- More functional programming style
- Reduces potential for user errors

**Before (C++ style)**:
```rust
gicp.set_input_source(&source);
gicp.set_input_target(&target);
let result = gicp.align()?;
```

**After (Direct style)**:
```rust
let result = gicp.align(&source, &target)?;
```

### 5. Conditional Documentation Tests

**Decision**: Use `cfg_attr` to make doc tests conditional on feature flags.

**Rationale**:
- docs.rs builds use stubs that contain `unreachable!()` macros
- Doc tests should run in development but not in docs-only builds
- Maintains single source of truth for documentation
- No code duplication required

**Implementation**:
```rust
#[cfg_attr(feature = "docs-only", doc = "```no_run")]
#[cfg_attr(not(feature = "docs-only"), doc = "```")]
/// use fast_gicp::FastGICP;
/// let gicp = FastGICP::new();
/// ```
```

## Error Handling Strategy

### Result Types for Fallible Operations

**Decision**: Only operations that can actually fail return `Result` types.

**Categories**:
- **Infallible**: Point cloud creation, algorithm construction
- **Fallible**: Registration operations, builder validation

**Rationale**:
- Reduces cognitive overhead for users
- Makes the API more pleasant to use for simple cases
- Clearly indicates which operations can fail

### Error Propagation

**Decision**: Use custom error types with `thiserror` for clear error messages.

**Design**:
```rust
#[derive(Error, Debug)]
pub enum Error {
    #[error("Point cloud is empty")]
    EmptyPointCloud,
    #[error("Registration failed to converge after {iterations} iterations")]
    ConvergenceFailed { iterations: i32 },
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
}
```

## Memory Management Strategy

### RAII with Drop Trait

**Decision**: All C++ resources are managed through Rust's Drop trait.

**Implementation**:
- C++ objects wrapped in `UniquePtr<T>` from cxx
- Drop trait automatically calls C++ destructor
- No manual resource management required from users

### Thread Safety

**Decision**: Algorithms are `!Send` and `!Sync`.

**Rationale**:
- Matches C++ library semantics (not thread-safe)
- Prevents data races at compile time
- Users must create separate instances for multi-threading
- Explicit design choice to maintain safety

## Feature Flag Architecture

### Minimal Default Features

**Decision**: Only enable OpenMP by default, make CUDA optional.

**Rationale**:
- OpenMP is widely available and lightweight
- CUDA requires significant system dependencies
- Allows users to opt into complexity as needed

### docs-only Feature

**Decision**: Provide docs-only feature for docs.rs compatibility.

**Problem Solved**: docs.rs cannot compile C++ code, breaking documentation builds.

**Solution**:
- Pre-generated stub files that provide API-compatible interfaces
- Stubs contain `unreachable!()` to prevent accidental use
- Separate stubs for CUDA and non-CUDA configurations

## Type System Design

### Strong Type Safety

**Decision**: Use distinct Rust types for different point cloud types and algorithms.

**Benefits**:
- Compile-time prevention of type confusion
- Clear API that self-documents capabilities
- Leverages Rust's type system for correctness

### Enum-Based Configuration

**Decision**: Use Rust enums for algorithm parameters instead of constants.

**Example**:
```rust
pub enum RegularizationMethod {
    None,
    Frobenius,
    SvdRemoval,
}
```

**Benefits**:
- Compile-time validation of parameter values
- Exhaustive matching ensures all cases handled
- Self-documenting API

## Performance Design Decisions

### Lazy Initialization

**Decision**: Defer expensive operations until registration time.

**Rationale**:
- Builder pattern creation should be fast
- KD-tree construction and voxelization are expensive
- Allows for better error reporting during registration

### Memory Layout

**Decision**: Use `#[repr(C)]` for FFI types to ensure compatible memory layout.

**Critical for**:
- Point cloud data structures
- Transform matrices
- Configuration parameters

## Testing Strategy

### Multi-Level Testing

**Decision**: Test at multiple levels of abstraction.

**Levels**:
1. **Unit Tests**: Individual components and functions
2. **Integration Tests**: End-to-end registration workflows  
3. **Property Tests**: Algorithm correctness with known transformations
4. **Performance Tests**: Benchmarks for regression detection

### Conditional Test Execution

**Decision**: Some tests only run with specific features enabled.

**Implementation**:
```rust
#[cfg(feature = "cuda")]
mod cuda_tests {
    // CUDA-specific tests
}
```

## Build System Design

### CMake Integration

**Decision**: Use CMake for C++ dependency management and compilation.

**Rationale**:
- fast_gicp C++ library uses CMake
- Excellent system dependency detection (PCL, Eigen3, CUDA)
- Cross-platform support for future expansion

### Feature Detection

**Decision**: Automatic feature detection with manual override capability.

**Implementation**:
- CMake detects available system capabilities
- Environment variables allow manual override
- Graceful degradation when dependencies missing

## Documentation Strategy

### Example-Heavy Documentation

**Decision**: Include working examples in all public API documentation.

**Rationale**:
- Point cloud registration is complex domain
- Examples demonstrate correct usage patterns
- Doc tests ensure examples stay current

### Separate Design Documentation

**Decision**: Maintain detailed design documents separate from code.

**Structure**:
- `ARCH.md`: System architecture and component relationships
- `DESIGN.md`: This document - design decisions and rationale
- `DESIGN_CODEGEN.md`: Code generation system details
- `PROGRESS.md`: Implementation tracking and project status
- `DEV.md`: Development workflow and build instructions

## Future-Proofing Decisions

### Workspace Structure

**Decision**: Use Cargo workspace with separate sys and high-level crates.

**Benefits**:
- Clear separation of concerns
- Allows independent versioning if needed
- Standard pattern in Rust FFI ecosystem

### API Versioning

**Decision**: Design for semantic versioning compatibility.

**Principles**:
- Builder pattern allows adding new optional parameters
- Enums use `#[non_exhaustive]` for future variants
- Error types designed for extension
- Public API surface kept minimal initially

### Platform Extensibility

**Decision**: Design for future cross-platform support.

**Current**: Linux-focused with system dependencies
**Future**: Windows/macOS support with vendored dependencies
**Strategy**: Abstract platform-specific concerns behind traits