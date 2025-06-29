# Code Generation System Design

## Overview

The Fast GICP Rust wrapper includes a sophisticated code generation system to enable documentation builds on docs.rs without requiring C++ compilation. This document details the design, implementation, and usage of this system.

## Problem Statement

### docs.rs Limitations

**Challenge**: docs.rs cannot compile C++ code, making it impossible to build documentation for FFI-heavy crates.

**Impact**:
- Documentation builds fail on docs.rs
- Users cannot browse API documentation online
- Reduces adoption and usability of the crate

**Requirements**:
- Generate documentation without C++ compilation
- Maintain API compatibility with real implementation
- Support conditional features (CUDA vs non-CUDA)
- Preserve all type information and signatures

## Solution Architecture

### Dual-Stub System

**Core Concept**: Pre-generate API-compatible stub files that provide the same interface as the real FFI implementation.

```
Feature Configuration → Stub Selection → Documentation Build
─────────────────────────────────────────────────────────────
docs-only            → stub.rs       → Core APIs only
docs-only + cuda     → stub_cuda.rs  → Core + CUDA APIs
```

### Implementation Components

1. **Stub Generator**: Parses CXX bridge and generates compatible stubs
2. **Feature Detection**: Selects appropriate stub based on enabled features  
3. **Build Integration**: Seamlessly switches between real FFI and stubs
4. **Testing Framework**: Validates stub compatibility and functionality

## Stub Generation Process

### Phase 1: CXX Bridge Analysis

**Input**: `src/lib.rs` containing `#[cxx::bridge]` definition

**Process**:
1. Parse Rust source using `syn` crate
2. Extract `cxx::bridge` module contents
3. Identify all extern types, functions, and their signatures
4. Categorize items as core, CUDA-specific, or conditional

**Example Input**:
```rust
#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        type FastGICP;
        type FastVGICPCuda;  // CUDA-specific
        
        fn create_fast_gicp() -> UniquePtr<FastGICP>;
        
        #[cfg(feature = "cuda")]
        fn create_fast_vgicp_cuda() -> UniquePtr<FastVGICPCuda>;
    }
}
```

### Phase 2: CUDA Item Detection

**Strategy**: Identify CUDA-specific items using multiple heuristics

**Detection Methods**:
1. **Attribute-based**: Items with `#[cfg(feature = "cuda")]`
2. **Naming patterns**: Types/functions containing "Cuda", "cuda", "NDT"
3. **Explicit mapping**: Hardcoded list of known CUDA items

**Implementation**:
```rust
fn is_cuda_item(item_name: &str) -> bool {
    const CUDA_PATTERNS: &[&str] = &[
        "cuda", "Cuda", "CUDA", "ndt", "NDT", "FastVGICPCuda", "NDTCuda"
    ];
    CUDA_PATTERNS.iter().any(|pattern| item_name.contains(pattern))
}
```

### Phase 3: Stub Code Generation

**Target**: Generate API-compatible Rust code that compiles without C++

**Generated Components**:

1. **Opaque Types**: 
```rust
#[repr(C)]
pub struct FastGICP {
    _private: ::cxx::private::Opaque,
}
```

2. **Trait Implementations**:
```rust
unsafe impl ::cxx::ExternType for FastGICP {
    type Id = ::cxx::type_id!(FastGICP);
    type Kind = ::cxx::kind::Opaque;
}

unsafe impl ::cxx::private::UniquePtrTarget for FastGICP {
    fn __typename(f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.write_str("FastGICP")
    }
    fn __null() -> ::cxx::private::MaybeUninit<*mut ::core::ffi::c_void> {
        ::cxx::private::MaybeUninit::new(::core::ptr::null_mut())
    }
    unsafe fn __raw(raw: *mut ::core::ffi::c_void) -> ::cxx::private::MaybeUninit<*mut ::core::ffi::c_void> {
        ::cxx::private::MaybeUninit::new(raw)
    }
    unsafe fn __get(repr: ::cxx::private::MaybeUninit<*mut ::core::ffi::c_void>) -> *const Self {
        ::cxx::private::MaybeUninit::assume_init(repr) as *const Self
    }
    unsafe fn __release(repr: ::cxx::private::MaybeUninit<*mut ::core::ffi::c_void>) -> *mut Self {
        ::cxx::private::MaybeUninit::assume_init(repr) as *mut Self
    }
    unsafe fn __drop(_repr: ::cxx::private::MaybeUninit<*mut ::core::ffi::c_void>) {
        unreachable!("docs-only stub")
    }
}
```

3. **Function Stubs**:
```rust
pub fn create_fast_gicp() -> ::cxx::UniquePtr<FastGICP> {
    unreachable!("docs-only stub")
}
```

### Phase 4: Feature-Specific Filtering

**Process**: Generate two stub variants with different item sets

**stub.rs (Non-CUDA)**:
- Includes all core types and functions
- Excludes CUDA-specific items
- Used when only `docs-only` feature is enabled

**stub_cuda.rs (CUDA-enabled)**:  
- Includes all core types and functions
- Includes CUDA-specific items
- Used when both `docs-only` and `cuda` features are enabled

## Build System Integration

### Feature Detection Logic

**Location**: `fast-gicp-sys/src/lib.rs`

```rust
// Normal builds: use actual cxx::bridge directly
#[cfg(not(feature = "docs-only"))]
#[cxx::bridge]
pub mod ffi {
    // Real FFI definitions
}

// Docs-only builds: use appropriate stub based on CUDA feature
#[cfg(all(feature = "docs-only", not(feature = "cuda")))]
include!("generated/stub.rs");

#[cfg(all(feature = "docs-only", feature = "cuda"))]
include!("generated/stub_cuda.rs");
```

### Build Script Coordination

**Location**: `fast-gicp-sys/build.rs`

**Logic**:
```rust
fn main() {
    if std::env::var("CARGO_FEATURE_DOCS_ONLY").is_ok() {
        println!("cargo:warning=docs-only mode: skipping C++ compilation, using pre-generated bindings");
        return;
    }
    
    // Normal C++ compilation...
}
```

## Stub Generation Tool

### Makefile Integration

**Command**: `make generate-stubs`

**Implementation**:
```makefile
.PHONY: generate-stubs
generate-stubs:
	@echo "🔧 Generating API stubs for docs.rs compatibility..."
	cd fast-gicp-sys && cargo build --features bindgen
	@echo "📝 Generated stub.rs (non-CUDA)"
	@echo "📝 Generated stub_cuda.rs (CUDA)"
```

### Validation Process

**Command**: `make verify-stubs`

**Checks**:
1. Both stub files compile successfully
2. CUDA stub contains more items than non-CUDA stub
3. All expected CUDA types present in CUDA stub
4. No CUDA types present in non-CUDA stub

### Testing Integration

**Challenge**: Stubs contain `unreachable!()` macros that panic when called

**Solution**: Compilation-only testing for stub validation

```makefile
test-stubs:
	@echo "🧪 Testing stub compilation..."
	cargo check --features docs-only --no-default-features --lib
	cargo check --features "docs-only cuda" --no-default-features --lib
	@echo "✅ Both stub variants compile successfully"
```

## API Compatibility Requirements

### Type Compatibility

**Requirement**: Generated stubs must provide identical public API surface

**Implementation**:
- Same struct names and visibility
- Same function signatures and return types
- Same trait implementations for cxx compatibility
- Same module structure and re-exports

### Memory Layout Compatibility

**Requirement**: Types must have compatible memory representation

**Solution**:
- Use `#[repr(C)]` for all opaque types
- Use `::cxx::private::Opaque` for internal representation
- Implement all required cxx traits correctly

### Error Compatibility

**Requirement**: Stub functions must fail predictably if accidentally called

**Solution**: Use `unreachable!("docs-only stub")` with descriptive messages

## Documentation Test Strategy

### Problem: Conflicting Requirements

**In Development**: Doc tests should execute to verify examples work
**In docs.rs**: Doc tests should not execute because stubs panic

### Solution: Conditional Doc Tests

**Implementation**: Use `cfg_attr` to conditionally change doc test behavior

```rust
#[cfg_attr(feature = "docs-only", doc = "```no_run")]
#[cfg_attr(not(feature = "docs-only"), doc = "```")]
/// use fast_gicp::FastGICP;
/// let gicp = FastGICP::new();
/// ```
pub struct FastGICP { /* ... */ }
```

**Behavior**:
- **With `docs-only`**: Code shows in docs as `no_run` (visible but not executed)
- **Without `docs-only`**: Code executes as normal doc test

## Maintenance and Evolution

### Adding New FFI Items

**Process**:
1. Add new items to `#[cxx::bridge]` in `src/lib.rs`
2. Run `make generate-stubs` to update stub files
3. Run `make verify-stubs` to ensure correctness
4. Commit both real bridge changes and updated stubs

### CUDA Feature Changes

**Process**:
1. Update CUDA detection patterns in stub generator if needed
2. Regenerate stubs with `make generate-stubs`
3. Verify CUDA items are correctly categorized
4. Test both stub variants compile

### Debugging Stub Issues

**Common Issues**:
1. **Compilation errors**: Usually missing trait implementations
2. **API mismatches**: Generated signatures don't match expectations
3. **Feature detection**: Items incorrectly categorized as CUDA/non-CUDA

**Debugging Tools**:
```bash
# Check specific stub variant
cargo check --features docs-only --no-default-features --lib

# Verbose stub generation
RUST_LOG=debug make generate-stubs

# Compare generated stubs
diff fast-gicp-sys/src/generated/stub.rs fast-gicp-sys/src/generated/stub_cuda.rs
```

## Performance Considerations

### Build Time Impact

**Stub Generation**: Minimal overhead (< 1 second)
**docs.rs Builds**: Significantly faster without C++ compilation
**Development Builds**: No impact (stubs not used)

### Maintenance Overhead

**Frequency**: Stubs only need regeneration when FFI bridge changes
**Automation**: Fully automated through Makefile targets
**Validation**: Automated testing ensures stub correctness

## Future Enhancements

### Advanced Code Generation

**Potential Improvements**:
1. **Smart Return Values**: Generate meaningful default values instead of panics
2. **Partial Functionality**: Implement some operations using pure Rust
3. **Mock Integration**: Integration with testing frameworks

### Cross-Platform Stubs

**Goal**: Generate platform-specific stubs for Windows/macOS compatibility
**Approach**: Extend generator to handle platform-specific FFI items

### IDE Integration

**Goal**: Better development experience with generated code
**Approach**: Generate rust-analyzer compatible metadata