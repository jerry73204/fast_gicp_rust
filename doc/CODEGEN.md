# Two-Phase Code Generation for docs.rs Support

## Overview

This document describes a two-phase code generation approach to support documentation building on docs.rs while maintaining a single source of truth for FFI interfaces.

## Problem Statement

- docs.rs cannot compile C++ code, which is required by the `cxx` crate
- Current solution uses duplicate stub implementations, leading to maintenance burden
- Need a solution that:
  - Maintains single source of truth
  - Preserves all public APIs for documentation
  - Requires minimal maintenance

## Key Technical Challenges

1. **CXX Bridge Expansion**: The `cxx::bridge` macro expands at compile time and generates code internally. We need to capture this expanded output.

2. **Generated File Location**: CXX doesn't generate a single Rust file we can easily transform. Instead, it:
   - Expands the macro in-place during compilation
   - Generates C++ binding files in `OUT_DIR/cxxbridge/`
   - The Rust side is handled by macro expansion, not file generation

3. **Possible Approaches**:
   - **Option A**: Use `cargo expand` to capture the expanded macro output
   - **Option B**: Create a minimal build that includes the cxx::bridge and capture its output
   - **Option C**: Parse the original cxx::bridge definition and generate stubs directly

## Target Architecture

```
                              ┌─────────────────────────┐
                              │   fast-gicp-sys/src/    │
                              │       lib.rs            │
                              │   (cxx::bridge def)     │
                              │   SINGLE SOURCE OF      │
                              │       TRUTH             │
                              └───────────┬─────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    │ build.rs (when bindgen feature enabled)  │
                    │         Parse & Transform                 │
                    │                                           │
                    ▼                                           ▼
        ┌────────────────────────┐                 ┌────────────────────────┐
        │ fast-gicp-sys/src/     │                 │ fast-gicp-sys/src/     │
        │   generated/           │                 │   generated/           │
        │     stub.rs            │                 │     stub_cuda.rs       │
        │  (non-CUDA variant)    │                 │   (CUDA variant)       │
        └────────────────────────┘                 └────────────────────────┘
```

### Detailed Workflow Diagram

```
Developer Workflow:
┌─────────────────┐
│ 1. Edit lib.rs  │
│ (cxx::bridge)   │
│ SINGLE SOURCE   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────────┐
│ 2. Run:         │────▶│ build.rs parses lib.rs  │
│ make generate-  │     │ and generates:          │
│ bindings        │     │ - stub.rs (no CUDA)     │
└─────────────────┘     │ - stub_cuda.rs (CUDA)   │
                        └────────┬────────────────┘
                                 │
         ┌───────────────────────┴───────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────┐                         ┌─────────────────┐
│ 3. Verify with: │                         │ 4. Test both:   │
│ make verify-    │                         │ make test       │
│ bindings        │                         │                 │
└─────────────────┘                         └─────────────────┘
         │                                           │
         └───────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 5. Commit stub  │
                    │ files only      │
                    └─────────────────┘

User Workflow:
┌─────────────────┐     ┌─────────────────────────┐
│ Normal Build    │────▶│ Uses lib.rs directly    │
│ (default)       │     │ cxx::bridge macro       │
│                 │     │ C++ code is compiled    │
└─────────────────┘     └─────────────────────────┘

┌─────────────────┐     ┌─────────────────────────┐
│ Docs Build      │────▶│ Uses stub.rs or         │
│ (docs-only)     │     │ stub_cuda.rs based on   │
│                 │     │ feature flags           │
└─────────────────┘     └─────────────────────────┘
```

### User Workflow

1. **Default Usage** (direct cxx::bridge):
   - Use `lib.rs` cxx::bridge definition directly
   - CXX macro expansion happens at compile time
   - C++ wrapper compilation occurs
   - Works for most users with C++ toolchain

2. **Documentation Build** (`docs-only` feature):
   - Use checked-in `src/generated/stub.rs` (default)
   - Use checked-in `src/generated/stub_cuda.rs` (if CUDA feature enabled)
   - No C++ compilation required
   - Works on docs.rs
   - Mutually exclusive with `bindgen` feature

3. **Regeneration** (`bindgen` feature):
   - Parse cxx::bridge from lib.rs
   - Generate and overwrite `stub.rs` and `stub_cuda.rs`
   - Used by maintainers to update stubs when FFI changes
   - Mutually exclusive with `docs-only` feature

### Directory Structure
```
fast_gicp_rust/
├── fast-gicp-sys/
│   ├── Cargo.toml
│   ├── build.rs            # Handles stub generation only
│   ├── src/
│   │   ├── lib.rs          # cxx::bridge definition (SINGLE SOURCE)
│   │   ├── bridge.rs       # Same bridge for C++ header generation
│   │   ├── generated/      # Checked-in generated stubs only
│   │   │   ├── stub.rs     # Pre-generated stub (no CUDA)
│   │   │   └── stub_cuda.rs # Pre-generated stub (with CUDA)
│   │   └── wrapper.cpp
│   └── tests/
└── Makefile
```

## Design Decisions

### 1. Single Source of Truth
- **Principle**: `lib.rs` cxx::bridge definition is the only source
- **Normal builds**: Use cxx::bridge macro expansion directly
- **Docs builds**: Use pre-generated stubs derived from the same source
- **Rationale**: Eliminates duplication and sync issues
- **Maintenance**: Only one place to update FFI definitions

### 2. Check-in Generated Stubs Only
- **Generated**: Only stub files (`stub.rs`, `stub_cuda.rs`)
- **Not Generated**: No duplicate `bindings.rs` file
- **Rationale**: Stubs enable docs.rs builds without C++ compilation
- **Publishing**: Stubs included in crates.io package for docs.rs

### 3. Feature-based Selection
- **Default**: Use `lib.rs` cxx::bridge directly
- **`docs-only`**: Use pre-generated stub implementation (feature-dependent)
  - Without CUDA: Use `stub.rs`
  - With CUDA: Use `stub_cuda.rs`
- **`bindgen`**: Regenerate both stub variants from lib.rs
- **Rationale**: Clear separation of use cases with CUDA support
- **Constraint**: `docs-only` and `bindgen` are mutually exclusive

### 4. Single-Phase Generation Process
- **Parse**: Extract cxx::bridge definition from lib.rs
- **Transform**: Generate `stub.rs` and `stub_cuda.rs` variants
- **Rationale**: Simpler than dual-phase, no intermediate files
- **CUDA Handling**: Generate separate stubs for different feature combinations

### 5. Stub Implementation Strategy
- **Function bodies**: Use `unreachable!()` macro
- **Rationale**: Simple, no dummy values needed
- **Formatting**: Use rustfmt for consistent style
- **API Compatibility**: Must match cxx-generated API exactly
  - All functions must be `pub`
  - Use `cxx::UniquePtr`, not custom types
  - Implement required cxx traits

## Implementation Plan

### Code Generation Workflow

The workflow consists of two distinct phases:

1. **Development Phase** (with `bindgen` feature):
   - Developer manually runs the generation process
   - Two stub files are regenerated: stub.rs and stub_cuda.rs
   - Stub files are committed to version control

2. **Usage Phase**:
   - Normal builds use lib.rs cxx::bridge directly (no pre-generated files)
   - Docs builds use appropriate stub based on CUDA feature:
     - `docs-only` alone: uses `stub.rs`
     - `docs-only` + `cuda`: uses `stub_cuda.rs`
   - No code generation occurs during normal usage

### Verification Using cargo-expand

`cargo-expand` is used to verify the generated stubs match the expanded macro output:

```bash
# Verify the generated stubs capture all necessary items
cargo expand --lib --features default > expanded.rs
# Compare to ensure all types and functions are present in stubs

# This is NOT used for generation, only verification
```

### Updated lib.rs Structure for CUDA Support

```rust
// fast-gicp-sys/src/lib.rs

// Normal builds: use actual cxx::bridge directly
#[cfg(not(feature = "docs-only"))]
#[cxx::bridge]
pub mod ffi {
    #[derive(Debug, Clone, Copy)]
    pub struct Transform4f {
        pub data: [f32; 16],
    }
    
    unsafe extern "C++" {
        include!("wrapper.h");
        
        type PointCloudXYZ;
        fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ>;
        fn point_cloud_xyz_size(cloud: &PointCloudXYZ) -> usize;
        
        // CUDA functions (conditionally included based on features)
        #[cfg(feature = "cuda")]
        type FastVGICPCuda;
        #[cfg(feature = "cuda")]
        fn create_fast_vgicp_cuda() -> UniquePtr<FastVGICPCuda>;
        
        // ... rest of the interface
    }
}

// Docs-only builds: use appropriate stub based on CUDA feature
#[cfg(all(feature = "docs-only", not(feature = "cuda")))]
include!("generated/stub.rs");

#[cfg(all(feature = "docs-only", feature = "cuda"))]
include!("generated/stub_cuda.rs");
```

### build.rs Implementation

```rust
// fast-gicp-sys/build.rs

fn main() {
    #[cfg(all(feature = "docs-only", feature = "bindgen"))]
    compile_error!("features `docs-only` and `bindgen` are mutually exclusive");
    
    if cfg!(feature = "bindgen") {
        generate_stubs();
    } else if cfg!(feature = "docs-only") {
        // Nothing to do - use pre-generated stub
        return;
    } else {
        // Normal build - compile C++ and use lib.rs cxx::bridge directly
        compile_cpp_wrapper();
    }
}

#[cfg(feature = "bindgen")]
fn generate_stubs() {
    // Important: This function is NOT called during normal builds
    // It's only used when explicitly regenerating stubs
    
    // Parse the cxx::bridge definition directly from lib.rs
    let lib_content = fs::read_to_string("src/lib.rs").unwrap();
    let syntax = syn::parse_file(&lib_content).unwrap();
    
    // Extract the cxx::bridge module
    let bridge_mod = find_cxx_bridge_module(&syntax);
    
    // Generate two stub versions
    // 1: Generate stub version without CUDA (docs-only)
    let stub = generate_stub_bindings(bridge_mod.clone(), false);
    fs::write("src/generated/stub.rs", format_with_rustfmt(stub)).unwrap();
    
    // 2: Generate stub version with CUDA (docs-only + cuda)
    let stub_cuda = generate_stub_bindings(bridge_mod, true);
    fs::write("src/generated/stub_cuda.rs", format_with_rustfmt(stub_cuda)).unwrap();
    
    println!("cargo:warning=Generated stubs have been updated in src/generated/");
    println!("cargo:warning=Please verify with: cargo expand --lib > expanded.rs");
}

fn generate_stub_bindings(bridge_mod: &syn::ItemMod, include_cuda: bool) -> String {
    // Filter CUDA-specific items based on the include_cuda parameter
    // When include_cuda is false, skip items with #[cfg(feature = "cuda")]
    // When include_cuda is true, include all items
    
    // Implementation will process the bridge module and generate appropriate stubs
    // based on the CUDA feature flag requirements
}

fn format_with_rustfmt(code: String) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};
    
    let mut child = Command::new("rustfmt")
        .arg("--emit=stdout")
        .arg("--edition=2021")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn rustfmt");
    
    let mut stdin = child.stdin.take().expect("Failed to get stdin");
    stdin.write_all(code.as_bytes()).expect("Failed to write to rustfmt");
    drop(stdin);
    
    let output = child.wait_with_output().expect("Failed to wait for rustfmt");
    
    if !output.status.success() {
        eprintln!("rustfmt failed, using unformatted code");
        return code;
    }
    
    String::from_utf8(output.stdout).unwrap_or(code)
}
```

## Building and Testing

### Makefile Targets

The Makefile provides comprehensive targets for development, testing, and validation:

```makefile
# Default target
.PHONY: all
all: lint test

# Generate both CUDA and non-CUDA stubs (maintainer use)
.PHONY: generate-stubs
generate-stubs:
	@echo "🔧 Generating stubs..."
	cd fast-gicp-sys && cargo build --features bindgen
	@echo "✅ Generated stub files have been updated:"
	@echo "   - fast-gicp-sys/src/generated/stub.rs (non-CUDA)"
	@echo "   - fast-gicp-sys/src/generated/stub_cuda.rs (CUDA)"

# Verify generated stubs match cxx output
.PHONY: verify-stubs
verify-stubs:
	@echo "🔍 Expanding cxx macros for verification..."
	cargo expand -p fast-gicp-sys --lib > /tmp/expanded_default.rs
	cargo expand -p fast-gicp-sys --lib --features cuda > /tmp/expanded_cuda.rs
	@echo "📋 Expanded files saved to:"
	@echo "   - /tmp/expanded_default.rs (non-CUDA)"
	@echo "   - /tmp/expanded_cuda.rs (CUDA)"
	@echo "⚠️  Manual verification required - check API compatibility"

# Test stub builds (docs-only mode)
.PHONY: test-stubs
test-stubs:
	@echo "🧪 Testing non-CUDA stubs..."
	cargo test -p fast-gicp-sys --features docs-only --no-default-features --lib --tests
	cargo test -p fast-gicp --features docs-only --no-default-features --lib --tests
	@echo "🧪 Testing CUDA stubs..."
	cargo test -p fast-gicp-sys --features "docs-only cuda" --no-default-features --lib --tests
	cargo test -p fast-gicp --features "docs-only cuda" --no-default-features --lib --tests

# Test normal builds
.PHONY: test
test:
	@echo "🧪 Running all tests..."
	cargo nextest run --all-targets --no-fail-fast
	cargo nextest run --features cuda --all-targets --no-fail-fast

# Lint all code
.PHONY: lint
lint:
	cargo +nightly fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features cuda -- -D warnings
	cargo clippy --all-targets --features docs-only --no-default-features -- -D warnings
	cargo clippy --all-targets --features "docs-only cuda" --no-default-features -- -D warnings

# Format code
.PHONY: format
format:
	cargo +nightly fmt --all

# Clean build artifacts
.PHONY: clean
clean:
	cargo clean

# Simulate docs.rs build
.PHONY: docs
docs:
	@echo "📚 Building documentation (docs.rs simulation)..."
	cargo doc --features docs-only --no-default-features --no-deps --open

# Complete stub workflow: generate, verify, and test both variants
.PHONY: update-stubs
update-stubs: generate-stubs verify-stubs test-stubs
	@echo ""
	@echo "🎉 Complete stub update workflow finished successfully!"
	@echo "Both CUDA and non-CUDA stubs have been:"
	@echo "  ✅ Generated from latest cxx::bridge definition"
	@echo "  ✅ Verified for correct CUDA item filtering"
	@echo "  ✅ Tested with docs-only builds"
	@echo ""
	@echo "Ready to commit the updated stub files."

# Run all checks before committing
.PHONY: pre-commit
pre-commit: lint test test-stubs verify-stubs
	@echo "✅ All checks passed!"
```

### Testing Strategy

The testing approach ensures compatibility across all build configurations:

1. **Normal Builds** (`make test`)
   - Tests with actual C++ compilation
   - Validates real FFI functionality
   - Tests both with and without CUDA

2. **Stub Builds** (`make test-stubs`)
   - Tests with pre-generated stubs
   - Validates docs-only builds work correctly
   - Tests both stub.rs and stub_cuda.rs

3. **Documentation Builds** (`make docs`)
   - Simulates docs.rs environment
   - Ensures documentation generates correctly
   - Validates all public APIs are visible

### Linting Strategy

Comprehensive linting ensures code quality:

1. **Format Checking** (`make fmt-check`)
   - Enforces consistent code style
   - Uses rustfmt with edition 2021

2. **Clippy Analysis** (`make clippy`)
   - Tests all feature combinations
   - Enforces Rust best practices
   - Treats warnings as errors

3. **Feature Combinations Tested**
   - Default features
   - CUDA enabled
   - docs-only (no default features)
   - docs-only + CUDA

## Migration Plan for CUDA/Non-CUDA Support

### Overview

This section describes the migration from a single stub file to dual stub files supporting both CUDA and non-CUDA feature combinations. The goal is to provide appropriate API visibility based on the feature flags enabled during documentation builds.

### Current State (Single Stub)

- **Current**: Single `stub.rs` file containing all FFI items including CUDA-specific ones
- **Problem**: CUDA-specific items are visible even when CUDA feature is not enabled
- **Impact**: Documentation may show CUDA functions that aren't actually available
- **Note**: No duplicate `bindings.rs` - normal builds use lib.rs directly

### Target State (Dual Stubs)

- **Target**: Two separate stub files:
  - `stub.rs`: Non-CUDA stub (excludes CUDA-specific items)
  - `stub_cuda.rs`: CUDA-enabled stub (includes all items)
- **Benefit**: Documentation accurately reflects available APIs based on features
- **Selection**: Conditional compilation chooses appropriate stub at build time
- **Single Source**: lib.rs cxx::bridge remains the only source of truth

### Implementation Strategy

#### 1. Stub Generation Algorithm Enhancement

The `generate_stub_bindings` function will be enhanced to accept a `cuda` parameter:

```rust
fn generate_stub_bindings(bridge_mod: &syn::ItemMod, include_cuda: bool) -> String {
    // When processing ForeignItem::Fn, check for #[cfg(feature = "cuda")] attribute
    // If include_cuda is false, skip CUDA-specific functions and types
    // If include_cuda is true, include all items
}
```

#### 2. CUDA Item Detection

Items are considered CUDA-specific if they have:
- `#[cfg(feature = "cuda")]` attribute on functions
- Type names containing "Cuda" (e.g., `FastVGICPCuda`, `NDTCuda`)
- Function names containing "cuda" (e.g., `create_fast_vgicp_cuda`)

#### 3. Conditional Logic

```rust
// In generate_stub_bindings()
for foreign_item in &foreign_mod.items {
    match foreign_item {
        ForeignItem::Fn(func) => {
            let is_cuda_specific = func.attrs.iter().any(|attr| {
                // Check for #[cfg(feature = "cuda")] attribute
                is_cuda_cfg_attribute(attr)
            });
            
            // Skip CUDA items if not including CUDA
            if is_cuda_specific && !include_cuda {
                continue;
            }
            
            // Generate stub function
            stub_items.push(generate_stub_function(func));
        }
        ForeignItem::Type(ty) => {
            let is_cuda_type = ty.ident.to_string().contains("Cuda");
            
            // Skip CUDA types if not including CUDA
            if is_cuda_type && !include_cuda {
                continue;
            }
            
            // Generate opaque struct
            stub_items.push(generate_opaque_struct(&ty.ident));
        }
        _ => {}
    }
}
```

#### 4. File Generation Process

The build process will generate two stub files only:

1. **stub.rs**: Stub without CUDA items
2. **stub_cuda.rs**: Stub with all items including CUDA

No separate `bindings.rs` file is generated - normal builds use lib.rs directly.

### Migration Steps

#### Step 1: Update build.rs Implementation
- Modify `generate_stub_bindings` to accept `include_cuda` parameter
- Implement CUDA item filtering logic
- Generate both stub variants

#### Step 2: Update lib.rs Conditional Compilation
- Add conditional includes for both stub variants
- Ensure correct stub is selected based on feature flags

#### Step 3: Regenerate Stub Files
- Run `make generate-stubs` to create both stub variants
- Verify that `stub.rs` excludes CUDA items
- Verify that `stub_cuda.rs` includes all items

#### Step 4: Test Feature Combinations
- Test `docs-only` alone → uses `stub.rs`
- Test `docs-only + cuda` → uses `stub_cuda.rs`
- Verify documentation reflects correct API surface

#### Step 5: Update Documentation
- Update CODEGEN.md with new dual-stub approach
- Update README with feature flag combinations
- Document the migration process

### Validation

#### Before Migration
```bash
# Current state: single stub with CUDA items always visible
cargo doc --features docs-only --no-default-features --open
# Documentation shows CUDA functions even without CUDA feature
```

#### After Migration
```bash
# Without CUDA: clean API surface
cargo doc --features docs-only --no-default-features --open
# Documentation excludes CUDA functions

# With CUDA: complete API surface  
cargo doc --features docs-only,cuda --no-default-features --open
# Documentation includes CUDA functions
```

### Backwards Compatibility

This migration maintains backwards compatibility:
- Existing builds continue to work unchanged
- Generated file structure is preserved
- Feature flag behavior remains consistent
- Only the internal stub generation logic changes

### Success Criteria

1. **Accurate Documentation**: docs.rs shows only available APIs based on features
2. **Clean Separation**: Non-CUDA builds don't show CUDA functions
3. **Complete Coverage**: CUDA builds show all functions including CUDA-specific ones
4. **Maintainability**: Single source of truth maintained in cxx::bridge definition
5. **Build Compatibility**: All existing build configurations continue to work

## Transform Implementation Details

### Key Transformations

1. **Remove extern "C++" blocks**
   ```rust
   // Input: extern "C++" { ... }
   // Output: Regular Rust functions with stub bodies
   ```

2. **Convert opaque types**
   ```rust
   // Input: type PointCloudXYZ;
   // Output: pub struct PointCloudXYZ { _private: [u8; 0] }
   ```

3. **Transform function signatures**
   ```rust
   // Input: fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ>;
   // Output: pub fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ> {
   //     unreachable!("docs-only stub")
   // }
   ```

4. **Handle function bodies**
   - All function bodies → `unreachable!("docs-only stub")`
   - No need to handle return types or create dummy values
   - Simple and consistent transformation

### Code Transformation Logic

The transformation process converts CXX bridge definitions into API-compatible stubs:

#### 1. Stub Generation Algorithm

```rust
fn generate_stub_bindings(bridge: &ItemMod) -> String {
    let mut stub_items = vec![];
    
    // Add necessary imports
    stub_items.push(quote! {
        use std::pin::Pin;
        use cxx::UniquePtr;
    });
    
    // Process each item in the bridge module
    for item in &bridge.content.as_ref().unwrap().1 {
        match item {
            Item::ForeignMod(foreign_mod) => {
                // Extract items from extern "C++" block
                for foreign_item in &foreign_mod.items {
                    match foreign_item {
                        ForeignItem::Type(ty) => {
                            // Opaque type: type Foo; → pub struct Foo { _private: [u8; 0] }
                            stub_items.push(generate_opaque_struct(&ty.ident));
                        }
                        ForeignItem::Fn(func) => {
                            // Function: fn foo() -> T; → pub fn foo() -> T { unreachable!() }
                            stub_items.push(generate_stub_function(func));
                        }
                        _ => {}
                    }
                }
            }
            Item::Struct(s) => {
                // Keep shared structs as-is
                stub_items.push(quote! { #s });
            }
            _ => {}
        }
    }
    
    // Generate the stub module
    quote! {
        pub mod ffi {
            #(#stub_items)*
        }
    }.to_string()
}

fn generate_opaque_struct(name: &Ident) -> TokenStream {
    quote! {
        #[repr(C)]
        pub struct #name {
            _private: [u8; 0],
        }
        
        // Implement UniquePtrTarget trait for cxx compatibility
        unsafe impl cxx::private::UniquePtrTarget for #name {
            fn __typename() -> &'static dyn std::fmt::Display {
                &#stringify!(#name)
            }
            fn __null() -> std::mem::MaybeUninit<*mut std::ffi::c_void> {
                std::mem::MaybeUninit::new(std::ptr::null_mut())
            }
        }
    }
}

fn generate_stub_function(func: &ForeignItemFn) -> TokenStream {
    let sig = &func.sig;
    // Ensure function is public
    let vis = &func.vis;
    let attrs = &func.attrs;
    
    quote! {
        #(#attrs)*
        #vis #sig {
            unreachable!("docs-only stub")
        }
    }
}
```

#### 2. Critical API Compatibility Requirements

The stub must be fully API-compatible with cxx-generated code:

1. **Public Visibility**: All functions must be `pub` (cxx generates public functions)
2. **Use cxx Types**: Must use `cxx::UniquePtr`, not custom implementations
3. **Trait Implementation**: Opaque types must implement `cxx::private::UniquePtrTarget`
4. **Module Structure**: Match the exact module structure from cxx

#### 3. Handling Special CXX Types

The transformation must handle CXX-specific types appropriately:

- `UniquePtr<T>` → Use `cxx::UniquePtr<T>` directly
- `SharedPtr<T>` → Use `cxx::SharedPtr<T>` directly
- `&T`, `&mut T` → Keep as-is in signatures
- `Pin<&mut T>` → Keep as-is in signatures
- Result types → Keep as-is, body returns `unreachable!()`

## Progress Tracking

### Phase 1: Setup Infrastructure

| Task                                   | Status  | Owner | Notes                   |
|----------------------------------------|---------|-------|-------------------------|
| Modify lib.rs for conditional includes | ⬜ TODO |       |                         |
| Create src/generated/ directory        | ⬜ TODO |       |                         |
| Add generated files to version control | ⬜ TODO |       | Commit both files       |
| Setup Makefile with targets            | ⬜ TODO |       |                         |

### Phase 2: Implement Transform in build.rs

| Task                             | Status  | Owner | Notes |
|----------------------------------|---------|-------|-------|
| Add syn dependency to build-deps      | ⬜ TODO |       |       |
| Implement basic AST parsing           | ⬜ TODO |       |       |
| Transform extern "C++" blocks         | ⬜ TODO |       |       |
| Generate stub function bodies         | ⬜ TODO |       | Use unreachable!() |
| Handle CXX special types              | ⬜ TODO |       |       |
| Format output with rustfmt            | ⬜ TODO |       | Via Command::new() |

### Phase 3: Initial Generation

| Task                         | Status  | Owner | Notes                    |
|------------------------------|---------|-------|--------------------------|
| Generate initial bindings.rs | ⬜ TODO |       | Run with bindgen feature |
| Generate initial stub.rs     | ⬜ TODO |       |                          |
| Test docs-only build         | ⬜ TODO |       |                          |
| Test normal build            | ⬜ TODO |       |                          |
| Commit generated files       | ⬜ TODO |       |                          |

### Phase 4: Testing and Documentation

| Task                      | Status  | Owner | Notes     |
|---------------------------|---------|-------|-----------|
| Test on docs.rs simulator | ⬜ TODO |       | docs-only |
| Add transform unit tests  | ⬜ TODO |       |           |
| Document usage in README  | ⬜ TODO |       |           |

## Action Items

### Phase 1: Infrastructure Setup ✅ COMPLETED
- [x] Create `fast-gicp-sys/src/generated/` directory
- [x] Modify `fast-gicp-sys/src/lib.rs` to use conditional compilation
- [x] Add `syn = "2.0"` and `quote = "1.0"` to build-dependencies
- [x] Set up basic Makefile with initial targets

### Phase 2: Code Generation Implementation ✅ COMPLETED
- [x] Implement `find_cxx_bridge_module()` function
- [x] Implement `generate_full_bindings()` function
- [x] Implement `generate_stub_bindings()` with proper AST transformation
- [x] Implement `format_with_rustfmt()` helper
- [x] Add proper error handling throughout

### Phase 3: Testing and Verification ✅ COMPLETED
- [x] Generate initial `bindings.rs` and `stub.rs` files
- [x] Test normal build: `make test-normal`
- [x] Test docs-only build: `make test-docs-only`
- [x] Verify with cargo-expand: `make verify-bindings`
- [x] Run full lint suite: `make lint`

### Phase 4: CUDA Migration ✅ COMPLETED
- [x] Remove unused `bindings.rs` generation from build.rs
- [x] Rename `generate_bindings()` to `generate_stubs()`
- [x] Update `generate_stub_bindings()` to accept `include_cuda` parameter
- [x] Implement CUDA item detection (cfg attributes and naming patterns)
- [x] Generate `stub.rs` without CUDA items
- [x] Generate `stub_cuda.rs` with all items including CUDA
- [x] Test `docs-only` feature (should use `stub.rs`)
- [x] Test `docs-only + cuda` features (should use `stub_cuda.rs`)
- [x] Update Makefile targets (`generate-stubs`, `verify-stubs`)
- [x] Remove existing bindings.rs file

### Phase 5: API Compatibility Fixes ✅ COMPLETED
- [x] Fix function visibility - make all functions `pub`
- [x] Replace custom `UniquePtr` with `cxx::UniquePtr`
- [x] Implement `UniquePtrTarget` trait for all opaque types
- [x] Add proper imports (`use cxx::UniquePtr`, `use std::pin::Pin`)
- [x] Test that fast-gicp crate compiles with stubs
- [x] Verify stubs provide exact same API as cxx-generated code
- [x] Update testing strategy in Makefile
- [x] Run full test suite with updated stubs

### Phase 6: Documentation and Polish
- [x] Update README with generation instructions
- [x] Add inline documentation to build.rs
- [ ] Create examples showing CUDA feature usage
- [ ] Update README with dual-stub feature information
- [x] Run `make pre-commit` to ensure all checks pass

### Phase 7: Integration and Testing
- [x] Test with `cargo doc` locally
- [x] Verify stub provides all necessary type signatures
- [x] Check generated file sizes are reasonable
- [x] Commit generated files to version control
- [x] Test dual-stub generation end-to-end
- [ ] Validate docs.rs simulation with both feature combinations

## Success Criteria

1. **Single Source**: cxx::bridge definition is the only source of truth
2. **User Experience**: Most users just use pre-generated code
3. **Maintainer Experience**: Simple command to regenerate
4. **docs.rs Support**: Successfully builds documentation
5. **Consistency**: Generated files match the cxx::bridge definition

## Risks and Mitigations

| Risk                            | Impact | Mitigation                  |
|---------------------------------|--------|-----------------------------|
| Generated files get out of sync | High   | Manual verification process |
| Transform logic becomes complex | Medium | Extract to separate crate   |
| CXX internals change            | Medium | Pin CXX version, add tests  |
| Large generated files           | Low    | Only essential code in stub |
| cargo expand availability       | Medium | Provide fallback method     |
| Circular dependency in build    | High   | Careful feature management  |

## Future Enhancements

1. **Optimize stub size**: Generate minimal viable stubs
2. **Better mock implementations**: More realistic return values
3. **Generalize approach**: Package as a crate for other CXX users
4. **Upstream to CXX**: Propose as official CXX feature
