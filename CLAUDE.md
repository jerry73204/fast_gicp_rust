# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust implementation of the Fast GICP (Generalized Iterative Closest Point) algorithm for 3D point cloud registration and alignment.

## Development Commands

Since this is a new repository without a Cargo.toml yet, these are the standard Rust commands that will be used once the project is set up:

```bash
# Build the project
cargo build
cargo build --release

# Run tests with nextest
cargo nextest run --no-fail-fast
cargo nextest run --no-fail-fast --nocapture  # Show println! output during tests

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open

# Check code without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy
```

## Project Structure

This is a new repository that needs initial setup. The typical structure for a Rust library implementing Fast GICP would be:

```
fast_gicp_rust/
├── Cargo.toml           # Project manifest
├── src/
│   ├── lib.rs          # Library root
│   ├── gicp.rs         # Core GICP implementation
│   ├── point_cloud.rs  # Point cloud data structures
│   ├── kdtree.rs       # Spatial indexing
│   └── math/           # Mathematical utilities
│       ├── mod.rs
│       └── transform.rs
├── examples/           # Example usage
├── benches/           # Performance benchmarks
└── tests/             # Integration tests
```

## Key Implementation Notes

When implementing Fast GICP in Rust:

1. **Performance**: Use `nalgebra` for linear algebra operations and consider parallelization with `rayon`
2. **Memory efficiency**: Leverage Rust's ownership system for zero-copy operations where possible
3. **Generic programming**: Make the implementation generic over floating-point types (f32/f64)
4. **Error handling**: Use Result types for fallible operations

## Code Style and Best Practices

- Avoid deep nested blocks in Rust like "if a { if b { if c { return Ok(); }  }} Err()". Prefer the flat style like "if !a {return Err();} if !b {return Err();} if !c {return Err();} Ok()"
- Make use of "let Some(x) = value else {};" syntax in Rust to avoid deeply nested blocks.

## Dependencies to Consider

When setting up Cargo.toml, consider these dependencies:
- `nalgebra`: Linear algebra (matrices, vectors, transformations)
- `ndarray`: Multi-dimensional arrays
- `rayon`: Data parallelism
- `kiddo` or similar: KD-tree for nearest neighbor search
- `criterion`: Benchmarking framework (dev dependency)