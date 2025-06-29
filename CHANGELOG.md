# Changelog

## [0.3.0] - 2024-12-29

### Added
- Conditional documentation tests that execute in regular builds but are skipped in docs-only mode
- Comprehensive design documentation in `doc/` directory:
  - `ARCH.md`: System architecture and component relationships
  - `DESIGN.md`: Core design decisions and rationale
  - `DESIGN_CODEGEN.md`: Detailed code generation system design
  - `PROGRESS.md`: Implementation progress tracking
  - `DEV.md`: Development workflow and build instructions
- Improved test infrastructure with proper handling of docs-only builds

### Changed
- Documentation tests now use larger point clouds (8 points) to avoid KD-tree search failures
- Makefile test target now skips docs-only tests and provides clear messaging
- Builder pattern examples in documentation now properly handle Result types with `?` operator

### Fixed
- Fixed confusing test failures when running `make test` with docs-only feature enabled
- Doc tests no longer panic with `unreachable!()` when docs-only feature is active
- Improved error handling in doc test examples

### Developer Experience
- Added `check-docs-only` make target for compilation-only verification
- Conditional doc tests use `cfg_attr` to switch between regular and `no_run` modes
- Module-level documentation remains as `no_run` due to cfg_attr limitations

## [0.2.0] - 2024-12-28

### Added
- Pre-generated FFI stub files for docs.rs compatibility
- Automatic stub generation system with CUDA feature filtering
- Documentation build support without C++ dependencies
- Comprehensive stub testing and verification tools

### Changed
- FFI bindings are now conditionally compiled based on docs-only feature
- Improved documentation build process for docs.rs

### Fixed
- Documentation builds on docs.rs without requiring PCL or C++ dependencies
- CUDA-specific APIs are properly hidden when CUDA feature is disabled

## [0.1.0] - 2024-06-27

### Added
- Initial release of fast-gicp Rust wrapper
- Support for FastGICP algorithm with multi-threading via OpenMP
- Support for FastVGICP (Voxelized Generalized ICP)
- CUDA acceleration support for FastVGICPCuda and NDTCuda algorithms
- Builder pattern API for algorithm configuration
- Zero-copy point cloud wrappers for PCL compatibility
- Comprehensive test suite with 64 CPU tests and 92 total with CUDA
- Examples for basic registration and CUDA usage
- Full API documentation

### Features
- **Point Cloud Types**: PointCloudXYZ and PointCloudXYZI with FromIterator support
- **Algorithms**: FastGICP, FastVGICP, FastVGICPCuda (GPU), NDTCuda (GPU)
- **Configuration**: Type-safe enums for all algorithm parameters
- **Error Handling**: Comprehensive error types with thiserror
- **Memory Safety**: All unsafe code isolated in sys crate

### Dependencies
- Requires PCL 1.8+ (Point Cloud Library)
- Requires CMake 3.15+ and C++17 compiler
- Optional: OpenMP for multi-threading (enabled by default)
- Optional: CUDA 11.0+ for GPU acceleration

### Known Limitations
- Linux-only platform support
- Limited to PointXYZ and PointXYZI point types
- No serialization support yet

[0.3.0]: https://github.com/jerry73204/fast_gicp_rust/releases/tag/v0.3.0
[0.2.0]: https://github.com/jerry73204/fast_gicp_rust/releases/tag/v0.2.0
[0.1.0]: https://github.com/jerry73204/fast_gicp_rust/releases/tag/v0.1.0
