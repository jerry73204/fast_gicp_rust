# Changelog

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

[0.1.0]: https://github.com/jerry73204/fast_gicp_rust/releases/tag/v0.1.0
