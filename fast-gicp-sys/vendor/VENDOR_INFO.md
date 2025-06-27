# Vendored fast_gicp Sources

This directory contains vendored C++ sources from the fast_gicp library.

## Source Information
- Repository: https://github.com/SMRT-AIST/fast_gicp
- Commit: 9db8634721559767f90e4f3a20806996fba242aa
- Date: 2025-06-27 05:30:24 UTC
- License: BSD 3-Clause (see fast_gicp/LICENSE)

## Contents
- fast_gicp/include/: C++ headers
- fast_gicp/src/fast_gicp/: C++ implementation files
- fast_gicp/thirdparty/: Bundled dependencies (Eigen, Sophus, nvbio)
- fast_gicp/CMakeLists.txt: CMake build configuration
- fast_gicp/LICENSE: BSD 3-Clause license

## Notes
These sources are used when building from crates.io or when the git submodule
is not available. The build.rs script automatically detects and uses these
vendored sources when present.
