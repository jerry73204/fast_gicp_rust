//! High-level Rust API for the fast_gicp point cloud registration library.
//!
//! This crate provides safe, idiomatic Rust bindings for the fast_gicp C++ library,
//! which implements efficient variants of the Generalized Iterative Closest Point (GICP) algorithm.
//!
//! # Features
//!
//! - **openmp**: Enables OpenMP parallelization (default)
//! - **cuda**: Enables CUDA GPU acceleration
//!
//! # Examples
//!
//! ## Using the Builder Pattern
//!
//! The builder pattern provides a fluent API for configuring registration algorithms:
//!
//! ```no_run
//! use fast_gicp::{types::RegularizationMethod, FastGICP, PointCloudXYZ};
//!
//! let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
//! let target = PointCloudXYZ::from_points(&[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]);
//!
//! // Create and configure FastGICP using the builder pattern
//! let gicp = FastGICP::builder()
//!     .max_iterations(50)
//!     .transformation_epsilon(1e-6)
//!     .regularization_method(RegularizationMethod::Frobenius)
//!     .build();
//!
//! let result = gicp.align(&source, &target)?;
//! println!("Final transformation: {:?}", result.final_transformation);
//! # Ok::<(), fast_gicp::Error>(())
//! ```
//!
//! ## VGICP with Voxelization
//!
//! ```no_run
//! use fast_gicp::{types::VoxelAccumulationMode, FastVGICP, PointCloudXYZ};
//!
//! let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
//! let target = PointCloudXYZ::from_points(&[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]);
//!
//! let vgicp = FastVGICP::builder()
//!     .resolution(0.5)
//!     .voxel_accumulation_mode(VoxelAccumulationMode::Additive)
//!     .build();
//!
//! let result = vgicp.align(&source, &target)?;
//! # Ok::<(), fast_gicp::Error>(())
//! ```
//!
//! ## CUDA Acceleration (requires "cuda" feature)
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use fast_gicp::{types::NeighborSearchMethod, FastVGICPCuda, PointCloudXYZ};
//!
//! let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
//! let target = PointCloudXYZ::from_points(&[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]);
//!
//! let cuda_vgicp = FastVGICPCuda::builder()
//!     .resolution(1.0)
//!     .neighbor_search_method(NeighborSearchMethod::Direct27)
//!     .build();
//!
//! let result = cuda_vgicp.align(&source, &target)?;
//! # }
//! # Ok::<(), fast_gicp::Error>(())
//! ```
//!
//! ## Direct API
//!
//! You can also use the algorithms directly:
//!
//! ```no_run
//! use fast_gicp::{FastGICP, PointCloudXYZ, Transform3f};
//!
//! let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
//! let target = PointCloudXYZ::from_points(&[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]);
//!
//! let gicp = FastGICP::new();
//! let result = gicp.align(&source, &target)?;
//! println!("Final transformation: {:?}", result.final_transformation);
//! # Ok::<(), fast_gicp::Error>(())
//! ```

/// Point cloud and registration modules
pub mod error;

#[cfg(not(feature = "docs-only"))]
pub mod point_cloud;
#[cfg(not(feature = "docs-only"))]
pub mod registration;
#[cfg(not(feature = "docs-only"))]
pub mod transform;

pub mod types;

#[cfg(all(feature = "cuda", not(feature = "docs-only")))]
pub mod fast_vgicp_cuda;

#[cfg(all(feature = "cuda", not(feature = "docs-only")))]
pub mod ndt_cuda;

// Re-exports for convenience
pub use error::{Error, Result};

#[cfg(not(feature = "docs-only"))]
pub use point_cloud::{PointCloudXYZ, PointCloudXYZI};
#[cfg(not(feature = "docs-only"))]
pub use registration::{FastGICP, FastVGICP, RegistrationResult};
#[cfg(not(feature = "docs-only"))]
pub use transform::Transform3f;

pub use types::{NeighborSearchMethod, RegularizationMethod, VoxelAccumulationMode};

#[cfg(all(feature = "cuda", not(feature = "docs-only")))]
pub use types::{NdtDistanceMode, NearestNeighborMethod};

#[cfg(all(feature = "cuda", not(feature = "docs-only")))]
pub use fast_vgicp_cuda::FastVGICPCuda;

#[cfg(all(feature = "cuda", not(feature = "docs-only")))]
pub use ndt_cuda::NDTCuda;

#[cfg(all(test, not(feature = "docs-only")))]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let cloud = PointCloudXYZ::new();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());
    }

    #[test]
    fn test_point_cloud_from_iterator() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cloud: PointCloudXYZ = points.into_iter().collect();
        assert_eq!(cloud.size(), 2);
    }
}

#[cfg(feature = "docs-only")]
#[cfg(test)]
mod docs_only_tests {
    use super::types::*;

    #[test]
    fn test_types_exist() {
        // Test that basic types can be created
        let _method = RegularizationMethod::Frobenius;
        let _mode = VoxelAccumulationMode::Additive;
        let _neighbor = NeighborSearchMethod::Direct1;
    }
}
