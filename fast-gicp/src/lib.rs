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
//! Basic point cloud registration:
//!
//! ```no_run
//! use fast_gicp::{FastGICP, PointCloudXYZ, Transform3f};
//!
//! let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])?;
//! let target = PointCloudXYZ::from_points(&[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])?;
//!
//! let mut gicp = FastGICP::new()?;
//! gicp.set_input_source(&source)?;
//! gicp.set_input_target(&target)?;
//!
//! let result = gicp.align(None)?;
//! println!("Final transformation: {:?}", result.final_transformation);
//! # Ok::<(), fast_gicp::Error>(())
//! ```

/// Point cloud and registration modules
pub mod error;
pub mod point_cloud;
pub mod registration;
pub mod transform;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

// Re-exports for convenience
pub use error::{Error, Result};
pub use point_cloud::{PointCloudXYZ, PointCloudXYZI};
pub use registration::{FastGICP, FastVGICP, RegistrationResult};
pub use transform::Transform3f;
pub use types::{NeighborSearchMethod, RegularizationMethod, VoxelAccumulationMode};

#[cfg(feature = "cuda")]
pub use types::{NdtDistanceMode, NearestNeighborMethod};

#[cfg(feature = "cuda")]
pub use cuda::{FastVGICPCuda, NDTCuda};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_creation() {
        let cloud = PointCloudXYZ::new().expect("Failed to create point cloud");
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
