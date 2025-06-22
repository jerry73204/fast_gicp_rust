//! High-level Rust API for the fast_gicp point cloud registration library.
//!
//! This crate provides safe, idiomatic Rust bindings for the fast_gicp C++ library,
//! which implements efficient variants of the Generalized Iterative Closest Point (GICP) algorithm.

use cxx::UniquePtr;
use std::pin::Pin;
use thiserror::Error;

pub use fast_gicp_sys::ffi::{
    LSQNonlinearOptimizationAlgorithm, NeighborSearchMethod, RegularizationMethod,
};

/// Point cloud and registration modules
pub mod point_cloud;
pub mod registration;
pub mod transform;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use point_cloud::{PointCloudXYZ, PointCloudXYZI};
pub use registration::{FastGICP, FastVGICP, RegistrationResult};
pub use transform::Transform3f;

/// Error types for the fast_gicp library
#[derive(Error, Debug)]
pub enum FastGicpError {
    #[error("Point cloud is empty")]
    EmptyPointCloud,

    #[error("Point cloud index out of bounds: {index}")]
    IndexOutOfBounds { index: usize },

    #[error("Registration failed to converge")]
    RegistrationFailed,

    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    #[error("Internal C++ error: {message}")]
    CppError { message: String },
}

/// Result type for fast_gicp operations
pub type Result<T> = std::result::Result<T, FastGicpError>;

#[cfg(test)]
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
