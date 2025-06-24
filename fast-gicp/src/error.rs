//! Error types for the fast_gicp library

use thiserror::Error;

/// Error types for fast_gicp operations
#[derive(Error, Debug)]
pub enum Error {
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

    #[error("Memory allocation failed")]
    AllocationError,

    #[error("Feature not available: {feature}")]
    FeatureNotAvailable { feature: String },
}

/// Result type for fast_gicp operations
pub type Result<T> = std::result::Result<T, Error>;
