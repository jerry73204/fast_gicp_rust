//! Validation helpers for builder parameters.
//!
//! This module provides common validation functions used by all builders
//! to ensure parameters are within valid ranges.

use crate::{Error, PointCloudXYZ, PointCloudXYZI};

/// Validates that a value is positive (greater than zero).
#[allow(dead_code)]
pub(crate) fn validate_positive<T>(value: T, name: &str) -> Result<(), Error>
where
    T: PartialOrd + Default + std::fmt::Display,
{
    if value <= T::default() {
        return Err(Error::InvalidParameter {
            message: format!("{name} must be positive, got: {value}"),
        });
    }
    Ok(())
}

/// Validates that a value is non-negative (greater than or equal to zero).
#[allow(dead_code)]
pub(crate) fn validate_non_negative<T>(value: T, name: &str) -> Result<(), Error>
where
    T: PartialOrd + Default + std::fmt::Display,
{
    if value < T::default() {
        return Err(Error::InvalidParameter {
            message: format!("{name} must be non-negative, got: {value}"),
        });
    }
    Ok(())
}

/// Validates that a value is within a specific range.
pub(crate) fn validate_range<T>(value: T, min: T, max: T, name: &str) -> Result<(), Error>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    if value < min || value > max {
        return Err(Error::InvalidParameter {
            message: format!("{name} must be between {min} and {max}, got: {value}"),
        });
    }
    Ok(())
}

/// Validates that a point cloud is not empty.
pub(crate) fn validate_point_cloud_xyz(cloud: &PointCloudXYZ, name: &str) -> Result<(), Error> {
    if cloud.is_empty() {
        return Err(Error::InvalidParameter {
            message: format!("{name} point cloud cannot be empty"),
        });
    }
    Ok(())
}

/// Validates that a point cloud with intensity is not empty.
#[allow(dead_code)]
pub(crate) fn validate_point_cloud_xyzi(cloud: &PointCloudXYZI, name: &str) -> Result<(), Error> {
    if cloud.is_empty() {
        return Err(Error::InvalidParameter {
            message: format!("{name} point cloud cannot be empty"),
        });
    }
    Ok(())
}

/// Validates epsilon values (must be positive and typically small).
#[allow(dead_code)]
pub(crate) fn validate_epsilon(value: f64, name: &str) -> Result<(), Error> {
    if value <= 0.0 {
        return Err(Error::InvalidParameter {
            message: format!("{name} must be positive, got: {value}"),
        });
    }
    if value > 1.0 {
        // Warning: epsilon values are typically much smaller than 1.0
        // but we don't enforce this as a hard limit
    }
    Ok(())
}

/// Validates thread count.
#[allow(dead_code)]
pub(crate) fn validate_threads(num_threads: u32) -> Result<(), Error> {
    if num_threads == 0 {
        return Err(Error::InvalidParameter {
            message: "Number of threads must be at least 1".to_string(),
        });
    }
    Ok(())
}

/// Validates neighbor search radius (can be -1.0 for default or positive).
#[allow(dead_code)]
pub(crate) fn validate_neighbor_radius(radius: f64) -> Result<(), Error> {
    if radius < -1.0 {
        return Err(Error::InvalidParameter {
            message: format!(
                "Neighbor search radius must be -1.0 (default) or positive, got: {radius}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(0.1, "test").is_ok());
        assert!(validate_positive(0.0, "test").is_err());
        assert!(validate_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(1.0, "test").is_ok());
        assert!(validate_non_negative(0.0, "test").is_ok());
        assert!(validate_non_negative(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(validate_range(0.0, 0.0, 1.0, "test").is_ok());
        assert!(validate_range(1.0, 0.0, 1.0, "test").is_ok());
        assert!(validate_range(-0.1, 0.0, 1.0, "test").is_err());
        assert!(validate_range(1.1, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_validate_epsilon() {
        assert!(validate_epsilon(1e-4, "test").is_ok());
        assert!(validate_epsilon(1e-10, "test").is_ok());
        assert!(validate_epsilon(0.0, "test").is_err());
        assert!(validate_epsilon(-1e-4, "test").is_err());
    }

    #[test]
    fn test_validate_threads() {
        assert!(validate_threads(1).is_ok());
        assert!(validate_threads(8).is_ok());
        assert!(validate_threads(0).is_err());
    }

    #[test]
    fn test_validate_neighbor_radius() {
        assert!(validate_neighbor_radius(-1.0).is_ok()); // Default value
        assert!(validate_neighbor_radius(0.0).is_ok());
        assert!(validate_neighbor_radius(1.0).is_ok());
        assert!(validate_neighbor_radius(-2.0).is_err());
    }
}
