//! Registration algorithms and traits.

use crate::{FastGicpError, PointCloudXYZ, Result, Transform3f};
use cxx::UniquePtr;
use fast_gicp_sys::ffi::{
    self, LSQNonlinearOptimizationAlgorithm, NeighborSearchMethod, RegularizationMethod,
};
use std::pin::Pin;

/// Result of a registration operation.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Final transformation from source to target.
    pub transformation: Transform3f,
    /// Fitness score (lower is better).
    pub fitness_score: f64,
    /// Whether the algorithm converged.
    pub has_converged: bool,
}

/// Fast GICP (Generalized Iterative Closest Point) registration.
pub struct FastGICP {
    inner: UniquePtr<ffi::FastGICP>,
}

impl FastGICP {
    /// Creates a new FastGICP instance.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_fast_gicp(),
        }
    }

    /// Sets the source point cloud.
    pub fn set_input_source(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(FastGicpError::EmptyPointCloud);
        }
        ffi::fast_gicp_set_input_source(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the target point cloud.
    pub fn set_input_target(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(FastGicpError::EmptyPointCloud);
        }
        ffi::fast_gicp_set_input_target(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the maximum number of iterations.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        if max_iterations <= 0 {
            return Err(FastGicpError::InvalidParameter {
                message: "max_iterations must be positive".to_string(),
            });
        }
        ffi::fast_gicp_set_max_iterations(self.inner.pin_mut(), max_iterations);
        Ok(())
    }

    /// Sets the transformation epsilon for convergence.
    pub fn set_transformation_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_gicp_set_transformation_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the Euclidean fitness epsilon for convergence.
    pub fn set_euclidean_fitness_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_gicp_set_euclidean_fitness_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the maximum correspondence distance.
    pub fn set_max_correspondence_distance(&mut self, distance: f64) -> Result<()> {
        if distance <= 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "distance must be positive".to_string(),
            });
        }
        ffi::fast_gicp_set_max_correspondence_distance(self.inner.pin_mut(), distance);
        Ok(())
    }

    /// Sets the LSQ nonlinear optimization algorithm.
    pub fn set_lsq_nonlinear_optimization_algorithm(
        &mut self,
        algorithm: LSQNonlinearOptimizationAlgorithm,
    ) {
        ffi::fast_gicp_set_lsq_nonlinear_optimization_algorithm(self.inner.pin_mut(), algorithm);
    }

    /// Sets the neighbor search method.
    pub fn set_neighbor_search_method(&mut self, method: NeighborSearchMethod) {
        ffi::fast_gicp_set_neighbor_search_method(self.inner.pin_mut(), method);
    }

    /// Sets the neighbor search radius.
    pub fn set_neighbor_search_radius(&mut self, radius: f64) -> Result<()> {
        if radius <= 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "radius must be positive".to_string(),
            });
        }
        ffi::fast_gicp_set_neighbor_search_radius(self.inner.pin_mut(), radius);
        Ok(())
    }

    /// Sets the regularization method.
    pub fn set_regularization_method(&mut self, method: RegularizationMethod) {
        ffi::fast_gicp_set_regularization_method(self.inner.pin_mut(), method);
    }

    /// Sets the number of threads to use.
    pub fn set_num_threads(&mut self, num_threads: i32) -> Result<()> {
        if num_threads <= 0 {
            return Err(FastGicpError::InvalidParameter {
                message: "num_threads must be positive".to_string(),
            });
        }
        ffi::fast_gicp_set_num_threads(self.inner.pin_mut(), num_threads);
        Ok(())
    }

    /// Performs registration with an initial guess.
    pub fn align(&mut self, initial_guess: &Transform3f) -> Result<RegistrationResult> {
        let mut output = PointCloudXYZ::new();
        let success = ffi::fast_gicp_align(
            self.inner.pin_mut(),
            output.inner.pin_mut(),
            initial_guess.as_ffi(),
        );

        if !success {
            return Err(FastGicpError::RegistrationFailed);
        }

        let transformation =
            Transform3f::from_ffi(ffi::fast_gicp_get_final_transformation(&self.inner));
        let fitness_score = ffi::fast_gicp_get_fitness_score(&self.inner);
        let has_converged = ffi::fast_gicp_has_converged(&self.inner);

        Ok(RegistrationResult {
            transformation,
            fitness_score,
            has_converged,
        })
    }

    /// Performs registration with identity as initial guess.
    pub fn align_simple(&mut self) -> Result<RegistrationResult> {
        self.align(&Transform3f::identity())
    }
}

impl Default for FastGICP {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast Voxelized GICP registration.
pub struct FastVGICP {
    inner: UniquePtr<ffi::FastVGICP>,
}

impl FastVGICP {
    /// Creates a new FastVGICP instance.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_fast_vgicp(),
        }
    }

    /// Sets the source point cloud.
    pub fn set_input_source(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(FastGicpError::EmptyPointCloud);
        }
        ffi::fast_vgicp_set_input_source(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the target point cloud.
    pub fn set_input_target(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(FastGicpError::EmptyPointCloud);
        }
        ffi::fast_vgicp_set_input_target(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the maximum number of iterations.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        if max_iterations <= 0 {
            return Err(FastGicpError::InvalidParameter {
                message: "max_iterations must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_set_max_iterations(self.inner.pin_mut(), max_iterations);
        Ok(())
    }

    /// Sets the transformation epsilon for convergence.
    pub fn set_transformation_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_vgicp_set_transformation_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the Euclidean fitness epsilon for convergence.
    pub fn set_euclidean_fitness_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_vgicp_set_euclidean_fitness_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the maximum correspondence distance.
    pub fn set_max_correspondence_distance(&mut self, distance: f64) -> Result<()> {
        if distance <= 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "distance must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_set_max_correspondence_distance(self.inner.pin_mut(), distance);
        Ok(())
    }

    /// Sets the voxel resolution.
    pub fn set_resolution(&mut self, resolution: f64) -> Result<()> {
        if resolution <= 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "resolution must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_set_resolution(self.inner.pin_mut(), resolution);
        Ok(())
    }

    /// Sets the neighbor search method.
    pub fn set_neighbor_search_method(&mut self, method: NeighborSearchMethod) {
        ffi::fast_vgicp_set_neighbor_search_method(self.inner.pin_mut(), method);
    }

    /// Sets the neighbor search radius.
    pub fn set_neighbor_search_radius(&mut self, radius: f64) -> Result<()> {
        if radius <= 0.0 {
            return Err(FastGicpError::InvalidParameter {
                message: "radius must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_set_neighbor_search_radius(self.inner.pin_mut(), radius);
        Ok(())
    }

    /// Sets the regularization method.
    pub fn set_regularization_method(&mut self, method: RegularizationMethod) {
        ffi::fast_vgicp_set_regularization_method(self.inner.pin_mut(), method);
    }

    /// Sets the number of threads to use.
    pub fn set_num_threads(&mut self, num_threads: i32) -> Result<()> {
        if num_threads <= 0 {
            return Err(FastGicpError::InvalidParameter {
                message: "num_threads must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_set_num_threads(self.inner.pin_mut(), num_threads);
        Ok(())
    }

    /// Performs registration with an initial guess.
    pub fn align(&mut self, initial_guess: &Transform3f) -> Result<RegistrationResult> {
        let mut output = PointCloudXYZ::new();
        let success = ffi::fast_vgicp_align(
            self.inner.pin_mut(),
            output.inner.pin_mut(),
            initial_guess.as_ffi(),
        );

        if !success {
            return Err(FastGicpError::RegistrationFailed);
        }

        let transformation =
            Transform3f::from_ffi(ffi::fast_vgicp_get_final_transformation(&self.inner));
        let fitness_score = ffi::fast_vgicp_get_fitness_score(&self.inner);
        let has_converged = ffi::fast_vgicp_has_converged(&self.inner);

        Ok(RegistrationResult {
            transformation,
            fitness_score,
            has_converged,
        })
    }

    /// Performs registration with identity as initial guess.
    pub fn align_simple(&mut self) -> Result<RegistrationResult> {
        self.align(&Transform3f::identity())
    }
}

impl Default for FastVGICP {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cloud() -> PointCloudXYZ {
        let points = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        points.into_iter().collect()
    }

    #[test]
    fn test_fast_gicp_creation() {
        let _gicp = FastGICP::new();
    }

    #[test]
    fn test_fast_gicp_set_input_clouds() {
        let mut gicp = FastGICP::new();
        let cloud = create_test_cloud();

        gicp.set_input_source(&cloud).unwrap();
        gicp.set_input_target(&cloud).unwrap();
    }

    #[test]
    fn test_fast_gicp_empty_cloud_error() {
        let mut gicp = FastGICP::new();
        let empty_cloud = PointCloudXYZ::new();

        let result = gicp.set_input_source(&empty_cloud);
        assert!(result.is_err());
        if let Err(FastGicpError::EmptyPointCloud) = result {
            // Expected error
        } else {
            panic!("Expected EmptyPointCloud error");
        }
    }

    #[test]
    fn test_fast_gicp_parameter_validation() {
        let mut gicp = FastGICP::new();

        // Test negative max_iterations
        assert!(gicp.set_max_iterations(-1).is_err());
        assert!(gicp.set_max_iterations(0).is_err());
        assert!(gicp.set_max_iterations(100).is_ok());

        // Test negative epsilon
        assert!(gicp.set_transformation_epsilon(-1.0).is_err());
        assert!(gicp.set_transformation_epsilon(0.0).is_ok());
        assert!(gicp.set_transformation_epsilon(1e-6).is_ok());
    }

    #[test]
    fn test_fast_vgicp_creation() {
        let _vgicp = FastVGICP::new();
    }

    #[test]
    fn test_fast_vgicp_set_input_clouds() {
        let mut vgicp = FastVGICP::new();
        let cloud = create_test_cloud();

        vgicp.set_input_source(&cloud).unwrap();
        vgicp.set_input_target(&cloud).unwrap();
    }

    #[test]
    fn test_fast_vgicp_resolution() {
        let mut vgicp = FastVGICP::new();

        assert!(vgicp.set_resolution(-1.0).is_err());
        assert!(vgicp.set_resolution(0.0).is_err());
        assert!(vgicp.set_resolution(1.0).is_ok());
    }
}
