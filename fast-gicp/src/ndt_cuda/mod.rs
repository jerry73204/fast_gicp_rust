//! NDT with CUDA acceleration.

mod builder;

pub use builder::NDTCudaBuilder;

use crate::{
    types::{Hessian6x6, NdtDistanceMode, NeighborSearchMethod},
    PointCloudXYZ, RegistrationResult, Result, Transform3f,
};
use fast_gicp_sys::ffi;

/// Extended result from NDT alignment including Hessian matrix.
/// The Hessian is useful for covariance estimation via Laplace approximation.
#[derive(Debug, Clone)]
pub struct NdtAlignResult {
    /// Standard registration result
    pub result: RegistrationResult,
    /// 6x6 Hessian matrix from the optimization
    pub hessian: Hessian6x6,
}

/// Configuration for NDTCuda algorithm.
#[derive(Debug, Clone)]
pub struct NDTCudaConfig {
    pub max_iterations: u32,
    pub transformation_epsilon: f64,
    pub euclidean_fitness_epsilon: f64,
    pub max_correspondence_distance: f64,
    pub resolution: f64,
    pub distance_mode: NdtDistanceMode,
    pub neighbor_search_method: NeighborSearchMethod,
    pub neighbor_search_radius: f64,
}

impl Default for NDTCudaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: 1.0,
            distance_mode: NdtDistanceMode::D2D,
            neighbor_search_method: NeighborSearchMethod::Direct1,
            neighbor_search_radius: 2.0,
        }
    }
}

/// Normal Distributions Transform with CUDA acceleration.
pub struct NDTCuda {
    config: NDTCudaConfig,
}

impl NDTCuda {
    /// Creates a new NDTCuda instance with default configuration.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// use fast_gicp::NDTCuda;
    ///
    /// let ndt = NDTCuda::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Self {
            config: NDTCudaConfig::default(),
        }
    }

    /// Creates a new NDTCuda instance with custom configuration.
    pub(crate) fn with_config(config: NDTCudaConfig) -> Self {
        Self { config }
    }

    /// Creates a new builder for constructing an NDTCuda instance.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// use fast_gicp::{NDTCuda, types::{NdtDistanceMode, NeighborSearchMethod}};
    ///
    /// let ndt = NDTCuda::builder()
    ///     .resolution(0.5)
    ///     .distance_mode(NdtDistanceMode::D2D)
    ///     .neighbor_search_method(NeighborSearchMethod::Direct1)
    ///     .build();
    /// # }
    /// ```
    pub fn builder() -> NDTCudaBuilder {
        NDTCudaBuilder::new()
    }

    /// Performs registration on the given source and target point clouds.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// # use fast_gicp::{NDTCuda, PointCloudXYZ, Result};
    /// # fn main() -> Result<()> {
    /// let source = PointCloudXYZ::from_points(&[
    ///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
    ///     [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    /// ]);
    /// let target = PointCloudXYZ::from_points(&[
    ///     [0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0],
    ///     [0.1, 0.0, 1.0], [1.1, 1.0, 0.0], [1.1, 0.0, 1.0],
    ///     [0.1, 1.0, 1.0], [1.1, 1.0, 1.0]
    /// ]);
    ///
    /// let ndt = NDTCuda::new();
    /// let result = ndt.align(&source, &target)?;
    ///
    /// println!("Converged: {}", result.has_converged);
    /// println!("Fitness score: {}", result.fitness_score);
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub fn align(
        &self,
        source: &PointCloudXYZ,
        target: &PointCloudXYZ,
    ) -> Result<RegistrationResult> {
        self.align_with_guess(source, target, None)
    }

    /// Performs registration with an initial transformation guess.
    pub fn align_with_guess(
        &self,
        source: &PointCloudXYZ,
        target: &PointCloudXYZ,
        initial_guess: Option<&Transform3f>,
    ) -> Result<RegistrationResult> {
        let full_result = self.align_with_guess_full(source, target, initial_guess)?;
        Ok(full_result.result)
    }

    /// Performs registration and returns extended result including Hessian matrix.
    /// Use this when you need the Hessian for covariance estimation.
    pub fn align_full(
        &self,
        source: &PointCloudXYZ,
        target: &PointCloudXYZ,
    ) -> Result<NdtAlignResult> {
        self.align_with_guess_full(source, target, None)
    }

    /// Performs registration with initial guess and returns extended result including Hessian.
    pub fn align_with_guess_full(
        &self,
        source: &PointCloudXYZ,
        target: &PointCloudXYZ,
        initial_guess: Option<&Transform3f>,
    ) -> Result<NdtAlignResult> {
        // Validate point clouds
        crate::registration::validation::validate_point_cloud_xyz(source, "source")?;
        crate::registration::validation::validate_point_cloud_xyz(target, "target")?;

        // Create FFI instance and configure
        let mut inner = ffi::create_ndt_cuda();
        ffi::ndt_cuda_set_input_source(inner.pin_mut(), source.as_ffi());
        ffi::ndt_cuda_set_input_target(inner.pin_mut(), target.as_ffi());
        ffi::ndt_cuda_set_max_iterations(inner.pin_mut(), self.config.max_iterations as i32);
        ffi::ndt_cuda_set_transformation_epsilon(
            inner.pin_mut(),
            self.config.transformation_epsilon,
        );
        ffi::ndt_cuda_set_euclidean_fitness_epsilon(
            inner.pin_mut(),
            self.config.euclidean_fitness_epsilon,
        );
        ffi::ndt_cuda_set_max_correspondence_distance(
            inner.pin_mut(),
            self.config.max_correspondence_distance,
        );
        ffi::ndt_cuda_set_resolution(inner.pin_mut(), self.config.resolution);
        ffi::ndt_cuda_set_distance_mode(inner.pin_mut(), self.config.distance_mode as i32);
        ffi::ndt_cuda_set_neighbor_search_method(
            inner.pin_mut(),
            self.config.neighbor_search_method as i32,
            self.config.neighbor_search_radius,
        );

        // Perform alignment
        let final_transformation = if let Some(guess) = initial_guess {
            let guess_ffi = guess.as_transform4f();
            ffi::ndt_cuda_align_with_guess(inner.pin_mut(), &guess_ffi)
        } else {
            ffi::ndt_cuda_align(inner.pin_mut())
        };

        let fitness_score = ffi::ndt_cuda_get_fitness_score(&inner);
        let has_converged = ffi::ndt_cuda_has_converged(&inner);
        let num_iterations = ffi::ndt_cuda_get_final_num_iterations(&inner);

        // Get Hessian matrix for covariance estimation
        let hessian_ffi = ffi::ndt_cuda_get_hessian(&inner);
        let hessian = Hessian6x6::from_data(hessian_ffi.data);

        Ok(NdtAlignResult {
            result: RegistrationResult {
                final_transformation: Transform3f::from_transform4f(&final_transformation),
                fitness_score,
                has_converged,
                num_iterations,
            },
            hessian,
        })
    }

    /// Evaluate the NDT cost (fitness score) at a given pose without running optimization.
    /// This is useful for MULTI_NDT_SCORE covariance estimation mode.
    ///
    /// Note: This requires source and target to be set first via align methods.
    pub fn evaluate_cost_at_pose(
        &self,
        source: &PointCloudXYZ,
        target: &PointCloudXYZ,
        pose: &Transform3f,
    ) -> Result<f64> {
        // Validate point clouds
        crate::registration::validation::validate_point_cloud_xyz(source, "source")?;
        crate::registration::validation::validate_point_cloud_xyz(target, "target")?;

        // Create FFI instance and configure
        let mut inner = ffi::create_ndt_cuda();
        ffi::ndt_cuda_set_input_source(inner.pin_mut(), source.as_ffi());
        ffi::ndt_cuda_set_input_target(inner.pin_mut(), target.as_ffi());
        ffi::ndt_cuda_set_resolution(inner.pin_mut(), self.config.resolution);
        ffi::ndt_cuda_set_distance_mode(inner.pin_mut(), self.config.distance_mode as i32);
        ffi::ndt_cuda_set_neighbor_search_method(
            inner.pin_mut(),
            self.config.neighbor_search_method as i32,
            self.config.neighbor_search_radius,
        );

        let pose_ffi = pose.as_transform4f();
        let cost = ffi::ndt_cuda_evaluate_cost(&inner, &pose_ffi);
        Ok(cost)
    }
}

impl Default for NDTCuda {
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
    fn test_ndt_cuda_creation() {
        let _ndt = NDTCuda::new();
    }

    #[test]
    fn test_ndt_cuda_align() {
        let ndt = NDTCuda::new();
        let source = create_test_cloud();
        let target = create_test_cloud();

        let result = ndt.align(&source, &target).unwrap();
        assert!(result.has_converged);
    }

    #[test]
    fn test_ndt_cuda_with_builder() {
        let ndt = NDTCuda::builder()
            .resolution(0.5)
            .distance_mode(NdtDistanceMode::D2D)
            .neighbor_search_method(NeighborSearchMethod::Direct1)
            .neighbor_search_radius(1.0)
            .build()
            .unwrap();
        let source = create_test_cloud();
        let target = create_test_cloud();

        let result = ndt.align(&source, &target).unwrap();
        assert!(result.has_converged);
    }
}
