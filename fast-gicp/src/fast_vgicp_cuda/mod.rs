//! FastVGICP with CUDA acceleration.

mod builder;

pub use builder::FastVGICPCudaBuilder;

use crate::{types::NeighborSearchMethod, PointCloudXYZ, RegistrationResult, Result, Transform3f};
use fast_gicp_sys::ffi;

/// Configuration for FastVGICPCuda algorithm.
#[derive(Debug, Clone)]
pub struct FastVGICPCudaConfig {
    pub max_iterations: u32,
    pub transformation_epsilon: f64,
    pub euclidean_fitness_epsilon: f64,
    pub max_correspondence_distance: f64,
    pub resolution: f64,
    pub neighbor_search_method: NeighborSearchMethod,
}

impl Default for FastVGICPCudaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: 1.0,
            neighbor_search_method: NeighborSearchMethod::Direct27,
        }
    }
}

/// Fast Voxelized GICP with CUDA acceleration.
pub struct FastVGICPCuda {
    config: FastVGICPCudaConfig,
}

impl FastVGICPCuda {
    /// Creates a new FastVGICPCuda instance with default configuration.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// use fast_gicp::FastVGICPCuda;
    ///
    /// let cuda_vgicp = FastVGICPCuda::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Self {
            config: FastVGICPCudaConfig::default(),
        }
    }

    /// Creates a new FastVGICPCuda instance with custom configuration.
    pub(crate) fn with_config(config: FastVGICPCudaConfig) -> Self {
        Self { config }
    }

    /// Creates a new builder for constructing a FastVGICPCuda instance.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// use fast_gicp::{FastVGICPCuda, types::NeighborSearchMethod};
    ///
    /// let cuda_vgicp = FastVGICPCuda::builder()
    ///     .resolution(1.0)
    ///     .neighbor_search_method(NeighborSearchMethod::Direct27)
    ///     .build();
    /// # }
    /// ```
    pub fn builder() -> FastVGICPCudaBuilder {
        FastVGICPCudaBuilder::new()
    }

    /// Performs registration on the given source and target point clouds.
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
        // Validate point clouds
        crate::registration::validation::validate_point_cloud_xyz(source, "source")?;
        crate::registration::validation::validate_point_cloud_xyz(target, "target")?;

        // Create FFI instance and configure
        let mut inner = ffi::create_fast_vgicp_cuda();
        ffi::fast_vgicp_cuda_set_input_source(inner.pin_mut(), source.as_ffi());
        ffi::fast_vgicp_cuda_set_input_target(inner.pin_mut(), target.as_ffi());
        ffi::fast_vgicp_cuda_set_max_iterations(inner.pin_mut(), self.config.max_iterations as i32);
        ffi::fast_vgicp_cuda_set_transformation_epsilon(
            inner.pin_mut(),
            self.config.transformation_epsilon,
        );
        ffi::fast_vgicp_cuda_set_euclidean_fitness_epsilon(
            inner.pin_mut(),
            self.config.euclidean_fitness_epsilon,
        );
        ffi::fast_vgicp_cuda_set_max_correspondence_distance(
            inner.pin_mut(),
            self.config.max_correspondence_distance,
        );
        ffi::fast_vgicp_cuda_set_resolution(inner.pin_mut(), self.config.resolution);
        ffi::fast_vgicp_cuda_set_neighbor_search_method(
            inner.pin_mut(),
            self.config.neighbor_search_method as i32,
        );

        // Perform alignment
        let final_transformation = if let Some(guess) = initial_guess {
            let guess_ffi = guess.as_transform4f();
            ffi::fast_vgicp_cuda_align_with_guess(inner.pin_mut(), &guess_ffi)
        } else {
            ffi::fast_vgicp_cuda_align(inner.pin_mut())
        };

        let fitness_score = ffi::fast_vgicp_cuda_get_fitness_score(&inner);
        let has_converged = ffi::fast_vgicp_cuda_has_converged(&inner);
        let num_iterations = ffi::fast_vgicp_cuda_get_final_num_iterations(&inner);

        Ok(RegistrationResult {
            final_transformation: Transform3f::from_transform4f(&final_transformation),
            fitness_score,
            has_converged,
            num_iterations,
        })
    }
}

impl Default for FastVGICPCuda {
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
    fn test_fast_vgicp_cuda_creation() {
        let _vgicp = FastVGICPCuda::new();
    }

    #[test]
    fn test_fast_vgicp_cuda_align() {
        let vgicp = FastVGICPCuda::new();
        let source = create_test_cloud();
        let target = create_test_cloud();

        let result = vgicp.align(&source, &target).unwrap();
        assert!(result.has_converged);
    }

    #[test]
    fn test_fast_vgicp_cuda_with_builder() {
        let vgicp = FastVGICPCuda::builder()
            .resolution(0.5)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build()
            .unwrap();
        let source = create_test_cloud();
        let target = create_test_cloud();

        let result = vgicp.align(&source, &target).unwrap();
        assert!(result.has_converged);
    }
}
