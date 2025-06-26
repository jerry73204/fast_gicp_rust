//! NDT with CUDA acceleration.

mod builder;

pub use builder::NDTCudaBuilder;

use crate::{
    types::{NdtDistanceMode, NeighborSearchMethod},
    PointCloudXYZ, RegistrationResult, Result, Transform3f,
};
use fast_gicp_sys::ffi;

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
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: NDTCudaConfig::default(),
        })
    }

    /// Creates a new NDTCuda instance with custom configuration.
    pub(crate) fn with_config(config: NDTCudaConfig) -> Self {
        Self { config }
    }

    /// Creates a new builder for constructing an NDTCuda instance.
    pub fn builder() -> NDTCudaBuilder {
        NDTCudaBuilder::new()
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

        Ok(RegistrationResult {
            final_transformation: Transform3f::from_transform4f(&final_transformation),
            fitness_score,
            has_converged,
            num_iterations,
        })
    }
}

impl Default for NDTCuda {
    fn default() -> Self {
        Self::new().expect("Failed to create default NDTCuda")
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
        let _ndt = NDTCuda::new().unwrap();
    }

    #[test]
    fn test_ndt_cuda_align() {
        let ndt = NDTCuda::new().unwrap();
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
