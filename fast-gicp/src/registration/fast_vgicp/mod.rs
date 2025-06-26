//! Fast Voxelized GICP registration.

mod builder;

pub use builder::FastVGICPBuilder;

use crate::{
    types::{NeighborSearchMethod, RegularizationMethod, VoxelAccumulationMode},
    PointCloudXYZ, Result, Transform3f,
};
use fast_gicp_sys::ffi;

use super::fast_gicp::RegistrationResult;

/// Configuration for FastVGICP algorithm.
#[derive(Debug, Clone)]
pub struct FastVGICPConfig {
    pub max_iterations: u32,
    pub transformation_epsilon: f64,
    pub euclidean_fitness_epsilon: f64,
    pub max_correspondence_distance: f64,
    pub resolution: f64,
    pub num_threads: i32,
    pub regularization_method: RegularizationMethod,
    pub voxel_accumulation_mode: VoxelAccumulationMode,
    pub neighbor_search_method: NeighborSearchMethod,
}

impl Default for FastVGICPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: 1.0,
            num_threads: 0,
            regularization_method: RegularizationMethod::None,
            voxel_accumulation_mode: VoxelAccumulationMode::Additive,
            neighbor_search_method: NeighborSearchMethod::Direct27,
        }
    }
}

/// Fast Voxelized GICP registration.
pub struct FastVGICP {
    config: FastVGICPConfig,
}

impl FastVGICP {
    /// Creates a new FastVGICP instance with default configuration.
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: FastVGICPConfig::default(),
        })
    }

    /// Creates a new FastVGICP instance with custom configuration.
    pub(crate) fn with_config(config: FastVGICPConfig) -> Self {
        Self { config }
    }

    /// Creates a new builder for constructing a FastVGICP instance.
    pub fn builder() -> FastVGICPBuilder {
        FastVGICPBuilder::new()
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
        let mut inner = ffi::create_fast_vgicp();
        ffi::fast_vgicp_set_input_source(inner.pin_mut(), source.as_ffi());
        ffi::fast_vgicp_set_input_target(inner.pin_mut(), target.as_ffi());
        ffi::fast_vgicp_set_max_iterations(inner.pin_mut(), self.config.max_iterations as i32);
        ffi::fast_vgicp_set_transformation_epsilon(
            inner.pin_mut(),
            self.config.transformation_epsilon,
        );
        ffi::fast_vgicp_set_euclidean_fitness_epsilon(
            inner.pin_mut(),
            self.config.euclidean_fitness_epsilon,
        );
        ffi::fast_vgicp_set_max_correspondence_distance(
            inner.pin_mut(),
            self.config.max_correspondence_distance,
        );
        ffi::fast_vgicp_set_resolution(inner.pin_mut(), self.config.resolution);
        ffi::fast_vgicp_set_num_threads(inner.pin_mut(), self.config.num_threads);
        ffi::fast_vgicp_set_regularization_method(
            inner.pin_mut(),
            self.config.regularization_method as i32,
        );
        ffi::fast_vgicp_set_voxel_accumulation_mode(
            inner.pin_mut(),
            self.config.voxel_accumulation_mode as i32,
        );
        ffi::fast_vgicp_set_neighbor_search_method(
            inner.pin_mut(),
            self.config.neighbor_search_method as i32,
        );

        // Perform alignment
        let final_transformation = if let Some(guess) = initial_guess {
            let guess_ffi = guess.as_transform4f();
            ffi::fast_vgicp_align_with_guess(inner.pin_mut(), &guess_ffi)
        } else {
            ffi::fast_vgicp_align(inner.pin_mut())
        };

        let fitness_score = ffi::fast_vgicp_get_fitness_score(&inner);
        let has_converged = ffi::fast_vgicp_has_converged(&inner);
        let num_iterations = ffi::fast_vgicp_get_final_num_iterations(&inner);

        Ok(RegistrationResult {
            final_transformation: Transform3f::from_transform4f(&final_transformation),
            fitness_score,
            has_converged,
            num_iterations,
        })
    }
}

impl Default for FastVGICP {
    fn default() -> Self {
        Self::new().expect("Failed to create default FastVGICP")
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
    fn test_fast_vgicp_creation() {
        let _vgicp = FastVGICP::new().unwrap();
    }

    #[test]
    fn test_fast_vgicp_align() {
        let vgicp = FastVGICP::new().unwrap();
        let source = create_test_cloud();

        // Create a slightly translated target
        let target_points = vec![
            [1.1, 0.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.1, 0.0, 1.0],
            [1.1, 1.0, 0.0],
            [1.1, 0.0, 1.0],
            [0.1, 1.0, 1.0],
        ];
        let target: PointCloudXYZ = target_points.into_iter().collect();

        let result = vgicp.align(&source, &target).unwrap();
        // Check that we got a result (might not converge with default params)
        assert!(result.num_iterations > 0);
    }

    #[test]
    fn test_fast_vgicp_with_custom_resolution() {
        let vgicp = FastVGICP::builder().resolution(0.5).build().unwrap();
        let source = create_test_cloud();
        let target = create_test_cloud();

        let result = vgicp.align(&source, &target).unwrap();
        assert!(result.has_converged);
    }
}
