//! Builder for FastVGICP algorithm.

use super::{FastVGICP, FastVGICPConfig};
use crate::{
    types::{NeighborSearchMethod, RegularizationMethod, VoxelAccumulationMode},
    Result,
};

/// Builder for constructing a FastVGICP instance with custom parameters.
pub struct FastVGICPBuilder {
    // Optional parameters only
    max_iterations: u32,
    transformation_epsilon: f64,
    euclidean_fitness_epsilon: f64,
    max_correspondence_distance: f64,
    resolution: f64,
    num_threads: i32,
    regularization_method: RegularizationMethod,
    voxel_accumulation_mode: VoxelAccumulationMode,
    neighbor_search_method: NeighborSearchMethod,
}

impl FastVGICPBuilder {
    /// Creates a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: 1.0,
            num_threads: 0, // 0 means use all available threads
            regularization_method: RegularizationMethod::None,
            voxel_accumulation_mode: VoxelAccumulationMode::Additive,
            neighbor_search_method: NeighborSearchMethod::Direct27,
        }
    }

    /// Sets the maximum number of iterations.
    pub fn max_iterations(mut self, iterations: u32) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Sets the transformation epsilon for convergence.
    pub fn transformation_epsilon(mut self, epsilon: f64) -> Self {
        self.transformation_epsilon = epsilon;
        self
    }

    /// Sets the Euclidean fitness epsilon for convergence.
    pub fn euclidean_fitness_epsilon(mut self, epsilon: f64) -> Self {
        self.euclidean_fitness_epsilon = epsilon;
        self
    }

    /// Sets the maximum correspondence distance.
    pub fn max_correspondence_distance(mut self, distance: f64) -> Self {
        self.max_correspondence_distance = distance;
        self
    }

    /// Sets the voxel resolution.
    pub fn resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Sets the number of threads to use (0 = all available).
    pub fn num_threads(mut self, threads: i32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Sets the regularization method.
    pub fn regularization_method(mut self, method: RegularizationMethod) -> Self {
        self.regularization_method = method;
        self
    }

    /// Sets the voxel accumulation mode.
    pub fn voxel_accumulation_mode(mut self, mode: VoxelAccumulationMode) -> Self {
        self.voxel_accumulation_mode = mode;
        self
    }

    /// Sets the neighbor search method.
    pub fn neighbor_search_method(mut self, method: NeighborSearchMethod) -> Self {
        self.neighbor_search_method = method;
        self
    }

    /// Builds the FastVGICP instance with the configured parameters.
    pub fn build(self) -> Result<FastVGICP> {
        // Validate parameters
        crate::registration::validation::validate_range(
            self.max_iterations as f64,
            1.0,
            f64::MAX,
            "max_iterations",
        )?;
        crate::registration::validation::validate_range(
            self.transformation_epsilon,
            0.0,
            f64::MAX,
            "transformation_epsilon",
        )?;
        crate::registration::validation::validate_range(
            self.euclidean_fitness_epsilon,
            0.0,
            f64::MAX,
            "euclidean_fitness_epsilon",
        )?;
        crate::registration::validation::validate_range(
            self.max_correspondence_distance,
            0.0,
            f64::MAX,
            "max_correspondence_distance",
        )?;
        crate::registration::validation::validate_range(
            self.resolution,
            0.0,
            f64::MAX,
            "resolution",
        )?;
        crate::registration::validation::validate_range(
            self.num_threads as f64,
            0.0,
            f64::MAX,
            "num_threads",
        )?;

        // Create configuration
        let config = FastVGICPConfig {
            max_iterations: self.max_iterations,
            transformation_epsilon: self.transformation_epsilon,
            euclidean_fitness_epsilon: self.euclidean_fitness_epsilon,
            max_correspondence_distance: self.max_correspondence_distance,
            resolution: self.resolution,
            num_threads: self.num_threads,
            regularization_method: self.regularization_method,
            voxel_accumulation_mode: self.voxel_accumulation_mode,
            neighbor_search_method: self.neighbor_search_method,
        };

        Ok(FastVGICP::with_config(config))
    }
}

impl Default for FastVGICPBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PointCloudXYZ;

    #[test]
    fn test_builder_defaults() {
        let _vgicp = FastVGICPBuilder::new().build();
        // Builder creates a valid instance with defaults
    }

    #[test]
    fn test_builder_with_parameters() {
        let _vgicp = FastVGICPBuilder::new()
            .max_iterations(100)
            .transformation_epsilon(1e-8)
            .euclidean_fitness_epsilon(1e-6)
            .max_correspondence_distance(2.0)
            .resolution(0.5)
            .num_threads(4)
            .regularization_method(RegularizationMethod::Frobenius)
            .voxel_accumulation_mode(VoxelAccumulationMode::Additive)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build();
    }

    #[test]
    fn test_builder_validation() {
        // Test invalid max_iterations
        let result = FastVGICPBuilder::new().max_iterations(0).build();
        assert!(result.is_err());

        // Test negative resolution
        let builder = FastVGICPBuilder {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: -1.0, // Invalid
            num_threads: 0,
            regularization_method: RegularizationMethod::None,
            voxel_accumulation_mode: VoxelAccumulationMode::Additive,
            neighbor_search_method: NeighborSearchMethod::Direct27,
        };
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_can_be_used_multiple_times() {
        let vgicp = FastVGICPBuilder::new()
            .max_iterations(50)
            .resolution(0.5)
            .build()
            .unwrap();

        // Create test clouds
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cloud: PointCloudXYZ = points.into_iter().collect();

        // Use with first pair
        let _result1 = vgicp.align(&cloud, &cloud).unwrap();

        // Reuse with another pair (same clouds for simplicity)
        let _result2 = vgicp.align(&cloud, &cloud).unwrap();
    }
}
