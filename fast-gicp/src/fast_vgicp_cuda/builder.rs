//! Builder for FastVGICPCuda algorithm.

use super::{FastVGICPCuda, FastVGICPCudaConfig};
use crate::{types::NeighborSearchMethod, Result};

/// Builder for constructing a FastVGICPCuda instance with custom parameters.
pub struct FastVGICPCudaBuilder {
    // Optional parameters only
    max_iterations: u32,
    transformation_epsilon: f64,
    euclidean_fitness_epsilon: f64,
    max_correspondence_distance: f64,
    resolution: f64,
    neighbor_search_method: NeighborSearchMethod,
}

impl FastVGICPCudaBuilder {
    /// Creates a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: 1.0,
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

    /// Sets the neighbor search method.
    pub fn neighbor_search_method(mut self, method: NeighborSearchMethod) -> Self {
        self.neighbor_search_method = method;
        self
    }

    /// Builds the FastVGICPCuda instance with the configured parameters.
    pub fn build(self) -> Result<FastVGICPCuda> {
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

        // Create configuration
        let config = FastVGICPCudaConfig {
            max_iterations: self.max_iterations,
            transformation_epsilon: self.transformation_epsilon,
            euclidean_fitness_epsilon: self.euclidean_fitness_epsilon,
            max_correspondence_distance: self.max_correspondence_distance,
            resolution: self.resolution,
            neighbor_search_method: self.neighbor_search_method,
        };

        Ok(FastVGICPCuda::with_config(config))
    }
}

impl Default for FastVGICPCudaBuilder {
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
        let _vgicp = FastVGICPCudaBuilder::new().build().unwrap();
        // Builder creates a valid instance with defaults
    }

    #[test]
    fn test_builder_with_parameters() {
        let _vgicp = FastVGICPCudaBuilder::new()
            .max_iterations(100)
            .transformation_epsilon(1e-8)
            .euclidean_fitness_epsilon(1e-6)
            .max_correspondence_distance(2.0)
            .resolution(0.5)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build()
            .unwrap();
    }

    #[test]
    fn test_builder_validation() {
        // Test invalid max_iterations
        let result = FastVGICPCudaBuilder::new().max_iterations(0).build();
        assert!(result.is_err());

        // Test negative resolution
        let builder = FastVGICPCudaBuilder {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            resolution: -1.0, // Invalid
            neighbor_search_method: NeighborSearchMethod::Direct27,
        };
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_can_be_used_multiple_times() {
        let vgicp = FastVGICPCudaBuilder::new()
            .max_iterations(50)
            .resolution(0.5)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
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
