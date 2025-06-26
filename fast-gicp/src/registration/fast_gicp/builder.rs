//! Builder for FastGICP algorithm.

use super::{FastGICP, FastGICPConfig};
use crate::{types::RegularizationMethod, Result};

/// Builder for constructing a FastGICP instance with custom parameters.
pub struct FastGICPBuilder {
    // Optional parameters only
    max_iterations: u32,
    transformation_epsilon: f64,
    euclidean_fitness_epsilon: f64,
    max_correspondence_distance: f64,
    num_threads: i32,
    correspondence_randomness: u32,
    regularization_method: RegularizationMethod,
    rotation_epsilon: f64,
}

impl FastGICPBuilder {
    /// Creates a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            num_threads: 0, // 0 means use all available threads
            correspondence_randomness: 20,
            regularization_method: RegularizationMethod::None,
            rotation_epsilon: 2e-3,
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

    /// Sets the number of threads to use (0 = all available).
    pub fn num_threads(mut self, threads: i32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Sets the correspondence randomness parameter.
    pub fn correspondence_randomness(mut self, k: u32) -> Self {
        self.correspondence_randomness = k;
        self
    }

    /// Sets the regularization method.
    pub fn regularization_method(mut self, method: RegularizationMethod) -> Self {
        self.regularization_method = method;
        self
    }

    /// Sets the rotation epsilon for convergence.
    pub fn rotation_epsilon(mut self, epsilon: f64) -> Self {
        self.rotation_epsilon = epsilon;
        self
    }

    /// Builds the FastGICP instance with the configured parameters.
    pub fn build(self) -> Result<FastGICP> {
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
            self.num_threads as f64,
            0.0,
            f64::MAX,
            "num_threads",
        )?;
        crate::registration::validation::validate_range(
            self.correspondence_randomness as f64,
            1.0,
            f64::MAX,
            "correspondence_randomness",
        )?;
        crate::registration::validation::validate_range(
            self.rotation_epsilon,
            0.0,
            f64::MAX,
            "rotation_epsilon",
        )?;

        // Create configuration
        let config = FastGICPConfig {
            max_iterations: self.max_iterations,
            transformation_epsilon: self.transformation_epsilon,
            euclidean_fitness_epsilon: self.euclidean_fitness_epsilon,
            max_correspondence_distance: self.max_correspondence_distance,
            num_threads: self.num_threads,
            correspondence_randomness: self.correspondence_randomness,
            regularization_method: self.regularization_method,
            rotation_epsilon: self.rotation_epsilon,
        };

        Ok(FastGICP::with_config(config))
    }
}

impl Default for FastGICPBuilder {
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
        let _gicp = FastGICPBuilder::new().build();
        // Builder creates a valid instance with defaults
    }

    #[test]
    fn test_builder_with_parameters() {
        let _gicp = FastGICPBuilder::new()
            .max_iterations(100)
            .transformation_epsilon(1e-8)
            .euclidean_fitness_epsilon(1e-6)
            .max_correspondence_distance(2.0)
            .num_threads(4)
            .correspondence_randomness(10)
            .regularization_method(RegularizationMethod::Frobenius)
            .rotation_epsilon(1e-4)
            .build();
    }

    #[test]
    fn test_builder_validation() {
        // Test invalid max_iterations
        let result = FastGICPBuilder::new().max_iterations(0).build();
        assert!(result.is_err());

        // Test negative correspondence distance
        let builder = FastGICPBuilder {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: -1.0, // Invalid
            num_threads: 0,
            correspondence_randomness: 20,
            regularization_method: RegularizationMethod::None,
            rotation_epsilon: 2e-3,
        };
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_can_be_used_multiple_times() {
        let gicp = FastGICPBuilder::new().max_iterations(50).build().unwrap();

        // Create test clouds
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cloud: PointCloudXYZ = points.into_iter().collect();

        // Use with first pair
        let _result1 = gicp.align(&cloud, &cloud).unwrap();

        // Reuse with another pair (same clouds for simplicity)
        let _result2 = gicp.align(&cloud, &cloud).unwrap();
    }
}
