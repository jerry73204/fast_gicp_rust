//! Fast GICP (Generalized Iterative Closest Point) registration.

mod builder;

pub use builder::FastGICPBuilder;

use crate::{types::RegularizationMethod, PointCloudXYZ, Result, Transform3f};
use fast_gicp_sys::ffi;

/// Result of a registration operation.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Final transformation from source to target.
    pub final_transformation: Transform3f,
    /// Fitness score (lower is better).
    pub fitness_score: f64,
    /// Whether the algorithm converged.
    pub has_converged: bool,
    /// Number of iterations performed.
    pub num_iterations: i32,
}

/// Configuration for FastGICP algorithm.
#[derive(Debug, Clone)]
pub struct FastGICPConfig {
    pub max_iterations: u32,
    pub transformation_epsilon: f64,
    pub euclidean_fitness_epsilon: f64,
    pub max_correspondence_distance: f64,
    pub num_threads: i32,
    pub correspondence_randomness: u32,
    pub regularization_method: RegularizationMethod,
    pub rotation_epsilon: f64,
}

impl Default for FastGICPConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            transformation_epsilon: 0.01,
            euclidean_fitness_epsilon: 0.01,
            max_correspondence_distance: 1.0,
            num_threads: 0,
            correspondence_randomness: 20,
            regularization_method: RegularizationMethod::None,
            rotation_epsilon: 2e-3,
        }
    }
}

/// Fast GICP (Generalized Iterative Closest Point) registration.
pub struct FastGICP {
    config: FastGICPConfig,
}

impl FastGICP {
    /// Creates a new FastGICP instance with default configuration.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// use fast_gicp::FastGICP;
    ///
    /// let gicp = FastGICP::new();
    /// ```
    pub fn new() -> Self {
        Self {
            config: FastGICPConfig::default(),
        }
    }

    /// Creates a new FastGICP instance with custom configuration.
    pub(crate) fn with_config(config: FastGICPConfig) -> Self {
        Self { config }
    }

    /// Creates a new builder for constructing a FastGICP instance.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// use fast_gicp::FastGICP;
    ///
    /// let gicp = FastGICP::builder()
    ///     .max_iterations(50)
    ///     .transformation_epsilon(1e-6)
    ///     .build();
    /// ```
    pub fn builder() -> FastGICPBuilder {
        FastGICPBuilder::new()
    }

    /// Performs registration on the given source and target point clouds.
    ///
    /// # Examples
    #[cfg_attr(feature = "docs-only", doc = "```no_run")]
    #[cfg_attr(not(feature = "docs-only"), doc = "```")]
    /// # use fast_gicp::{FastGICP, PointCloudXYZ, Result};
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
    /// let gicp = FastGICP::new();
    /// let result = gicp.align(&source, &target)?;
    ///
    /// println!("Converged: {}", result.has_converged);
    /// println!("Fitness score: {}", result.fitness_score);
    /// # Ok(())
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
        // Validate point clouds
        crate::registration::validation::validate_point_cloud_xyz(source, "source")?;
        crate::registration::validation::validate_point_cloud_xyz(target, "target")?;

        // Create FFI instance and configure
        let mut inner = ffi::create_fast_gicp();
        ffi::fast_gicp_set_input_source(inner.pin_mut(), source.as_ffi());
        ffi::fast_gicp_set_input_target(inner.pin_mut(), target.as_ffi());
        ffi::fast_gicp_set_max_iterations(inner.pin_mut(), self.config.max_iterations as i32);
        ffi::fast_gicp_set_transformation_epsilon(
            inner.pin_mut(),
            self.config.transformation_epsilon,
        );
        ffi::fast_gicp_set_euclidean_fitness_epsilon(
            inner.pin_mut(),
            self.config.euclidean_fitness_epsilon,
        );
        ffi::fast_gicp_set_max_correspondence_distance(
            inner.pin_mut(),
            self.config.max_correspondence_distance,
        );
        ffi::fast_gicp_set_num_threads(inner.pin_mut(), self.config.num_threads);
        ffi::fast_gicp_set_correspondence_randomness(
            inner.pin_mut(),
            self.config.correspondence_randomness as i32,
        );
        ffi::fast_gicp_set_regularization_method(
            inner.pin_mut(),
            self.config.regularization_method as i32,
        );
        ffi::fast_gicp_set_rotation_epsilon(inner.pin_mut(), self.config.rotation_epsilon);

        // Perform alignment
        let final_transformation = if let Some(guess) = initial_guess {
            let guess_ffi = guess.as_transform4f();
            ffi::fast_gicp_align_with_guess(inner.pin_mut(), &guess_ffi)
        } else {
            ffi::fast_gicp_align(inner.pin_mut())
        };

        let fitness_score = ffi::fast_gicp_get_fitness_score(&inner);
        let has_converged = ffi::fast_gicp_has_converged(&inner);
        let num_iterations = ffi::fast_gicp_get_final_num_iterations(&inner);

        Ok(RegistrationResult {
            final_transformation: Transform3f::from_transform4f(&final_transformation),
            fitness_score,
            has_converged,
            num_iterations,
        })
    }
}

impl Default for FastGICP {
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
    fn test_fast_gicp_align() {
        let gicp = FastGICP::new();
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

        let result = gicp.align(&source, &target).unwrap();
        // Check that we got a result (might not converge with default params)
        assert!(result.num_iterations > 0);
    }

    #[test]
    fn test_fast_gicp_empty_cloud_error() {
        let gicp = FastGICP::new();
        let empty_cloud = PointCloudXYZ::new();
        let valid_cloud = create_test_cloud();

        // Test empty source
        let result = gicp.align(&empty_cloud, &valid_cloud);
        assert!(result.is_err());

        // Test empty target
        let result = gicp.align(&valid_cloud, &empty_cloud);
        assert!(result.is_err());
    }

    #[test]
    fn test_fast_gicp_reusable() {
        let gicp = FastGICP::new();
        let source1 = create_test_cloud();
        let target1 = create_test_cloud();

        // First alignment
        let result1 = gicp.align(&source1, &target1).unwrap();
        assert!(result1.has_converged);

        // Reuse for second alignment
        let source2 = create_test_cloud();
        let target2 = create_test_cloud();
        let result2 = gicp.align(&source2, &target2).unwrap();
        assert!(result2.has_converged);
    }
}
