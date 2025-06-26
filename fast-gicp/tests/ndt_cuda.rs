//! Tests for NDT CUDA functionality.

#[cfg(feature = "cuda")]
mod tests {
    use fast_gicp::{NDTCuda, NdtDistanceMode, NeighborSearchMethod, PointCloudXYZ, Transform3f};

    #[test]
    fn test_ndt_cuda_creation() {
        let ndt = NDTCuda::builder().build().unwrap();
        // If we get here, the NDTCuda was created successfully
        let _ = ndt; // Use the variable to avoid warnings
    }

    #[test]
    fn test_ndt_cuda_configuration() {
        // Test basic configuration methods with builder pattern
        let ndt = NDTCuda::builder()
            .max_iterations(100)
            .transformation_epsilon(1e-6)
            .euclidean_fitness_epsilon(1e-4)
            .max_correspondence_distance(1.0)
            .resolution(0.5)
            .distance_mode(NdtDistanceMode::P2D)
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build()
            .unwrap();

        // Test changing distance mode after creation
        let ndt2 = NDTCuda::builder()
            .distance_mode(NdtDistanceMode::D2D)
            .build()
            .unwrap();

        let _ = ndt; // Use the variables to avoid warnings
        let _ = ndt2;
    }

    #[test]
    fn test_ndt_cuda_registration() {
        // Create simple test point clouds
        let source = PointCloudXYZ::from_points(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);

        let target = PointCloudXYZ::from_points(&[
            [0.1, 0.1, 0.0],
            [1.1, 0.1, 0.0],
            [0.1, 1.1, 0.0],
            [1.1, 1.1, 0.0],
        ]);

        // Create NDT with configuration using builder pattern
        let ndt = NDTCuda::builder()
            .max_iterations(50)
            .resolution(0.5)
            .distance_mode(NdtDistanceMode::P2D)
            .build()
            .unwrap();

        // Perform registration with new API
        let result = ndt
            .align(&source, &target)
            .expect("Failed to perform registration");

        // Check that we got reasonable results
        assert!(result.num_iterations > 0);
        assert!(result.num_iterations <= 50);

        // The fitness score should be finite and non-negative
        assert!(result.fitness_score.is_finite());
        assert!(result.fitness_score >= 0.0);

        // Transform should be reasonable (not all zeros)
        let transform_matrix = result.final_transformation.to_matrix();
        let has_non_zero = transform_matrix.iter().any(|&x| x != 0.0);
        assert!(has_non_zero, "Transform matrix should not be all zeros");
    }

    #[test]
    fn test_ndt_cuda_with_initial_guess() {
        // Create test point clouds
        let source =
            PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

        let target =
            PointCloudXYZ::from_points(&[[0.2, 0.2, 0.0], [1.2, 0.2, 0.0], [0.2, 1.2, 0.0]]);

        // Create NDT with configuration using builder pattern
        let ndt = NDTCuda::builder()
            .max_iterations(30)
            .resolution(0.3)
            .build()
            .unwrap();

        // Create an initial guess transformation
        let initial_guess = Transform3f::from_translation(0.1, 0.1, 0.0);

        // Perform registration with initial guess using new API
        let result = ndt
            .align_with_guess(&source, &target, Some(&initial_guess))
            .expect("Failed to perform registration with guess");

        // Check results
        assert!(result.num_iterations <= 30);
        assert!(result.fitness_score.is_finite());

        // The result should be close to the expected translation
        let final_translation = result.final_transformation.translation();
        assert!(
            (final_translation[0] - 0.2).abs() < 0.5,
            "X translation should be close to 0.2"
        );
        assert!(
            (final_translation[1] - 0.2).abs() < 0.5,
            "Y translation should be close to 0.2"
        );
        assert!(
            final_translation[2].abs() < 0.1,
            "Z translation should be close to 0.0"
        );
    }

    #[test]
    fn test_ndt_cuda_parameter_validation() {
        // Test invalid parameters should fail during build
        // Test with invalid iterations (0 is invalid, must be >= 1)
        assert!(NDTCuda::builder().max_iterations(0).build().is_err());
        assert!(NDTCuda::builder()
            .transformation_epsilon(-1.0)
            .build()
            .is_err());
        assert!(NDTCuda::builder()
            .euclidean_fitness_epsilon(-1.0)
            .build()
            .is_err());
        assert!(NDTCuda::builder()
            .max_correspondence_distance(-1.0)
            .build()
            .is_err());
        assert!(NDTCuda::builder().resolution(-1.0).build().is_err());
        // neighbor_search_method is valid by itself, so no test needed here

        // Test valid parameters should succeed
        assert!(NDTCuda::builder().max_iterations(100).build().is_ok());
        assert!(NDTCuda::builder()
            .transformation_epsilon(0.0)
            .build()
            .is_ok());
        assert!(NDTCuda::builder()
            .euclidean_fitness_epsilon(0.0)
            .build()
            .is_ok());
        assert!(NDTCuda::builder()
            .max_correspondence_distance(1.0)
            .build()
            .is_ok());
        assert!(NDTCuda::builder().resolution(0.1).build().is_ok());
        assert!(NDTCuda::builder()
            .neighbor_search_method(NeighborSearchMethod::Direct7)
            .build()
            .is_ok());
    }

    #[test]
    fn test_ndt_cuda_empty_cloud_error() {
        let ndt = NDTCuda::builder().build().unwrap();
        let empty_cloud = PointCloudXYZ::new();

        // Should fail with empty clouds during alignment
        assert!(ndt.align(&empty_cloud, &empty_cloud).is_err());
    }

    #[test]
    fn test_ndt_cuda_distance_modes() {
        let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);

        let target = PointCloudXYZ::from_points(&[[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]]);

        // Test P2D mode
        let ndt_p2d = NDTCuda::builder()
            .max_iterations(10)
            .resolution(0.5)
            .distance_mode(NdtDistanceMode::P2D)
            .build()
            .unwrap();

        let result_p2d = ndt_p2d
            .align(&source, &target)
            .expect("Failed to align with P2D mode");
        assert!(result_p2d.fitness_score.is_finite());

        // Test D2D mode
        let ndt_d2d = NDTCuda::builder()
            .max_iterations(10)
            .resolution(0.5)
            .distance_mode(NdtDistanceMode::D2D)
            .build()
            .unwrap();

        let result_d2d = ndt_d2d
            .align(&source, &target)
            .expect("Failed to align with D2D mode");
        assert!(result_d2d.fitness_score.is_finite());

        // Both modes should produce valid results
        assert!(result_p2d.num_iterations > 0);
        assert!(result_d2d.num_iterations > 0);
    }
}
