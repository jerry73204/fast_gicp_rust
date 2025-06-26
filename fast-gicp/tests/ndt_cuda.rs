//! Tests for NDT CUDA functionality.

#[cfg(feature = "cuda")]
mod tests {
    use fast_gicp::{NDTCuda, NdtDistanceMode, NeighborSearchMethod, PointCloudXYZ, Transform3f};

    #[test]
    fn test_ndt_cuda_creation() {
        let ndt = NDTCuda::new().expect("Failed to create NDTCuda");
        // If we get here, the NDTCuda was created successfully
        drop(ndt);
    }

    #[test]
    fn test_ndt_cuda_configuration() {
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");

        // Test basic configuration methods
        ndt.set_max_iterations(100)
            .expect("Failed to set max iterations");
        ndt.set_transformation_epsilon(1e-6)
            .expect("Failed to set transformation epsilon");
        ndt.set_euclidean_fitness_epsilon(1e-4)
            .expect("Failed to set euclidean fitness epsilon");
        ndt.set_max_correspondence_distance(1.0)
            .expect("Failed to set max correspondence distance");
        ndt.set_resolution(0.5).expect("Failed to set resolution");

        // Test distance mode setting
        ndt.set_distance_mode(NdtDistanceMode::P2D);
        ndt.set_distance_mode(NdtDistanceMode::D2D);

        // Test neighbor search method
        ndt.set_neighbor_search_method(NeighborSearchMethod::Direct7, -1.0)
            .expect("Failed to set neighbor search method");
    }

    #[test]
    fn test_ndt_cuda_registration() {
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");

        // Create simple test point clouds
        let source = PointCloudXYZ::from_points(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        .expect("Failed to create source cloud");

        let target = PointCloudXYZ::from_points(&[
            [0.1, 0.1, 0.0],
            [1.1, 0.1, 0.0],
            [0.1, 1.1, 0.0],
            [1.1, 1.1, 0.0],
        ])
        .expect("Failed to create target cloud");

        // Set input clouds
        ndt.set_input_source(&source)
            .expect("Failed to set source cloud");
        ndt.set_input_target(&target)
            .expect("Failed to set target cloud");

        // Configure NDT
        ndt.set_max_iterations(50)
            .expect("Failed to set max iterations");
        ndt.set_resolution(0.5).expect("Failed to set resolution");
        ndt.set_distance_mode(NdtDistanceMode::P2D);

        // Perform registration
        let result = ndt.align(None).expect("Failed to perform registration");

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
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");

        // Create test point clouds
        let source =
            PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                .expect("Failed to create source cloud");

        let target =
            PointCloudXYZ::from_points(&[[0.2, 0.2, 0.0], [1.2, 0.2, 0.0], [0.2, 1.2, 0.0]])
                .expect("Failed to create target cloud");

        ndt.set_input_source(&source)
            .expect("Failed to set source cloud");
        ndt.set_input_target(&target)
            .expect("Failed to set target cloud");
        ndt.set_max_iterations(30)
            .expect("Failed to set max iterations");
        ndt.set_resolution(0.3).expect("Failed to set resolution");

        // Create an initial guess transformation
        let initial_guess = Transform3f::from_translation(0.1, 0.1, 0.0);

        // Perform registration with initial guess
        let result = ndt
            .align(Some(&initial_guess))
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
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");

        // Test invalid parameters
        assert!(ndt.set_max_iterations(-1).is_err());
        assert!(ndt.set_max_iterations(0).is_err());
        assert!(ndt.set_transformation_epsilon(-1.0).is_err());
        assert!(ndt.set_euclidean_fitness_epsilon(-1.0).is_err());
        assert!(ndt.set_max_correspondence_distance(-1.0).is_err());
        assert!(ndt.set_max_correspondence_distance(0.0).is_err());
        assert!(ndt.set_resolution(-1.0).is_err());
        assert!(ndt.set_resolution(0.0).is_err());
        assert!(ndt
            .set_neighbor_search_method(NeighborSearchMethod::Direct7, -2.0)
            .is_err());

        // Test valid parameters
        assert!(ndt.set_max_iterations(100).is_ok());
        assert!(ndt.set_transformation_epsilon(0.0).is_ok());
        assert!(ndt.set_euclidean_fitness_epsilon(0.0).is_ok());
        assert!(ndt.set_max_correspondence_distance(1.0).is_ok());
        assert!(ndt.set_resolution(0.1).is_ok());
        assert!(ndt
            .set_neighbor_search_method(NeighborSearchMethod::Direct7, -1.0)
            .is_ok());
    }

    #[test]
    fn test_ndt_cuda_empty_cloud_error() {
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");
        let empty_cloud = PointCloudXYZ::new().expect("Failed to create empty cloud");

        // Should fail with empty clouds
        assert!(ndt.set_input_source(&empty_cloud).is_err());
        assert!(ndt.set_input_target(&empty_cloud).is_err());
    }

    #[test]
    fn test_ndt_cuda_distance_modes() {
        let mut ndt = NDTCuda::new().expect("Failed to create NDTCuda");

        let source = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            .expect("Failed to create source cloud");

        let target = PointCloudXYZ::from_points(&[[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])
            .expect("Failed to create target cloud");

        ndt.set_input_source(&source).expect("Failed to set source");
        ndt.set_input_target(&target).expect("Failed to set target");
        ndt.set_max_iterations(10)
            .expect("Failed to set max iterations");
        ndt.set_resolution(0.5).expect("Failed to set resolution");

        // Test P2D mode
        ndt.set_distance_mode(NdtDistanceMode::P2D);
        let result_p2d = ndt.align(None).expect("Failed to align with P2D mode");
        assert!(result_p2d.fitness_score.is_finite());

        // Test D2D mode
        ndt.set_distance_mode(NdtDistanceMode::D2D);
        let result_d2d = ndt.align(None).expect("Failed to align with D2D mode");
        assert!(result_d2d.fitness_score.is_finite());

        // Both modes should produce valid results
        assert!(result_p2d.num_iterations > 0);
        assert!(result_d2d.num_iterations > 0);
    }
}
