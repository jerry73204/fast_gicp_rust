//! Tests for CUDA-specific error handling.

#[cfg(feature = "cuda")]
mod tests {
    use fast_gicp::{Error, FastVGICPCuda, NDTCuda, PointCloudXYZ};

    #[test]
    fn test_cuda_vgicp_creation_failure_handling() {
        // This test verifies that if CUDA initialization fails, we get a proper error
        // In a real scenario, this could happen if CUDA drivers are not available
        let result = FastVGICPCuda::builder().build();

        // The creation should either succeed (if CUDA is available) or fail gracefully
        match result {
            Ok(_) => {
                // CUDA is available and working
                println!("CUDA FastVGICP created successfully");
            }
            Err(e) => {
                // CUDA is not available or failed to initialize
                println!("FastVGICPCuda creation failed as expected: {e:?}");
                assert!(matches!(
                    e,
                    Error::CppError { .. } | Error::FeatureNotAvailable { .. }
                ));
            }
        }
    }

    #[test]
    fn test_cuda_ndt_creation_failure_handling() {
        // Similar test for NDTCuda
        let result = NDTCuda::builder().build();

        match result {
            Ok(_) => {
                println!("CUDA NDT created successfully");
            }
            Err(e) => {
                println!("NDTCuda creation failed as expected: {e:?}");
                assert!(matches!(
                    e,
                    Error::CppError { .. } | Error::FeatureNotAvailable { .. }
                ));
            }
        }
    }

    #[test]
    fn test_cuda_registration_with_insufficient_points() {
        // Test behavior when trying to register point clouds with too few points
        if let Ok(ndt) = NDTCuda::builder()
            .max_iterations(10)
            .resolution(0.1)
            .build()
        {
            let single_point = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0]]);

            // Registration might fail or succeed with poor results
            match ndt.align(&single_point, &single_point) {
                Ok(result) => {
                    // If it succeeds, the fitness score should indicate poor registration
                    println!(
                        "Registration with single point succeeded, fitness: {}",
                        result.fitness_score
                    );
                    // With insufficient points, fitness might be 0, NaN, or very high (bad)
                    assert!(
                        result.fitness_score.is_nan() || result.fitness_score >= 0.0,
                        "Fitness score should be finite and non-negative or NaN"
                    );
                }
                Err(e) => {
                    // Registration failure is also acceptable for insufficient data
                    println!("Registration failed as expected with insufficient points: {e:?}");
                }
            }
        }
    }

    #[test]
    fn test_cuda_memory_stress() {
        // Test creating multiple CUDA objects to stress GPU memory
        if NDTCuda::builder().build().is_ok() {
            let mut cuda_objects = Vec::new();

            // Try to create multiple CUDA objects
            for i in 0..5 {
                match NDTCuda::builder().build() {
                    Ok(ndt) => {
                        cuda_objects.push(ndt);
                        println!("Created CUDA object {}", i + 1);
                    }
                    Err(e) => {
                        println!("Failed to create CUDA object {}: {e:?}", i + 1);
                        break;
                    }
                }
            }

            // At least one should have been created successfully
            assert!(
                !cuda_objects.is_empty(),
                "Should be able to create at least one CUDA object"
            );

            // Test that we can still use them
            if let Some(ndt) = cuda_objects.into_iter().next() {
                let cloud = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);

                // Should still be able to align with the object
                assert!(ndt.align(&cloud, &cloud).is_ok());
            }
        }
    }

    #[test]
    fn test_cuda_large_point_cloud_handling() {
        // Test behavior with unusually large point clouds
        if let Ok(ndt) = NDTCuda::builder().resolution(1.0).max_iterations(5).build() {
            // Create a large point cloud (10,000 points)
            let mut points = Vec::new();
            for i in 0..10000 {
                let x = (i as f32 * 0.01) % 100.0;
                let y = ((i / 100) as f32 * 0.01) % 100.0;
                let z = ((i / 10000) as f32 * 0.01) % 10.0;
                points.push([x, y, z]);
            }

            let large_cloud = PointCloudXYZ::from_points(&points);
            println!(
                "Created large point cloud with {} points",
                large_cloud.size()
            );

            // Registration might succeed or fail - both are acceptable
            match ndt.align(&large_cloud, &large_cloud) {
                Ok(result) => {
                    println!(
                        "Large cloud registration succeeded: fitness = {}",
                        result.fitness_score
                    );
                    assert!(result.fitness_score.is_finite());
                }
                Err(e) => {
                    println!("Large cloud registration failed: {e:?}");
                }
            }
        }
    }

    #[test]
    fn test_cuda_invalid_configuration_sequences() {
        // Test various invalid configuration sequences
        if let Ok(ndt) = NDTCuda::builder().max_iterations(5).resolution(1.0).build() {
            // Test that we can detect empty clouds during alignment
            let empty_cloud = PointCloudXYZ::new();
            assert!(
                ndt.align(&empty_cloud, &empty_cloud).is_err(),
                "Should reject empty source and target clouds"
            );

            // Test that valid clouds are accepted
            let cloud = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);

            match ndt.align(&cloud, &cloud) {
                Ok(result) => {
                    println!(
                        "Alignment succeeded with valid configuration: fitness = {}",
                        result.fitness_score
                    );
                    assert!(result.fitness_score.is_finite());
                }
                Err(e) => {
                    println!("Alignment failed even with valid configuration: {e:?}");
                    // This is also acceptable - NDT might not converge with simple test data
                }
            }
        }
    }

    #[test]
    fn test_cuda_extreme_parameter_values() {
        // Test behavior with extreme parameter values

        // Test with extremely small resolution
        match NDTCuda::builder().resolution(1e-10).build() {
            Ok(_) => println!("Extremely small resolution accepted"),
            Err(e) => println!("Extremely small resolution rejected: {e:?}"),
        }

        // Test with extremely large resolution
        match NDTCuda::builder().resolution(1e10).build() {
            Ok(_) => println!("Extremely large resolution accepted"),
            Err(e) => println!("Extremely large resolution rejected: {e:?}"),
        }

        // Test with extremely large max iterations
        match NDTCuda::builder().max_iterations(1000000).build() {
            Ok(_) => println!("Extremely large max iterations accepted"),
            Err(e) => println!("Extremely large max iterations rejected: {e:?}"),
        }

        // Test with extremely small epsilon values
        match NDTCuda::builder().transformation_epsilon(1e-20).build() {
            Ok(_) => println!("Extremely small transformation epsilon accepted"),
            Err(e) => println!("Extremely small transformation epsilon rejected: {e:?}"),
        }
    }
}
