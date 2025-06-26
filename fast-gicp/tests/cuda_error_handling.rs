//! Tests for CUDA-specific error handling.

#[cfg(feature = "cuda")]
mod tests {
    use fast_gicp::{Error, FastVGICPCuda, NDTCuda, PointCloudXYZ};

    #[test]
    fn test_cuda_vgicp_creation_failure_handling() {
        // This test verifies that if CUDA initialization fails, we get a proper error
        // In a real scenario, this could happen if CUDA drivers are not available
        let result = FastVGICPCuda::new();

        // The creation should either succeed (if CUDA is available) or fail gracefully
        match result {
            Ok(_) => {
                // CUDA is available and working
                println!("CUDA FastVGICP created successfully");
            }
            Err(e) => {
                // CUDA is not available or failed to initialize
                println!("FastVGICPCuda creation failed as expected: {:?}", e);
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
        let result = NDTCuda::new();

        match result {
            Ok(_) => {
                println!("CUDA NDT created successfully");
            }
            Err(e) => {
                println!("NDTCuda creation failed as expected: {:?}", e);
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
        if let Ok(mut ndt) = NDTCuda::new() {
            let single_point = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0]])
                .expect("Failed to create single point cloud");

            // Set the single point as both source and target
            ndt.set_input_source(&single_point)
                .expect("Failed to set source");
            ndt.set_input_target(&single_point)
                .expect("Failed to set target");

            // Configure with reasonable parameters
            ndt.set_max_iterations(10)
                .expect("Failed to set max iterations");
            ndt.set_resolution(0.1).expect("Failed to set resolution");

            // Registration might fail or succeed with poor results
            match ndt.align(None) {
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
                    println!(
                        "Registration failed as expected with insufficient points: {:?}",
                        e
                    );
                }
            }
        }
    }

    #[test]
    fn test_cuda_memory_stress() {
        // Test creating multiple CUDA objects to stress GPU memory
        if let Ok(_) = NDTCuda::new() {
            let mut cuda_objects = Vec::new();

            // Try to create multiple CUDA objects
            for i in 0..5 {
                match NDTCuda::new() {
                    Ok(ndt) => {
                        cuda_objects.push(ndt);
                        println!("Created CUDA object {}", i + 1);
                    }
                    Err(e) => {
                        println!("Failed to create CUDA object {}: {:?}", i + 1, e);
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
            if let Some(mut ndt) = cuda_objects.into_iter().next() {
                let cloud = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
                    .expect("Failed to create test cloud");

                // Should still be able to configure the object
                assert!(ndt.set_input_source(&cloud).is_ok());
                assert!(ndt.set_input_target(&cloud).is_ok());
            }
        }
    }

    #[test]
    fn test_cuda_large_point_cloud_handling() {
        // Test behavior with unusually large point clouds
        if let Ok(mut ndt) = NDTCuda::new() {
            // Create a large point cloud (10,000 points)
            let mut points = Vec::new();
            for i in 0..10000 {
                let x = (i as f32 * 0.01) % 100.0;
                let y = ((i / 100) as f32 * 0.01) % 100.0;
                let z = ((i / 10000) as f32 * 0.01) % 10.0;
                points.push([x, y, z]);
            }

            match PointCloudXYZ::from_points(&points) {
                Ok(large_cloud) => {
                    println!(
                        "Created large point cloud with {} points",
                        large_cloud.size()
                    );

                    // Test setting input - this might fail due to GPU memory constraints
                    match ndt.set_input_source(&large_cloud) {
                        Ok(_) => {
                            println!("Successfully set large point cloud as source");

                            // Try to set as target too
                            match ndt.set_input_target(&large_cloud) {
                                Ok(_) => {
                                    println!("Successfully set large point cloud as target");

                                    // Configure with appropriate parameters for large clouds
                                    ndt.set_resolution(1.0).expect("Failed to set resolution");
                                    ndt.set_max_iterations(5)
                                        .expect("Failed to set max iterations");

                                    // Registration might succeed or fail - both are acceptable
                                    match ndt.align(None) {
                                        Ok(result) => {
                                            println!(
                                                "Large cloud registration succeeded: fitness = {}",
                                                result.fitness_score
                                            );
                                            assert!(result.fitness_score.is_finite());
                                        }
                                        Err(e) => {
                                            println!("Large cloud registration failed: {:?}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("Failed to set large cloud as target: {:?}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("Failed to set large cloud as source: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("Failed to create large point cloud: {:?}", e);
                }
            }
        }
    }

    #[test]
    fn test_cuda_invalid_configuration_sequences() {
        // Test various invalid configuration sequences
        if let Ok(mut ndt) = NDTCuda::new() {
            // Note: Calling align() without input clouds can cause segfaults in the underlying C++ library
            // This is expected behavior as the C++ library doesn't handle this gracefully
            // We'll test parameter validation instead

            // Test that we can detect empty clouds during input setting
            let empty_cloud = PointCloudXYZ::new().expect("Failed to create empty cloud");
            assert!(
                ndt.set_input_source(&empty_cloud).is_err(),
                "Should reject empty source cloud"
            );
            assert!(
                ndt.set_input_target(&empty_cloud).is_err(),
                "Should reject empty target cloud"
            );

            // Test that valid clouds are accepted
            let cloud = PointCloudXYZ::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
                .expect("Failed to create test cloud");

            assert!(
                ndt.set_input_source(&cloud).is_ok(),
                "Should accept valid source cloud"
            );
            assert!(
                ndt.set_input_target(&cloud).is_ok(),
                "Should accept valid target cloud"
            );

            // Now alignment should work
            ndt.set_max_iterations(5).expect("Failed to set iterations");
            ndt.set_resolution(1.0).expect("Failed to set resolution");

            match ndt.align(None) {
                Ok(result) => {
                    println!(
                        "Alignment succeeded with valid configuration: fitness = {}",
                        result.fitness_score
                    );
                    assert!(result.fitness_score.is_finite());
                }
                Err(e) => {
                    println!("Alignment failed even with valid configuration: {:?}", e);
                    // This is also acceptable - NDT might not converge with simple test data
                }
            }
        }
    }

    #[test]
    fn test_cuda_extreme_parameter_values() {
        // Test behavior with extreme parameter values
        if let Ok(mut ndt) = NDTCuda::new() {
            // Test with extremely small resolution
            match ndt.set_resolution(1e-10) {
                Ok(_) => println!("Extremely small resolution accepted"),
                Err(e) => println!("Extremely small resolution rejected: {:?}", e),
            }

            // Test with extremely large resolution
            match ndt.set_resolution(1e10) {
                Ok(_) => println!("Extremely large resolution accepted"),
                Err(e) => println!("Extremely large resolution rejected: {:?}", e),
            }

            // Test with extremely large max iterations
            match ndt.set_max_iterations(1000000) {
                Ok(_) => println!("Extremely large max iterations accepted"),
                Err(e) => println!("Extremely large max iterations rejected: {:?}", e),
            }

            // Test with extremely small epsilon values
            match ndt.set_transformation_epsilon(1e-20) {
                Ok(_) => println!("Extremely small transformation epsilon accepted"),
                Err(e) => println!("Extremely small transformation epsilon rejected: {:?}", e),
            }
        }
    }
}
