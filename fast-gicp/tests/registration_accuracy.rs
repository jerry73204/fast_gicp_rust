//! Registration accuracy tests based on upstream gicp_test.cpp
//!
//! These tests compare against expected behavior from the C++ implementation,
//! treating it as ground truth.

#![cfg(not(feature = "docs-only"))]

use fast_gicp::{FastGICP, FastVGICP, PointCloudXYZ, RegularizationMethod, Transform3f};
use nalgebra::Vector3;
use std::f32::consts::PI;

/// Create a test point cloud with a simple cubic pattern
fn create_test_cube() -> PointCloudXYZ {
    let points = vec![
        // Bottom face
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        // Top face
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        // Add some points in between for better registration
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
    ];

    points.into_iter().collect()
}

/// Create a denser point cloud for more robust testing
fn create_dense_cloud(size: usize) -> PointCloudXYZ {
    let mut points = Vec::new();
    let grid_size = (size as f32).sqrt().ceil() as usize;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if points.len() >= size {
                break;
            }
            let x = -1.0 + 2.0 * (i as f32) / (grid_size as f32 - 1.0);
            let y = -1.0 + 2.0 * (j as f32) / (grid_size as f32 - 1.0);
            // Create a more interesting 3D surface
            let z = 0.5 * ((x * 2.0).sin() + (y * 2.0).cos());
            points.push([x, y, z]);
        }
    }

    points.into_iter().collect()
}

/// Apply a transformation to a point cloud
fn transform_cloud(cloud: &PointCloudXYZ, transform: &Transform3f) -> PointCloudXYZ {
    let mut transformed = PointCloudXYZ::new();
    transformed.reserve(cloud.size());

    for i in 0..cloud.size() {
        let point = cloud.get(i).unwrap();
        let transformed_point =
            transform.transform_point(&Vector3::new(point[0], point[1], point[2]));
        transformed.push([
            transformed_point.x,
            transformed_point.y,
            transformed_point.z,
        ]);
    }

    transformed
}

/// Calculate the error between two transforms
fn transform_error(t1: &Transform3f, t2: &Transform3f) -> (f64, f64) {
    // Translation error (Euclidean distance)
    let trans_diff = t1.translation() - t2.translation();
    let translation_error = trans_diff.norm() as f64;

    // Rotation error (angle difference in radians)
    let rot1 = t1.rotation();
    let rot2 = t2.rotation();
    // Using quaternions to calculate angle difference
    let q1 = nalgebra::UnitQuaternion::from_matrix(&rot1);
    let q2 = nalgebra::UnitQuaternion::from_matrix(&rot2);
    let rotation_error = q1.angle_to(&q2) as f64;

    (translation_error, rotation_error)
}

#[test]
fn test_fast_gicp_identity_transform() {
    let cloud = create_test_cube();

    let gicp = FastGICP::builder().max_iterations(10).build().unwrap();

    let result = gicp.align(&cloud, &cloud).unwrap();

    assert!(
        result.has_converged,
        "GICP should converge for identity transform"
    );
    assert!(
        result.fitness_score < 1e-6,
        "Fitness score should be near zero for identical clouds"
    );

    let identity = Transform3f::identity();
    let (trans_err, rot_err) = transform_error(&result.final_transformation, &identity);

    assert!(
        trans_err < 1e-6,
        "Translation error should be minimal for identity"
    );
    assert!(
        rot_err < 1e-6,
        "Rotation error should be minimal for identity"
    );
}

#[test]
fn test_fast_gicp_translation_only() {
    let source = create_test_cube();
    let translation = Transform3f::from_translation(0.5, -0.3, 0.2);
    let target = transform_cloud(&source, &translation);

    let gicp = FastGICP::builder()
        .max_iterations(50)
        .transformation_epsilon(1e-8)
        .build()
        .unwrap();

    let result = gicp.align(&source, &target).unwrap();

    assert!(
        result.has_converged,
        "GICP should converge for simple translation"
    );

    let (trans_err, rot_err) = transform_error(&result.final_transformation, &translation);

    // Based on C++ test expectations
    assert!(
        trans_err < 0.1,
        "Translation error should be < 0.1, got {trans_err}"
    );
    assert!(
        rot_err < 0.01,
        "Rotation error should be < 0.01 radians, got {rot_err}"
    );
}

#[test]
fn test_fast_gicp_rotation_only() {
    // Use more points and a smaller rotation for better accuracy
    let source = create_dense_cloud(500);
    let rotation_matrix =
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::z_axis(), PI / 12.0); // 15 degrees
    let rotation = Transform3f::from_rotation_matrix(&rotation_matrix.into_inner());
    let target = transform_cloud(&source, &rotation);

    let gicp = FastGICP::builder()
        .max_iterations(100)
        .transformation_epsilon(1e-8)
        .rotation_epsilon(1e-6)
        .max_correspondence_distance(1.0)
        .build()
        .unwrap();

    let result = gicp.align(&source, &target).unwrap();

    assert!(result.has_converged, "GICP should converge for rotation");

    let (trans_err, rot_err) = transform_error(&result.final_transformation, &rotation);

    println!("Rotation only test:");
    println!("  Translation error: {trans_err}");
    println!("  Rotation error: {rot_err} radians");
    println!("  Converged: {}", result.has_converged);
    println!("  Fitness score: {}", result.fitness_score);

    assert!(
        trans_err < 0.1,
        "Translation error should be < 0.1, got {trans_err}"
    );
    assert!(
        rot_err < 0.05,
        "Rotation error should be < 0.05 radians, got {rot_err}"
    );
}

#[test]
fn test_fast_gicp_combined_transform() {
    let source = create_dense_cloud(200);

    // Create a combined rotation and translation
    let rotation =
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::y_axis(), PI / 8.0).into_inner();
    let translation = Vector3::new(0.3, -0.2, 0.5);
    let transform = Transform3f::from_parts(translation, rotation);

    let target = transform_cloud(&source, &transform);

    let gicp = FastGICP::builder()
        .max_iterations(100)
        .transformation_epsilon(1e-8)
        .euclidean_fitness_epsilon(1e-6)
        .build()
        .unwrap();

    let result = gicp.align(&source, &target).unwrap();

    assert!(result.has_converged, "GICP should converge");

    let (trans_err, rot_err) = transform_error(&result.final_transformation, &transform);

    assert!(trans_err < 0.1, "Translation error should be < 0.1");
    assert!(rot_err < 0.01, "Rotation error should be < 0.01 radians");
}

#[test]
fn test_fast_gicp_with_initial_guess() {
    let source = create_dense_cloud(150);
    let transform = Transform3f::from_parts(
        Vector3::new(1.0, 0.5, -0.3),
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::x_axis(), PI / 4.0).into_inner(),
    );
    let target = transform_cloud(&source, &transform);

    // Provide a close initial guess
    let initial_guess = Transform3f::from_parts(
        Vector3::new(0.9, 0.4, -0.2),
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::x_axis(), PI / 5.0).into_inner(),
    );

    let gicp = FastGICP::builder().max_iterations(30).build().unwrap();

    let result = gicp
        .align_with_guess(&source, &target, Some(&initial_guess))
        .unwrap();

    assert!(
        result.has_converged,
        "GICP should converge with initial guess"
    );
    assert!(
        result.num_iterations <= 30,
        "Should converge faster with good initial guess"
    );

    let (trans_err, rot_err) = transform_error(&result.final_transformation, &transform);

    assert!(trans_err < 0.1, "Translation error should be < 0.1");
    assert!(rot_err < 0.01, "Rotation error should be < 0.01 radians");
}

#[test]
fn test_fast_gicp_regularization_methods() {
    let source = create_test_cube();
    let transform = Transform3f::from_translation(0.2, 0.1, 0.15);
    let target = transform_cloud(&source, &transform);

    let methods = [
        RegularizationMethod::None,
        RegularizationMethod::MinEig,
        RegularizationMethod::Frobenius,
    ];

    for method in methods {
        let gicp = FastGICP::builder()
            .regularization_method(method)
            .max_iterations(50)
            .build()
            .unwrap();

        let result = gicp.align(&source, &target).unwrap();

        assert!(
            result.has_converged,
            "GICP should converge with {method:?} regularization"
        );

        let (trans_err, _) = transform_error(&result.final_transformation, &transform);
        assert!(
            trans_err < 0.2,
            "Translation error should be reasonable with {method:?}"
        );
    }
}

#[test]
fn test_fast_vgicp_basic() {
    let source = create_dense_cloud(300);
    let transform = Transform3f::from_parts(
        Vector3::new(0.1, -0.1, 0.2),
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::z_axis(), PI / 12.0).into_inner(),
    );
    let target = transform_cloud(&source, &transform);

    let vgicp = FastVGICP::builder()
        .resolution(0.2) // Voxel size
        .max_iterations(50)
        .build()
        .unwrap();

    let result = vgicp.align(&source, &target).unwrap();

    assert!(result.has_converged, "VGICP should converge");

    let (trans_err, rot_err) = transform_error(&result.final_transformation, &transform);

    // VGICP may be slightly less accurate due to voxelization
    println!("VGICP test:");
    println!("  Translation error: {trans_err}");
    println!("  Rotation error: {rot_err} radians");
    println!("  Converged: {}", result.has_converged);
    println!("  Fitness score: {}", result.fitness_score);

    assert!(
        trans_err < 0.2,
        "Translation error should be < 0.2, got {trans_err}"
    );
    assert!(
        rot_err < 0.1,
        "Rotation error should be < 0.1 radians, got {rot_err}"
    );
}

#[test]
fn test_fast_gicp_correspondence_randomness() {
    let source = create_dense_cloud(500);
    let transform = Transform3f::from_translation(0.05, 0.05, 0.05);
    let target = transform_cloud(&source, &transform);

    let gicp = FastGICP::builder()
        .correspondence_randomness(10) // Sample 10 random correspondences
        .max_iterations(50)
        .build()
        .unwrap();

    let result = gicp.align(&source, &target).unwrap();

    assert!(
        result.has_converged,
        "GICP should converge with correspondence sampling"
    );

    let (trans_err, _) = transform_error(&result.final_transformation, &transform);
    assert!(trans_err < 0.1, "Translation error should be < 0.1");
}

#[test]
fn test_fast_gicp_multithreading() {
    let source = create_dense_cloud(1000);
    let transform = Transform3f::from_translation(0.1, 0.0, 0.0);
    let target = transform_cloud(&source, &transform);

    // Test with different thread counts
    for num_threads in [1, 2, 4, 0] {
        // 0 means use all available threads
        let gicp = FastGICP::builder()
            .num_threads(num_threads)
            .max_iterations(20)
            .build()
            .unwrap();

        let result = gicp.align(&source, &target).unwrap();

        assert!(
            result.has_converged,
            "GICP should converge with {num_threads} threads"
        );

        let (trans_err, _) = transform_error(&result.final_transformation, &transform);
        assert!(
            trans_err < 0.1,
            "Translation error should be < 0.1 with {num_threads} threads"
        );
    }
}
