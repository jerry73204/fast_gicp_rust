//! Integration tests based on upstream integration_test.cu
//!
//! End-to-end registration with synthetic data.

use fast_gicp::{FastGICP, FastVGICP, PointCloudXYZ, Transform3f};
use nalgebra::Vector3;
use std::f32::consts::PI;

/// Generate a synthetic bunny-like shape for consistent testing
fn generate_synthetic_bunny(num_points: usize) -> PointCloudXYZ {
    let mut points = Vec::with_capacity(num_points);

    // Create a bunny-like shape using parametric equations
    for i in 0..num_points {
        let t = 2.0 * PI * (i as f32) / (num_points as f32);
        let s = (i as f32) / (num_points as f32);

        // Body (ellipsoid)
        let body_x = 2.0 * t.cos() * (1.0 - s * 0.3);
        let body_y = 1.5 * t.sin() * (1.0 - s * 0.3);
        let body_z = 3.0 * s - 1.5;

        // Add some variation for ears
        let ear_offset = if s > 0.7 && s < 0.9 {
            let ear_t = (s - 0.7) * 10.0 * PI;
            ear_t.sin() * 0.5
        } else {
            0.0
        };

        points.push([body_x, body_y + ear_offset, body_z]);
    }

    // Add some noise for realism
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42); // Deterministic for testing

    for point in &mut points {
        point[0] += rng.gen_range(-0.01..0.01);
        point[1] += rng.gen_range(-0.01..0.01);
        point[2] += rng.gen_range(-0.01..0.01);
    }

    points.into_iter().collect()
}

/// Test registration with different point cloud sizes
#[test]
fn test_multi_scale_registration() {
    let sizes = [100, 1000, 10000];

    for size in sizes {
        let source = generate_synthetic_bunny(size);

        // Apply a known transformation
        let transform = Transform3f::from_parts(
            Vector3::new(1.0f32, 0.5, -0.5),
            nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::y_axis(), PI / 6.0).into_inner(),
        );

        let mut target = PointCloudXYZ::new().unwrap();
        for i in 0..source.size() {
            let p = source.get(i).unwrap();
            let v = Vector3::new(p[0], p[1], p[2]);
            let transformed = transform.transform_point(&v);
            target
                .push([transformed.x, transformed.y, transformed.z])
                .unwrap();
        }

        // Test with FastGICP
        let mut gicp = FastGICP::new().unwrap();
        gicp.set_input_source(&source).unwrap();
        gicp.set_input_target(&target).unwrap();
        gicp.set_max_iterations(50).unwrap();

        let result = gicp.align(None).unwrap();

        assert!(
            result.has_converged,
            "GICP should converge for {} points",
            size
        );

        // Calculate error
        let recovered = result.final_transformation;
        let trans_diff = (recovered.translation() - transform.translation()).norm();
        let rot1 = recovered.rotation();
        let rot2 = transform.rotation();
        let q1 = nalgebra::UnitQuaternion::from_matrix(&rot1);
        let q2 = nalgebra::UnitQuaternion::from_matrix(&rot2);
        let rot_diff = q1.angle_to(&q2);

        // Larger point clouds should give better results
        let expected_trans_err = match size {
            100 => 0.2,
            1000 => 0.1,
            10000 => 0.05,
            _ => 0.1,
        };

        assert!(
            trans_diff < expected_trans_err,
            "Translation error for {} points should be < {}, got {}",
            size,
            expected_trans_err,
            trans_diff
        );
        assert!(
            rot_diff < 0.05,
            "Rotation error for {} points should be < 0.05, got {}",
            size,
            rot_diff
        );
    }
}

/// Test bidirectional registration (forward and backward)
#[test]
fn test_bidirectional_registration() {
    let source = generate_synthetic_bunny(500);
    let transform = Transform3f::from_parts(
        Vector3::new(0.3f32, -0.2, 0.1),
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::z_axis(), PI / 8.0).into_inner(),
    );

    let mut target = PointCloudXYZ::new().unwrap();
    for i in 0..source.size() {
        let p = source.get(i).unwrap();
        let v = Vector3::new(p[0], p[1], p[2]);
        let transformed = transform.transform_point(&v);
        target
            .push([transformed.x, transformed.y, transformed.z])
            .unwrap();
    }

    // Forward registration (source to target)
    let mut gicp_forward = FastGICP::new().unwrap();
    gicp_forward.set_input_source(&source).unwrap();
    gicp_forward.set_input_target(&target).unwrap();
    gicp_forward.set_max_iterations(50).unwrap();

    let forward_result = gicp_forward.align(None).unwrap();
    assert!(
        forward_result.has_converged,
        "Forward registration should converge"
    );

    // Backward registration (target to source)
    let mut gicp_backward = FastGICP::new().unwrap();
    gicp_backward.set_input_source(&target).unwrap();
    gicp_backward.set_input_target(&source).unwrap();
    gicp_backward.set_max_iterations(50).unwrap();

    let backward_result = gicp_backward.align(None).unwrap();
    assert!(
        backward_result.has_converged,
        "Backward registration should converge"
    );

    // The forward and backward transforms should be inverses
    let forward_trans = forward_result.final_transformation;
    let backward_trans = backward_result.final_transformation;

    // Compose the transforms - should be close to identity
    let composed =
        Transform3f::from_matrix(&(forward_trans.to_matrix() * backward_trans.to_matrix()));
    let identity = Transform3f::identity();

    let trans_diff = (composed.translation() - identity.translation()).norm();
    let rot1 = composed.rotation();
    let rot2 = identity.rotation();
    let q1 = nalgebra::UnitQuaternion::from_matrix(&rot1);
    let q2 = nalgebra::UnitQuaternion::from_matrix(&rot2);
    let rot_diff = q1.angle_to(&q2);

    assert!(
        trans_diff < 0.1,
        "Composed translation should be near identity, got {}",
        trans_diff
    );
    assert!(
        rot_diff < 0.1,
        "Composed rotation should be near identity, got {}",
        rot_diff
    );
}

/// Test VGICP with different voxel resolutions
#[test]
fn test_vgicp_resolution_impact() {
    let source = generate_synthetic_bunny(2000);
    let transform = Transform3f::from_parts(
        Vector3::new(0.2f32, 0.1, -0.1),
        nalgebra::Rotation3::<f32>::from_axis_angle(&Vector3::x_axis(), PI / 10.0).into_inner(),
    );

    let mut target = PointCloudXYZ::new().unwrap();
    for i in 0..source.size() {
        let p = source.get(i).unwrap();
        let v = Vector3::new(p[0], p[1], p[2]);
        let transformed = transform.transform_point(&v);
        target
            .push([transformed.x, transformed.y, transformed.z])
            .unwrap();
    }

    let resolutions = [0.1, 0.2, 0.5, 1.0];

    for resolution in resolutions {
        let mut vgicp = FastVGICP::new().unwrap();
        vgicp.set_input_source(&source).unwrap();
        vgicp.set_input_target(&target).unwrap();
        vgicp.set_resolution(resolution).unwrap();
        vgicp.set_max_iterations(50).unwrap();

        let result = vgicp.align(None).unwrap();

        assert!(
            result.has_converged,
            "VGICP should converge with resolution {}",
            resolution
        );

        // Coarser resolutions may have higher error
        let expected_error = (resolution * 1.0) as f32; // More lenient for voxelized approach
        let trans_diff =
            (result.final_transformation.translation() - transform.translation()).norm();

        assert!(
            trans_diff < expected_error.max(0.3),
            "Translation error with resolution {} should be reasonable, got {}",
            resolution,
            trans_diff
        );
    }
}

/// Test registration robustness with partial overlap
#[test]
fn test_partial_overlap_registration() {
    let full_cloud = generate_synthetic_bunny(1000);

    // Create source as first 70% of points
    let mut source = PointCloudXYZ::new().unwrap();
    for i in 0..(full_cloud.size() * 7 / 10) {
        let p = full_cloud.get(i).unwrap();
        source.push(p).unwrap();
    }

    // Create target as last 70% of points (40% overlap)
    let mut target = PointCloudXYZ::new().unwrap();
    for i in (full_cloud.size() * 3 / 10)..full_cloud.size() {
        let p = full_cloud.get(i).unwrap();
        target.push(p).unwrap();
    }

    // Apply small transformation to target
    let transform = Transform3f::from_translation(0.05, 0.05, 0.05);
    let mut transformed_target = PointCloudXYZ::new().unwrap();
    for i in 0..target.size() {
        let p = target.get(i).unwrap();
        let v = Vector3::new(p[0], p[1], p[2]);
        let t = transform.transform_point(&v);
        transformed_target.push([t.x, t.y, t.z]).unwrap();
    }

    let mut gicp = FastGICP::new().unwrap();
    gicp.set_input_source(&source).unwrap();
    gicp.set_input_target(&transformed_target).unwrap();
    gicp.set_max_iterations(100).unwrap();
    gicp.set_max_correspondence_distance(0.5).unwrap(); // Limit correspondence distance

    let result = gicp.align(None).unwrap();

    // With partial overlap, we expect convergence but potentially higher error
    assert!(
        result.has_converged,
        "GICP should converge even with partial overlap"
    );

    let trans_diff = (result.final_transformation.translation() - transform.translation()).norm();
    assert!(
        trans_diff < 0.2,
        "Translation error should be reasonable with partial overlap"
    );
}
