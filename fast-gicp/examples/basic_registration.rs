//! Basic point cloud registration example using FastGICP with the builder pattern.

use fast_gicp::{FastGICP, PointCloudXYZ, Transform3f};
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fast GICP Basic Registration Example");

    // Create source point cloud (a simple cube)
    let source_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // Add some random points for robustness
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ];

    let source_cloud: PointCloudXYZ = source_points.iter().copied().collect();
    println!("Created source cloud with {} points", source_cloud.size());

    // Create target point cloud (same cube, but translated and rotated)
    let translation = Vector3::new(2.0, 1.0, 0.5);
    let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let true_transform = nalgebra::Isometry3::from_parts(translation.into(), rotation);

    let target_points: Vec<[f32; 3]> = source_points
        .iter()
        .map(|&p| {
            let point = Vector3::new(p[0], p[1], p[2]);
            let transformed = true_transform * point;
            [transformed.x, transformed.y, transformed.z]
        })
        .collect();

    let target_cloud: PointCloudXYZ = target_points.into_iter().collect();
    println!("Created target cloud with {} points", target_cloud.size());

    // Create and configure FastGICP using the builder pattern
    let gicp = FastGICP::builder()
        .max_iterations(50)
        .transformation_epsilon(1e-6)
        .euclidean_fitness_epsilon(1e-6)
        .max_correspondence_distance(1.0)
        .build()?;

    println!("FastGICP configured with builder pattern");

    // Perform registration with identity initial guess
    println!("Performing registration...");
    let result = gicp.align(&source_cloud, &target_cloud)?;

    // Display results
    println!("Registration completed!");
    println!("Converged: {}", result.has_converged);
    println!("Fitness score: {:.6}", result.fitness_score);
    println!("Number of iterations: {}", result.num_iterations);

    // Compare with ground truth
    let estimated_transform = result.final_transformation.to_isometry();
    let _ground_truth_transform = Transform3f::from_isometry(&true_transform);

    println!("\nTransformation comparison:");
    println!(
        "Ground truth translation: [{:.3}, {:.3}, {:.3}]",
        translation.x, translation.y, translation.z
    );
    println!(
        "Estimated translation:    [{:.3}, {:.3}, {:.3}]",
        estimated_transform.translation.x,
        estimated_transform.translation.y,
        estimated_transform.translation.z
    );

    // Calculate translation error
    let translation_error = (estimated_transform.translation.vector - translation).norm();
    println!("Translation error: {translation_error:.6} units");

    // Calculate rotation error (angle difference)
    let rotation_error = (estimated_transform.rotation.inverse() * rotation).angle();
    println!(
        "Rotation error: {:.6} radians ({:.2} degrees)",
        rotation_error,
        rotation_error.to_degrees()
    );

    if result.has_converged && translation_error < 0.1 && rotation_error < 0.1 {
        println!("\n✓ Registration successful!");
    } else {
        println!("\n⚠ Registration may not be optimal");
    }

    Ok(())
}
