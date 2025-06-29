//! CUDA-accelerated point cloud registration example demonstrating performance benefits.
//!
//! This example shows:
//! - Using FastVGICPCuda for GPU-accelerated registration
//! - Comparing performance with CPU-based FastVGICP
//! - Working with larger point clouds where GPU acceleration shines

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example cuda_registration --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fast_gicp::{types::NeighborSearchMethod, FastVGICP, FastVGICPCuda, PointCloudXYZ};
    use nalgebra::{UnitQuaternion, Vector3};
    use std::time::Instant;

    println!("Fast VGICP CUDA Registration Example");
    println!("=====================================");

    // Create a larger point cloud to demonstrate GPU acceleration benefits
    let grid_size = 20; // 20x20x20 = 8000 points
    let mut source_points = Vec::new();

    // Generate a 3D grid of points
    for x in 0..grid_size {
        for y in 0..grid_size {
            for z in 0..grid_size {
                source_points.push([x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1]);
            }
        }
    }

    // Add some noise to make it more realistic
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    for point in &mut source_points {
        point[0] += rng.random_range(-0.01..0.01);
        point[1] += rng.random_range(-0.01..0.01);
        point[2] += rng.random_range(-0.01..0.01);
    }

    let source_cloud: PointCloudXYZ = source_points.iter().copied().collect();
    println!("Created source cloud with {} points", source_cloud.size());

    // Create target cloud with known transformation
    let translation = Vector3::new(0.5, 0.3, 0.2);
    let rotation = UnitQuaternion::from_euler_angles(0.1, 0.15, 0.2);
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

    // First, run CPU-based FastVGICP for comparison
    println!("\n--- CPU-based FastVGICP ---");
    let cpu_vgicp = FastVGICP::builder()
        .resolution(0.1)
        .max_iterations(50)
        .transformation_epsilon(1e-6)
        .build()?;

    let cpu_start = Instant::now();
    let cpu_result = cpu_vgicp.align(&source_cloud, &target_cloud)?;
    let cpu_duration = cpu_start.elapsed();

    println!(
        "CPU Registration completed in: {:.3} ms",
        cpu_duration.as_secs_f64() * 1000.0
    );
    println!("Converged: {}", cpu_result.has_converged);
    println!("Fitness score: {:.6}", cpu_result.fitness_score);
    println!("Iterations: {}", cpu_result.num_iterations);

    // Now run CUDA-accelerated FastVGICPCuda
    println!("\n--- CUDA-accelerated FastVGICPCuda ---");
    let cuda_vgicp = FastVGICPCuda::builder()
        .resolution(0.1)
        .max_iterations(50)
        .transformation_epsilon(1e-6)
        .neighbor_search_method(NeighborSearchMethod::Direct27)
        .build()?;

    let cuda_start = Instant::now();
    let cuda_result = cuda_vgicp.align(&source_cloud, &target_cloud)?;
    let cuda_duration = cuda_start.elapsed();

    println!(
        "CUDA Registration completed in: {:.3} ms",
        cuda_duration.as_secs_f64() * 1000.0
    );
    println!("Converged: {}", cuda_result.has_converged);
    println!("Fitness score: {:.6}", cuda_result.fitness_score);
    println!("Iterations: {}", cuda_result.num_iterations);

    // Compare performance
    let speedup = cpu_duration.as_secs_f64() / cuda_duration.as_secs_f64();
    println!("\n--- Performance Comparison ---");
    println!("Speedup: {speedup:.2}x");

    // Verify both methods produce similar results
    let cpu_transform = cpu_result.final_transformation.to_isometry();
    let cuda_transform = cuda_result.final_transformation.to_isometry();

    let translation_diff =
        (cpu_transform.translation.vector - cuda_transform.translation.vector).norm();
    let rotation_diff = (cpu_transform.rotation.inverse() * cuda_transform.rotation).angle();

    println!("\n--- Result Comparison ---");
    println!("Translation difference: {translation_diff:.6} units");
    println!("Rotation difference: {rotation_diff:.6} radians");

    if translation_diff < 0.001 && rotation_diff < 0.001 {
        println!("✓ CPU and CUDA results are consistent!");
    } else {
        println!("⚠ CPU and CUDA results differ slightly");
    }

    // Calculate accuracy vs ground truth
    let estimated_transform = cuda_result.final_transformation.to_isometry();
    let translation_error = (estimated_transform.translation.vector - translation).norm();
    let rotation_error = (estimated_transform.rotation.inverse() * rotation).angle();

    println!("\n--- Accuracy vs Ground Truth ---");
    println!("Translation error: {translation_error:.6} units");
    println!(
        "Rotation error: {:.6} radians ({:.2} degrees)",
        rotation_error,
        rotation_error.to_degrees()
    );

    if cuda_result.has_converged && translation_error < 0.01 && rotation_error < 0.01 {
        println!("\n✓ CUDA registration successful!");
    } else {
        println!("\n⚠ CUDA registration may not be optimal");
    }

    Ok(())
}
