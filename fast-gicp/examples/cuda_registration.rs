//! CUDA-accelerated point cloud registration example.

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fast_gicp::{cuda::FastVGICPCuda, PointCloudXYZ, Transform3f};

    println!("Fast VGICP CUDA Registration Example");

    // Create a simple test cloud
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];

    let source_cloud: PointCloudXYZ = points.iter().copied().collect();
    let target_cloud: PointCloudXYZ = points.iter().copied().collect();

    println!("Created clouds with {} points each", source_cloud.size());

    // Create CUDA-accelerated FastVGICP
    let mut vgicp_cuda = FastVGICPCuda::new();

    // TODO: This will fail until CUDA methods are implemented
    println!("Setting input clouds...");
    match vgicp_cuda.set_input_source(&source_cloud) {
        Ok(_) => println!("Source cloud set successfully"),
        Err(e) => println!("Failed to set source cloud: {}", e),
    }

    match vgicp_cuda.set_input_target(&target_cloud) {
        Ok(_) => println!("Target cloud set successfully"),
        Err(e) => println!("Failed to set target cloud: {}", e),
    }

    // TODO: This will fail until align is implemented
    println!("Performing CUDA registration...");
    match vgicp_cuda.align(&Transform3f::identity()) {
        Ok(result) => {
            println!("Registration completed!");
            println!("Converged: {}", result.has_converged);
            println!("Fitness score: {:.6}", result.fitness_score);
        }
        Err(e) => println!("Registration failed: {}", e),
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example cuda_registration --features cuda");
}
