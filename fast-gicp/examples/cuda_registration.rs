//! CUDA-accelerated point cloud registration example.

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fast_gicp::{cuda::FastVGICPCuda, PointCloudXYZ};

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
    let mut vgicp_cuda = FastVGICPCuda::new()?;

    // Set input clouds
    println!("Setting input clouds...");
    vgicp_cuda.set_input_source(&source_cloud)?;
    vgicp_cuda.set_input_target(&target_cloud)?;
    println!("Input clouds set successfully");

    // Configure CUDA-specific parameters
    vgicp_cuda.set_max_iterations(50)?;
    vgicp_cuda.set_resolution(1.0)?;
    vgicp_cuda.set_neighbor_search_method(1)?; // GPU_BRUTEFORCE

    // Perform CUDA registration
    println!("Performing CUDA registration...");
    let result = vgicp_cuda.align(None)?;

    println!("Registration completed!");
    println!("Converged: {}", result.has_converged);
    println!("Fitness score: {:.6}", result.fitness_score);
    println!("Number of iterations: {}", result.num_iterations);

    let transform = result.final_transformation;
    let translation = transform.translation();
    println!(
        "Final translation: [{:.3}, {:.3}, {:.3}]",
        translation[0], translation[1], translation[2]
    );

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the 'cuda' feature to be enabled.");
    println!("Run with: cargo run --example cuda_registration --features cuda");
}
