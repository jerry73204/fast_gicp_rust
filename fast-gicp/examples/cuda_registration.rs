//! CUDA-accelerated point cloud registration example using the builder pattern.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fast_gicp::{fast_vgicp_cuda::FastVGICPCuda, types::NeighborSearchMethod, PointCloudXYZ};

    println!("Fast VGICP CUDA Registration Example");

    // Create a simple test cloud
    let points = [
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

    // Create CUDA-accelerated FastVGICP using the builder pattern
    println!("Creating FastVGICPCuda with builder...");
    let vgicp_cuda = FastVGICPCuda::builder()
        .max_iterations(50)
        .resolution(1.0)
        .neighbor_search_method(NeighborSearchMethod::Direct27)
        .build()?;

    println!("FastVGICPCuda configured with builder pattern");

    // Perform CUDA registration
    println!("Performing CUDA registration...");
    let result = vgicp_cuda.align(&source_cloud, &target_cloud)?;

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
