//! Minimal FFI bindings to get the build working

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("minimal_wrapper.h");

        // Just the basic types without custom structs
        type PointCloudXYZ;
        // type FastGICP;  // Disabled for CUDA compilation focus

        // Minimal factory functions
        fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ>;
        // fn create_fast_gicp() -> UniquePtr<FastGICP>;  // Disabled for now

        // Basic info functions
        fn point_cloud_xyz_size(cloud: &PointCloudXYZ) -> usize;
        fn point_cloud_xyz_empty(cloud: &PointCloudXYZ) -> bool;

        // Basic registration function - disabled for CUDA compilation focus
        // fn fast_gicp_set_input_source(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ);
        // fn fast_gicp_set_input_target(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ);
        // fn fast_gicp_set_max_iterations(gicp: Pin<&mut FastGICP>, max_iterations: i32);
        // fn fast_gicp_has_converged(gicp: &FastGICP) -> bool;
    }
}
