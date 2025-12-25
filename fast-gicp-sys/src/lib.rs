//! Low-level FFI bindings for the fast_gicp C++ library.
//!
//! This crate provides unsafe bindings to the fast_gicp C++ library through the cxx crate.
//! For a safe, high-level API, use the `fast-gicp` crate instead.

// Configuration based on features:
// - Default: Use cxx::bridge directly with C++ compilation
// - docs-only: Use pre-generated stub (no C++ compilation)
// - bindgen: Regenerate stubs (for maintainers)

// Normal builds: use actual cxx::bridge directly
#[cfg(not(feature = "docs-only"))]
#[cxx::bridge]
pub mod ffi {
    // FFI-safe structs
    #[derive(Debug, Clone, Copy)]
    pub struct Transform4f {
        pub data: [f32; 16],
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Point3f {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Point4f {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        pub intensity: f32,
    }

    /// 6x6 Hessian matrix (row-major order)
    #[derive(Debug, Clone, Copy)]
    pub struct Hessian6x6 {
        pub data: [f64; 36],
    }

    unsafe extern "C++" {
        include!("wrapper.h");

        // === Core Types ===
        type PointCloudXYZ;
        type PointCloudXYZI;

        // Registration types
        type FastGICP;
        type FastVGICP;
        type FastGICPI;
        type FastVGICPI;

        // CUDA types (always declared for CXX compatibility, dummy types when CUDA disabled)
        type FastVGICPCuda;
        type NDTCuda;

        // === Point Cloud Factory Functions ===
        fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ>;
        fn create_point_cloud_xyzi() -> UniquePtr<PointCloudXYZI>;

        // === Point Cloud Operations ===
        fn point_cloud_xyz_size(cloud: &PointCloudXYZ) -> usize;
        fn point_cloud_xyz_empty(cloud: &PointCloudXYZ) -> bool;
        fn point_cloud_xyz_clear(cloud: Pin<&mut PointCloudXYZ>);
        fn point_cloud_xyz_reserve(cloud: Pin<&mut PointCloudXYZ>, capacity: usize);
        fn point_cloud_xyz_push_point(cloud: Pin<&mut PointCloudXYZ>, x: f32, y: f32, z: f32);
        fn point_cloud_xyz_get_point(cloud: &PointCloudXYZ, index: usize) -> Point3f;
        fn point_cloud_xyz_set_point(
            cloud: Pin<&mut PointCloudXYZ>,
            index: usize,
            x: f32,
            y: f32,
            z: f32,
        );

        fn point_cloud_xyzi_size(cloud: &PointCloudXYZI) -> usize;
        fn point_cloud_xyzi_empty(cloud: &PointCloudXYZI) -> bool;
        fn point_cloud_xyzi_clear(cloud: Pin<&mut PointCloudXYZI>);
        fn point_cloud_xyzi_reserve(cloud: Pin<&mut PointCloudXYZI>, capacity: usize);
        fn point_cloud_xyzi_push_point(
            cloud: Pin<&mut PointCloudXYZI>,
            x: f32,
            y: f32,
            z: f32,
            intensity: f32,
        );
        fn point_cloud_xyzi_get_point(cloud: &PointCloudXYZI, index: usize) -> Point4f;
        fn point_cloud_xyzi_set_point(
            cloud: Pin<&mut PointCloudXYZI>,
            index: usize,
            x: f32,
            y: f32,
            z: f32,
            intensity: f32,
        );

        // === GICP Factory Functions ===
        fn create_fast_gicp() -> UniquePtr<FastGICP>;
        fn create_fast_vgicp() -> UniquePtr<FastVGICP>;
        fn create_fast_gicp_i() -> UniquePtr<FastGICPI>;
        fn create_fast_vgicp_i() -> UniquePtr<FastVGICPI>;

        // CUDA factory functions (conditionally compiled)
        #[cfg(feature = "cuda")]
        fn create_fast_vgicp_cuda() -> UniquePtr<FastVGICPCuda>;
        #[cfg(feature = "cuda")]
        fn create_ndt_cuda() -> UniquePtr<NDTCuda>;

        // === Registration Configuration ===
        fn fast_gicp_set_input_source(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ);
        fn fast_gicp_set_input_target(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ);
        fn fast_gicp_set_max_iterations(gicp: Pin<&mut FastGICP>, max_iterations: i32);
        fn fast_gicp_set_transformation_epsilon(gicp: Pin<&mut FastGICP>, eps: f64);
        fn fast_gicp_set_euclidean_fitness_epsilon(gicp: Pin<&mut FastGICP>, eps: f64);
        fn fast_gicp_set_max_correspondence_distance(gicp: Pin<&mut FastGICP>, distance: f64);
        fn fast_gicp_set_num_threads(gicp: Pin<&mut FastGICP>, num_threads: i32);
        fn fast_gicp_set_correspondence_randomness(gicp: Pin<&mut FastGICP>, k: i32);
        fn fast_gicp_set_regularization_method(gicp: Pin<&mut FastGICP>, method: i32);
        fn fast_gicp_set_rotation_epsilon(gicp: Pin<&mut FastGICP>, eps: f64);

        fn fast_vgicp_set_input_source(vgicp: Pin<&mut FastVGICP>, cloud: &PointCloudXYZ);
        fn fast_vgicp_set_input_target(vgicp: Pin<&mut FastVGICP>, cloud: &PointCloudXYZ);
        fn fast_vgicp_set_max_iterations(vgicp: Pin<&mut FastVGICP>, max_iterations: i32);
        fn fast_vgicp_set_transformation_epsilon(vgicp: Pin<&mut FastVGICP>, eps: f64);
        fn fast_vgicp_set_euclidean_fitness_epsilon(vgicp: Pin<&mut FastVGICP>, eps: f64);
        fn fast_vgicp_set_max_correspondence_distance(vgicp: Pin<&mut FastVGICP>, distance: f64);
        fn fast_vgicp_set_resolution(vgicp: Pin<&mut FastVGICP>, resolution: f64);
        fn fast_vgicp_set_num_threads(vgicp: Pin<&mut FastVGICP>, num_threads: i32);
        fn fast_vgicp_set_regularization_method(vgicp: Pin<&mut FastVGICP>, method: i32);
        fn fast_vgicp_set_voxel_accumulation_mode(vgicp: Pin<&mut FastVGICP>, mode: i32);
        fn fast_vgicp_set_neighbor_search_method(vgicp: Pin<&mut FastVGICP>, method: i32);

        // CUDA configuration functions (conditionally compiled)
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_input_source(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            cloud: &PointCloudXYZ,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_input_target(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            cloud: &PointCloudXYZ,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_max_iterations(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            max_iterations: i32,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_transformation_epsilon(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            eps: f64,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_euclidean_fitness_epsilon(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            eps: f64,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_max_correspondence_distance(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            distance: f64,
        );
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_resolution(cuda_vgicp: Pin<&mut FastVGICPCuda>, resolution: f64);
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_set_neighbor_search_method(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            method: i32,
        );

        // === Registration Execution ===
        fn fast_gicp_align(gicp: Pin<&mut FastGICP>) -> Transform4f;
        fn fast_gicp_align_with_guess(gicp: Pin<&mut FastGICP>, guess: &Transform4f)
            -> Transform4f;

        fn fast_vgicp_align(vgicp: Pin<&mut FastVGICP>) -> Transform4f;
        fn fast_vgicp_align_with_guess(
            vgicp: Pin<&mut FastVGICP>,
            guess: &Transform4f,
        ) -> Transform4f;

        // CUDA execution functions (conditionally compiled)
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_align(cuda_vgicp: Pin<&mut FastVGICPCuda>) -> Transform4f;
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_align_with_guess(
            cuda_vgicp: Pin<&mut FastVGICPCuda>,
            guess: &Transform4f,
        ) -> Transform4f;

        // === Registration Status ===
        fn fast_gicp_has_converged(gicp: &FastGICP) -> bool;
        fn fast_gicp_get_fitness_score(gicp: &FastGICP) -> f64;
        fn fast_gicp_get_final_num_iterations(gicp: &FastGICP) -> i32;

        fn fast_vgicp_has_converged(vgicp: &FastVGICP) -> bool;
        fn fast_vgicp_get_fitness_score(vgicp: &FastVGICP) -> f64;
        fn fast_vgicp_get_final_num_iterations(vgicp: &FastVGICP) -> i32;

        // CUDA status functions (conditionally compiled)
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_has_converged(cuda_vgicp: &FastVGICPCuda) -> bool;
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_get_fitness_score(cuda_vgicp: &FastVGICPCuda) -> f64;
        #[cfg(feature = "cuda")]
        fn fast_vgicp_cuda_get_final_num_iterations(cuda_vgicp: &FastVGICPCuda) -> i32;

        // === NDTCuda Operations ===
        // NDTCuda type and factory function are already declared above for CXX compatibility

        // NDTCuda configuration
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_input_source(ndt_cuda: Pin<&mut NDTCuda>, cloud: &PointCloudXYZ);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_input_target(ndt_cuda: Pin<&mut NDTCuda>, cloud: &PointCloudXYZ);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_max_iterations(ndt_cuda: Pin<&mut NDTCuda>, max_iterations: i32);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_transformation_epsilon(ndt_cuda: Pin<&mut NDTCuda>, eps: f64);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_euclidean_fitness_epsilon(ndt_cuda: Pin<&mut NDTCuda>, eps: f64);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_max_correspondence_distance(ndt_cuda: Pin<&mut NDTCuda>, distance: f64);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_resolution(ndt_cuda: Pin<&mut NDTCuda>, resolution: f64);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_distance_mode(ndt_cuda: Pin<&mut NDTCuda>, mode: i32);
        #[cfg(feature = "cuda")]
        fn ndt_cuda_set_neighbor_search_method(
            ndt_cuda: Pin<&mut NDTCuda>,
            method: i32,
            radius: f64,
        );

        // NDTCuda execution
        #[cfg(feature = "cuda")]
        fn ndt_cuda_align(ndt_cuda: Pin<&mut NDTCuda>) -> Transform4f;
        #[cfg(feature = "cuda")]
        fn ndt_cuda_align_with_guess(
            ndt_cuda: Pin<&mut NDTCuda>,
            guess: &Transform4f,
        ) -> Transform4f;

        // NDTCuda status
        #[cfg(feature = "cuda")]
        fn ndt_cuda_has_converged(ndt_cuda: &NDTCuda) -> bool;
        #[cfg(feature = "cuda")]
        fn ndt_cuda_get_fitness_score(ndt_cuda: &NDTCuda) -> f64;
        #[cfg(feature = "cuda")]
        fn ndt_cuda_get_final_num_iterations(ndt_cuda: &NDTCuda) -> i32;

        // NDTCuda Hessian and cost evaluation (for covariance estimation)
        #[cfg(feature = "cuda")]
        fn ndt_cuda_get_hessian(ndt_cuda: &NDTCuda) -> Hessian6x6;
        #[cfg(feature = "cuda")]
        fn ndt_cuda_evaluate_cost(ndt_cuda: &NDTCuda, pose: &Transform4f) -> f64;

        // NDTCuda NVTL (Nearest Voxel Transformation Likelihood) scoring
        #[cfg(feature = "cuda")]
        fn ndt_cuda_evaluate_nvtl(
            ndt_cuda: Pin<&mut NDTCuda>,
            pose: &Transform4f,
            outlier_ratio: f64,
        ) -> f64;

        // === Transform Utilities ===
        fn transform_identity() -> Transform4f;
        fn transform_from_translation(x: f32, y: f32, z: f32) -> Transform4f;
        fn transform_multiply(a: &Transform4f, b: &Transform4f) -> Transform4f;
        fn transform_inverse(t: &Transform4f) -> Transform4f;
    }
}

// Docs-only builds: use appropriate stub based on CUDA feature
#[cfg(all(feature = "docs-only", not(feature = "cuda")))]
include!("generated/stub.rs");

#[cfg(all(feature = "docs-only", feature = "cuda"))]
include!("generated/stub_cuda.rs");

// Re-export the types for convenience
pub use ffi::{Hessian6x6, Point3f, Point4f, Transform4f};

#[cfg(test)]
mod test;

#[cfg(feature = "docs-only")]
#[cfg(test)]
mod docs_only_tests {
    use super::ffi::{Point3f, Point4f, Transform4f};

    #[test]
    fn test_types_exist() {
        // Test that basic types can be created
        let _point3f = Point3f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let _point4f = Point4f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 4.0,
        };
        let _transform = Transform4f { data: [0.0; 16] };
    }
}
