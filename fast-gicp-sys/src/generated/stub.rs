#[doc = r" FFI bindings stub for documentation generation."]
#[doc = r""]
#[doc = r" This module provides type definitions for documentation purposes when"]
#[doc = r" building on docs.rs where C++ dependencies are not available."]
pub mod ffi {
    use std::pin::Pin;
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
    #[repr(C)]
    pub struct PointCloudXYZ {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct PointCloudXYZI {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct FastGICP {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct FastVGICP {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct FastGICPI {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct FastVGICPI {
        _private: [u8; 0],
    }
    #[allow(unused_variables, dead_code)]
    fn create_point_cloud_xyz() -> UniquePtr<PointCloudXYZ> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn create_point_cloud_xyzi() -> UniquePtr<PointCloudXYZI> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_size(cloud: &PointCloudXYZ) -> usize {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_empty(cloud: &PointCloudXYZ) -> bool {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_clear(cloud: Pin<&mut PointCloudXYZ>) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_reserve(cloud: Pin<&mut PointCloudXYZ>, capacity: usize) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_push_point(cloud: Pin<&mut PointCloudXYZ>, x: f32, y: f32, z: f32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_get_point(cloud: &PointCloudXYZ, index: usize) -> Point3f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyz_set_point(
        cloud: Pin<&mut PointCloudXYZ>,
        index: usize,
        x: f32,
        y: f32,
        z: f32,
    ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_size(cloud: &PointCloudXYZI) -> usize {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_empty(cloud: &PointCloudXYZI) -> bool {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_clear(cloud: Pin<&mut PointCloudXYZI>) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_reserve(cloud: Pin<&mut PointCloudXYZI>, capacity: usize) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_push_point(
        cloud: Pin<&mut PointCloudXYZI>,
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_get_point(cloud: &PointCloudXYZI, index: usize) -> Point4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn point_cloud_xyzi_set_point(
        cloud: Pin<&mut PointCloudXYZI>,
        index: usize,
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn create_fast_gicp() -> UniquePtr<FastGICP> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn create_fast_vgicp() -> UniquePtr<FastVGICP> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn create_fast_gicp_i() -> UniquePtr<FastGICPI> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn create_fast_vgicp_i() -> UniquePtr<FastVGICPI> {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_input_source(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_input_target(gicp: Pin<&mut FastGICP>, cloud: &PointCloudXYZ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_max_iterations(gicp: Pin<&mut FastGICP>, max_iterations: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_transformation_epsilon(gicp: Pin<&mut FastGICP>, eps: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_euclidean_fitness_epsilon(gicp: Pin<&mut FastGICP>, eps: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_max_correspondence_distance(gicp: Pin<&mut FastGICP>, distance: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_num_threads(gicp: Pin<&mut FastGICP>, num_threads: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_correspondence_randomness(gicp: Pin<&mut FastGICP>, k: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_regularization_method(gicp: Pin<&mut FastGICP>, method: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_set_rotation_epsilon(gicp: Pin<&mut FastGICP>, eps: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_input_source(vgicp: Pin<&mut FastVGICP>, cloud: &PointCloudXYZ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_input_target(vgicp: Pin<&mut FastVGICP>, cloud: &PointCloudXYZ) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_max_iterations(vgicp: Pin<&mut FastVGICP>, max_iterations: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_transformation_epsilon(vgicp: Pin<&mut FastVGICP>, eps: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_euclidean_fitness_epsilon(vgicp: Pin<&mut FastVGICP>, eps: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_max_correspondence_distance(vgicp: Pin<&mut FastVGICP>, distance: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_resolution(vgicp: Pin<&mut FastVGICP>, resolution: f64) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_num_threads(vgicp: Pin<&mut FastVGICP>, num_threads: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_regularization_method(vgicp: Pin<&mut FastVGICP>, method: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_voxel_accumulation_mode(vgicp: Pin<&mut FastVGICP>, mode: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_set_neighbor_search_method(vgicp: Pin<&mut FastVGICP>, method: i32) {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_align(gicp: Pin<&mut FastGICP>) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_align_with_guess(gicp: Pin<&mut FastGICP>, guess: &Transform4f) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_align(vgicp: Pin<&mut FastVGICP>) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_align_with_guess(vgicp: Pin<&mut FastVGICP>, guess: &Transform4f) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_has_converged(gicp: &FastGICP) -> bool {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_get_fitness_score(gicp: &FastGICP) -> f64 {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_gicp_get_final_num_iterations(gicp: &FastGICP) -> i32 {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_has_converged(vgicp: &FastVGICP) -> bool {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_get_fitness_score(vgicp: &FastVGICP) -> f64 {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn fast_vgicp_get_final_num_iterations(vgicp: &FastVGICP) -> i32 {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn transform_identity() -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn transform_from_translation(x: f32, y: f32, z: f32) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn transform_multiply(a: &Transform4f, b: &Transform4f) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[allow(unused_variables, dead_code)]
    fn transform_inverse(t: &Transform4f) -> Transform4f {
        unreachable!("docs-only stub")
    }
    #[doc = r" CXX UniquePtr type (stub implementation)"]
    pub struct UniquePtr<T> {
        _ptr: *mut T,
        _marker: std::marker::PhantomData<T>,
    }
    impl<T> UniquePtr<T> {
        #[doc = r" Create a null UniquePtr"]
        pub fn null() -> Self {
            Self {
                _ptr: std::ptr::null_mut(),
                _marker: std::marker::PhantomData,
            }
        }
    }
    unsafe impl<T> Send for UniquePtr<T> where T: Send {}
    unsafe impl<T> Sync for UniquePtr<T> where T: Sync {}
}
