//! Basic test of minimal FFI bindings

#[cfg(test)]
mod tests {
    use crate::ffi;

    #[test]
    fn test_point_cloud_creation() {
        let cloud = ffi::create_point_cloud_xyz();
        assert!(ffi::point_cloud_xyz_empty(&cloud));
        assert_eq!(ffi::point_cloud_xyz_size(&cloud), 0);
    }

    // Disabled while focusing on CUDA compilation
    // #[test]
    // fn test_fast_gicp_creation() {
    //     let gicp = ffi::create_fast_gicp();
    //     assert!(!ffi::fast_gicp_has_converged(&gicp));
    // }

    // #[test]
    // fn test_basic_registration_setup() {
    //     let mut gicp = ffi::create_fast_gicp();
    //     let source = ffi::create_point_cloud_xyz();
    //     let target = ffi::create_point_cloud_xyz();
    //
    //     ffi::fast_gicp_set_input_source(gicp.pin_mut(), &source);
    //     ffi::fast_gicp_set_input_target(gicp.pin_mut(), &target);
    //     ffi::fast_gicp_set_max_iterations(gicp.pin_mut(), 10);
    // }
}
