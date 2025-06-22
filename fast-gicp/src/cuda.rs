//! CUDA-accelerated registration algorithms.

#[cfg(feature = "cuda")]
pub mod fast_vgicp_cuda {
    //! FastVGICP with CUDA acceleration.

    use crate::{FastGicpError, PointCloudXYZ, RegistrationResult, Result, Transform3f};
    use cxx::UniquePtr;
    use fast_gicp_sys::ffi;
    use std::pin::Pin;

    /// Fast Voxelized GICP with CUDA acceleration.
    pub struct FastVGICPCuda {
        inner: UniquePtr<ffi::FastVGICPCuda>,
    }

    impl FastVGICPCuda {
        /// Creates a new FastVGICPCuda instance.
        pub fn new() -> Self {
            Self {
                inner: ffi::create_fast_vgicp_cuda(),
            }
        }

        /// TODO: Implement CUDA-specific methods
        pub fn set_input_source(&mut self, _cloud: &PointCloudXYZ) -> Result<()> {
            todo!("Implement CUDA FastVGICP set_input_source")
        }

        /// TODO: Implement CUDA-specific methods
        pub fn set_input_target(&mut self, _cloud: &PointCloudXYZ) -> Result<()> {
            todo!("Implement CUDA FastVGICP set_input_target")
        }

        /// TODO: Implement CUDA-specific methods
        pub fn align(&mut self, _initial_guess: &Transform3f) -> Result<RegistrationResult> {
            todo!("Implement CUDA FastVGICP align")
        }
    }

    impl Default for FastVGICPCuda {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "cuda")]
pub mod ndt_cuda {
    //! NDT with CUDA acceleration.

    use crate::{FastGicpError, PointCloudXYZ, RegistrationResult, Result, Transform3f};
    use cxx::UniquePtr;
    use fast_gicp_sys::ffi;
    use std::pin::Pin;

    /// Normal Distributions Transform with CUDA acceleration.
    pub struct NDTCuda {
        inner: UniquePtr<ffi::NDTCuda>,
    }

    impl NDTCuda {
        /// Creates a new NDTCuda instance.
        pub fn new() -> Self {
            Self {
                inner: ffi::create_ndt_cuda(),
            }
        }

        /// TODO: Implement CUDA-specific methods
        pub fn set_input_source(&mut self, _cloud: &PointCloudXYZ) -> Result<()> {
            todo!("Implement CUDA NDT set_input_source")
        }

        /// TODO: Implement CUDA-specific methods
        pub fn set_input_target(&mut self, _cloud: &PointCloudXYZ) -> Result<()> {
            todo!("Implement CUDA NDT set_input_target")
        }

        /// TODO: Implement CUDA-specific methods
        pub fn align(&mut self, _initial_guess: &Transform3f) -> Result<RegistrationResult> {
            todo!("Implement CUDA NDT align")
        }
    }

    impl Default for NDTCuda {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "cuda")]
pub use fast_vgicp_cuda::FastVGICPCuda;

#[cfg(feature = "cuda")]
pub use ndt_cuda::NDTCuda;

#[cfg(not(feature = "cuda"))]
compile_error!("CUDA module requires the 'cuda' feature to be enabled");

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_creation() {
        let _fast_vgicp_cuda = fast_vgicp_cuda::FastVGICPCuda::new();
        let _ndt_cuda = ndt_cuda::NDTCuda::new();
    }
}
