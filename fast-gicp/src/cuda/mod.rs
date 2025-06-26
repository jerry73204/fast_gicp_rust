//! CUDA-accelerated registration algorithms.

pub mod fast_vgicp;
pub mod ndt;

pub use fast_vgicp::FastVGICPCuda;
pub use ndt::NDTCuda;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_creation() {
        let _fast_vgicp_cuda = FastVGICPCuda::new();
        let _ndt_cuda = NDTCuda::new();
    }
}
