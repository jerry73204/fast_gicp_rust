//! FastVGICP with CUDA acceleration.

use crate::{Error, PointCloudXYZ, RegistrationResult, Result, Transform3f};
use cxx::UniquePtr;
use fast_gicp_sys::ffi;

/// Fast Voxelized GICP with CUDA acceleration.
pub struct FastVGICPCuda {
    inner: UniquePtr<ffi::FastVGICPCuda>,
}

impl FastVGICPCuda {
    /// Creates a new FastVGICPCuda instance.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_fast_vgicp_cuda(),
        })
    }

    /// Sets the source point cloud.
    pub fn set_input_source(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(Error::EmptyPointCloud);
        }
        ffi::fast_vgicp_cuda_set_input_source(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the target point cloud.
    pub fn set_input_target(&mut self, cloud: &PointCloudXYZ) -> Result<()> {
        if cloud.is_empty() {
            return Err(Error::EmptyPointCloud);
        }
        ffi::fast_vgicp_cuda_set_input_target(self.inner.pin_mut(), cloud.as_ffi());
        Ok(())
    }

    /// Sets the maximum number of iterations.
    pub fn set_max_iterations(&mut self, max_iterations: i32) -> Result<()> {
        if max_iterations <= 0 {
            return Err(Error::InvalidParameter {
                message: "max_iterations must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_cuda_set_max_iterations(self.inner.pin_mut(), max_iterations);
        Ok(())
    }

    /// Sets the transformation epsilon for convergence.
    pub fn set_transformation_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(Error::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_vgicp_cuda_set_transformation_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the Euclidean fitness epsilon for convergence.
    pub fn set_euclidean_fitness_epsilon(&mut self, epsilon: f64) -> Result<()> {
        if epsilon < 0.0 {
            return Err(Error::InvalidParameter {
                message: "epsilon must be non-negative".to_string(),
            });
        }
        ffi::fast_vgicp_cuda_set_euclidean_fitness_epsilon(self.inner.pin_mut(), epsilon);
        Ok(())
    }

    /// Sets the maximum correspondence distance.
    pub fn set_max_correspondence_distance(&mut self, distance: f64) -> Result<()> {
        if distance <= 0.0 {
            return Err(Error::InvalidParameter {
                message: "distance must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_cuda_set_max_correspondence_distance(self.inner.pin_mut(), distance);
        Ok(())
    }

    /// Sets the voxel resolution.
    pub fn set_resolution(&mut self, resolution: f64) -> Result<()> {
        if resolution <= 0.0 {
            return Err(Error::InvalidParameter {
                message: "resolution must be positive".to_string(),
            });
        }
        ffi::fast_vgicp_cuda_set_resolution(self.inner.pin_mut(), resolution);
        Ok(())
    }

    /// Sets the neighbor search method.
    ///
    /// # Arguments
    /// * `method` - 0: CPU_PARALLEL_KDTREE, 1: GPU_BRUTEFORCE, 2: GPU_RBF_KERNEL
    pub fn set_neighbor_search_method(&mut self, method: i32) -> Result<()> {
        if !(0..=2).contains(&method) {
            return Err(Error::InvalidParameter {
                    message: "method must be 0 (CPU_PARALLEL_KDTREE), 1 (GPU_BRUTEFORCE), or 2 (GPU_RBF_KERNEL)".to_string(),
                });
        }
        ffi::fast_vgicp_cuda_set_neighbor_search_method(self.inner.pin_mut(), method);
        Ok(())
    }

    /// Performs registration with an initial guess.
    pub fn align(&mut self, initial_guess: Option<&Transform3f>) -> Result<RegistrationResult> {
        let final_transformation = if let Some(guess) = initial_guess {
            let guess_ffi = guess.as_transform4f();
            ffi::fast_vgicp_cuda_align_with_guess(self.inner.pin_mut(), &guess_ffi)
        } else {
            ffi::fast_vgicp_cuda_align(self.inner.pin_mut())
        };

        let fitness_score = ffi::fast_vgicp_cuda_get_fitness_score(&self.inner);
        let has_converged = ffi::fast_vgicp_cuda_has_converged(&self.inner);
        let num_iterations = ffi::fast_vgicp_cuda_get_final_num_iterations(&self.inner);

        Ok(RegistrationResult {
            final_transformation: Transform3f::from_transform4f(&final_transformation),
            fitness_score,
            has_converged,
            num_iterations,
        })
    }
}

impl Default for FastVGICPCuda {
    fn default() -> Self {
        Self::new().expect("Failed to create default FastVGICPCuda")
    }
}
