//! Type definitions for fast_gicp algorithms.

/// Regularization methods for covariance matrices in GICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RegularizationMethod {
    /// No regularization
    None = 0,
    /// Minimum eigenvalue regularization
    MinEig = 1,
    /// Normalized minimum eigenvalue regularization
    NormalizedMinEig = 2,
    /// Plane regularization
    Plane = 3,
    /// Frobenius norm regularization
    Frobenius = 4,
}

impl Default for RegularizationMethod {
    fn default() -> Self {
        Self::None
    }
}

/// Voxel accumulation modes for FastVGICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum VoxelAccumulationMode {
    /// Additive accumulation (ADD_COV in some contexts)
    Additive = 0,
    /// Additive weighted accumulation (ADD_POINT in some contexts)
    AdditiveWeighted = 1,
    /// Multiplicative accumulation
    Multiplicative = 2,
}

impl Default for VoxelAccumulationMode {
    fn default() -> Self {
        Self::Additive
    }
}

/// Neighbor search methods for voxelized GICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NeighborSearchMethod {
    /// Direct search with 27 neighbors (3x3x3 cube)
    Direct27 = 0,
    /// Direct search with 7 neighbors (center + 6 faces)
    Direct7 = 1,
    /// Direct search with 1 neighbor (center only)
    Direct1 = 2,
    /// Direct search within radius
    DirectRadius = 3,
}

impl Default for NeighborSearchMethod {
    fn default() -> Self {
        Self::Direct27
    }
}

/// Nearest neighbor search methods for CUDA implementation.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NearestNeighborMethod {
    /// CPU parallel KD-tree
    CpuParallelKdtree = 0,
    /// GPU brute force search
    GpuBruteforce = 1,
    /// GPU RBF kernel search
    GpuRbfKernel = 2,
}

#[cfg(feature = "cuda")]
impl Default for NearestNeighborMethod {
    fn default() -> Self {
        Self::GpuBruteforce
    }
}

/// NDT distance calculation mode.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NdtDistanceMode {
    /// Point-to-Distribution distance
    P2D = 0,
    /// Distribution-to-Distribution distance
    D2D = 1,
}

#[cfg(feature = "cuda")]
impl Default for NdtDistanceMode {
    fn default() -> Self {
        Self::P2D
    }
}
