//! Type definitions for fast_gicp algorithms.

/// 6x6 Hessian matrix for covariance estimation.
/// The Hessian is computed during NDT optimization and can be used
/// to estimate uncertainty via Laplace approximation.
#[derive(Debug, Clone, Copy)]
pub struct Hessian6x6 {
    /// Matrix data in row-major order (36 elements for 6x6)
    pub data: [f64; 36],
}

impl Hessian6x6 {
    /// Create a new Hessian from raw data array
    pub fn from_data(data: [f64; 36]) -> Self {
        Self { data }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < 6 && col < 6);
        self.data[row * 6 + col]
    }

    /// Extract the 2x2 XY block (top-left) for 2D covariance estimation
    pub fn xy_block(&self) -> [[f64; 2]; 2] {
        [
            [self.get(0, 0), self.get(0, 1)],
            [self.get(1, 0), self.get(1, 1)],
        ]
    }
}

impl Default for Hessian6x6 {
    fn default() -> Self {
        Self { data: [0.0; 36] }
    }
}

/// Regularization methods for covariance matrices in GICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum RegularizationMethod {
    /// No regularization
    #[default]
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

/// Voxel accumulation modes for FastVGICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum VoxelAccumulationMode {
    /// Additive accumulation (ADD_COV in some contexts)
    #[default]
    Additive = 0,
    /// Additive weighted accumulation (ADD_POINT in some contexts)
    AdditiveWeighted = 1,
    /// Multiplicative accumulation
    Multiplicative = 2,
}

/// Neighbor search methods for voxelized GICP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum NeighborSearchMethod {
    /// Direct search with 27 neighbors (3x3x3 cube)
    #[default]
    Direct27 = 0,
    /// Direct search with 7 neighbors (center + 6 faces)
    Direct7 = 1,
    /// Direct search with 1 neighbor (center only)
    Direct1 = 2,
    /// Direct search within radius
    DirectRadius = 3,
}

/// Nearest neighbor search methods for CUDA implementation.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum NearestNeighborMethod {
    /// CPU parallel KD-tree
    CpuParallelKdtree = 0,
    /// GPU brute force search
    #[default]
    GpuBruteforce = 1,
    /// GPU RBF kernel search
    GpuRbfKernel = 2,
}

/// NDT distance calculation mode.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum NdtDistanceMode {
    /// Point-to-Distribution distance
    #[default]
    P2D = 0,
    /// Distribution-to-Distribution distance
    D2D = 1,
}
