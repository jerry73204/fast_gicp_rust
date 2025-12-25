#ifndef FAST_GICP_CUDA_NVTL_SCORE_CUH
#define FAST_GICP_CUDA_NVTL_SCORE_CUH

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "cuda_types.h"

namespace fast_gicp {
namespace cuda {

class GaussianVoxelMap;

/// Compute NVTL (Nearest Voxel Transformation Likelihood) score.
///
/// This implements Autoware's NVTL metric for evaluating how well
/// a transformed point cloud matches a target voxel map.
///
/// @param target_voxelmap The target (map) voxel grid with means and covariances
/// @param source_points The source points to transform and score
/// @param transform The transformation to apply to source points
/// @param resolution Voxel resolution (for Gaussian fitting parameters)
/// @param outlier_ratio Outlier ratio for Gaussian parameters (Autoware default: 0.55)
/// @return NVTL score (higher = better alignment), typically in range [0, ~5]
double compute_nvtl_score(
    const GaussianVoxelMap& target_voxelmap,
    const device_vector<Eigen::Vector3f>& source_points,
    const Eigen::Isometry3f& transform,
    double resolution,
    double outlier_ratio = 0.55);

}  // namespace cuda
}  // namespace fast_gicp

#endif
