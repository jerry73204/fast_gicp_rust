#include <fast_gicp/cuda/nvtl_score.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/vector3_hash.cuh>
#include <fast_gicp/cuda/cuda_context.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>

namespace fast_gicp {
namespace cuda {

/// Gaussian fitting parameters from Autoware's NDT implementation
/// Based on [Magnusson 2009] equations
struct GaussParams {
  float d1;
  float d2;

  __host__ __device__ GaussParams(float resolution, float outlier_ratio) {
    // Autoware's Gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
    float gauss_c1 = 10.0f * (1.0f - outlier_ratio);
    float gauss_c2 = outlier_ratio / (resolution * resolution * resolution);
    float gauss_d3 = -logf(gauss_c2);
    d1 = -(logf(gauss_c1 + gauss_c2) - gauss_d3);
    d2 = -2.0f * logf((-logf(gauss_c1 * expf(-0.5f) + gauss_c2) - gauss_d3) / d1);
  }

  /// Compute score using Mahalanobis distance
  /// This is Autoware's score_inc = -gauss_d1_ * exp(-gauss_d2_ * mahal_dist_sq / 2)
  __host__ __device__ float score(float mahal_dist_sq) const {
    return -d1 * expf(-d2 * mahal_dist_sq / 2.0f);
  }
};

/// Kernel to compute NVTL score for each source point
struct nvtl_score_kernel {
  nvtl_score_kernel(
      const thrust::device_ptr<const VoxelMapInfo>& voxelmap_info_ptr,
      const thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>>& buckets_ptr,
      const thrust::device_ptr<const int>& num_points_ptr,
      const thrust::device_ptr<const Eigen::Vector3f>& voxel_means_ptr,
      const thrust::device_ptr<const Eigen::Matrix3f>& voxel_covs_ptr,
      const thrust::device_ptr<const Eigen::Isometry3f>& transform_ptr,
      const GaussParams& gauss)
      : voxelmap_info_ptr(voxelmap_info_ptr),
        buckets_ptr(buckets_ptr),
        num_points_ptr(num_points_ptr),
        voxel_means_ptr(voxel_means_ptr),
        voxel_covs_ptr(voxel_covs_ptr),
        transform_ptr(transform_ptr),
        gauss(gauss) {}

  /// Look up voxel index from coordinate using hash table
  __device__ int lookup_voxel(const Eigen::Vector3i& coord) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);
    uint64_t hash = vector3i_hash(coord);

    for (int i = 0; i < info.max_bucket_scan_count; i++) {
      uint64_t bucket_index = (hash + i) % info.num_buckets;
      const auto& bucket = thrust::raw_pointer_cast(buckets_ptr)[bucket_index];

      if (equal(bucket.first, coord)) {
        return bucket.second;
      }
    }
    return -1;
  }

  /// Compute score for a single source point
  __device__ thrust::pair<float, int> operator()(const Eigen::Vector3f& source_pt) const {
    const auto& info = *thrust::raw_pointer_cast(voxelmap_info_ptr);
    const auto& transform = *thrust::raw_pointer_cast(transform_ptr);

    // Transform source point
    Eigen::Vector3f transformed_pt = transform.linear() * source_pt + transform.translation();

    // Get voxel coordinate
    Eigen::Vector3i coord = calc_voxel_coord(transformed_pt, info.voxel_resolution);

    // Search 3x3x3 neighborhood for best score
    float max_score = -1e10f;
    bool found = false;

    for (int di = -1; di <= 1; di++) {
      for (int dj = -1; dj <= 1; dj++) {
        for (int dk = -1; dk <= 1; dk++) {
          Eigen::Vector3i neighbor_coord(coord[0] + di, coord[1] + dj, coord[2] + dk);
          int voxel_idx = lookup_voxel(neighbor_coord);

          if (voxel_idx < 0) continue;

          int num_pts = thrust::raw_pointer_cast(num_points_ptr)[voxel_idx];
          if (num_pts < 6) continue;  // Autoware requires at least 6 points

          const Eigen::Vector3f& mean = thrust::raw_pointer_cast(voxel_means_ptr)[voxel_idx];
          const Eigen::Matrix3f& cov = thrust::raw_pointer_cast(voxel_covs_ptr)[voxel_idx];

          // Compute Mahalanobis distance squared
          // First invert the covariance matrix
          // Note: cov is already regularized, so it should be invertible
          Eigen::Matrix3f cov_inv = cov.inverse();

          Eigen::Vector3f diff = transformed_pt - mean;
          float mahal_sq = diff.dot(cov_inv * diff);

          float score = gauss.score(mahal_sq);
          if (score > max_score) {
            max_score = score;
            found = true;
          }
        }
      }
    }

    if (found) {
      return thrust::make_pair(max_score, 1);
    } else {
      return thrust::make_pair(0.0f, 0);
    }
  }

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;
  thrust::device_ptr<const int> num_points_ptr;
  thrust::device_ptr<const Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> voxel_covs_ptr;
  thrust::device_ptr<const Eigen::Isometry3f> transform_ptr;
  GaussParams gauss;
};

/// Reduction functor to sum scores and counts
struct score_sum_functor {
  __host__ __device__ thrust::pair<float, int> operator()(
      const thrust::pair<float, int>& a,
      const thrust::pair<float, int>& b) const {
    return thrust::make_pair(a.first + b.first, a.second + b.second);
  }
};

double compute_nvtl_score(
    const GaussianVoxelMap& target_voxelmap,
    const device_vector<Eigen::Vector3f>& source_points,
    const Eigen::Isometry3f& transform,
    double resolution,
    double outlier_ratio) {

  if (source_points.empty()) {
    return 0.0;
  }

  CudaExecutionContext ctx("nvtl_score");

  // Upload transform to device
  device_vector<Eigen::Isometry3f> transform_device(1);
  transform_device[0] = transform;

  // Create Gaussian parameters
  GaussParams gauss(static_cast<float>(resolution), static_cast<float>(outlier_ratio));

  // Compute NVTL score using transform_reduce
  auto result = thrust::transform_reduce(
      thrust::cuda::par.on(ctx.stream()),
      source_points.begin(),
      source_points.end(),
      nvtl_score_kernel(
          target_voxelmap.voxelmap_info_ptr.data(),
          target_voxelmap.buckets.data(),
          target_voxelmap.num_points.data(),
          target_voxelmap.voxel_means.data(),
          target_voxelmap.voxel_covs.data(),
          transform_device.data(),
          gauss),
      thrust::make_pair(0.0f, 0),
      score_sum_functor());

  ctx.synchronize();

  if (result.second > 0) {
    return static_cast<double>(result.first) / static_cast<double>(result.second);
  } else {
    return 0.0;
  }
}

}  // namespace cuda
}  // namespace fast_gicp
