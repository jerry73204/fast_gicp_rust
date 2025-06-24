#include "fast-gicp-sys/src/lib.rs.h"
#include <Eigen/Dense>
#include <stdexcept>

// Explicit template instantiations to ensure the right templates are used
template class fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ, KdTreeType,
                                   KdTreeType>;
template class fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI, KdTreeTypeI,
                                   KdTreeTypeI>;
template class fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI>;

#ifdef BUILD_VGICP_CUDA
template class fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#endif

// Helper function to convert Eigen::Matrix4f to Transform4f
Transform4f eigen_to_transform4f(const Eigen::Matrix4f &eigen_matrix) {
  Transform4f result;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      result.data[i * 4 + j] = eigen_matrix(i, j);
    }
  }
  return result;
}

// Helper function to convert Transform4f to Eigen::Matrix4f
Eigen::Matrix4f transform4f_to_eigen(const Transform4f &transform) {
  Eigen::Matrix4f result;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      result(i, j) = transform.data[i * 4 + j];
    }
  }
  return result;
}

// === Point Cloud Factory Functions ===
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz() {
  return std::make_unique<PointCloudXYZ>();
}

std::unique_ptr<PointCloudXYZI> create_point_cloud_xyzi() {
  return std::make_unique<PointCloudXYZI>();
}

// === Point Cloud Operations ===
size_t point_cloud_xyz_size(const PointCloudXYZ &cloud) { return cloud.size(); }

bool point_cloud_xyz_empty(const PointCloudXYZ &cloud) { return cloud.empty(); }

void point_cloud_xyz_clear(PointCloudXYZ &cloud) { cloud.clear(); }

void point_cloud_xyz_reserve(PointCloudXYZ &cloud, size_t capacity) {
  cloud.reserve(capacity);
}

void point_cloud_xyz_push_point(PointCloudXYZ &cloud, float x, float y,
                                float z) {
  pcl::PointXYZ point;
  point.x = x;
  point.y = y;
  point.z = z;
  cloud.push_back(point);
}

Point3f point_cloud_xyz_get_point(const PointCloudXYZ &cloud, size_t index) {
  if (index >= cloud.size()) {
    throw std::out_of_range("Point cloud index out of range");
  }
  const auto &point = cloud[index];
  return {point.x, point.y, point.z};
}

void point_cloud_xyz_set_point(PointCloudXYZ &cloud, size_t index, float x,
                               float y, float z) {
  if (index >= cloud.size()) {
    throw std::out_of_range("Point cloud index out of range");
  }
  cloud[index].x = x;
  cloud[index].y = y;
  cloud[index].z = z;
}

size_t point_cloud_xyzi_size(const PointCloudXYZI &cloud) {
  return cloud.size();
}

bool point_cloud_xyzi_empty(const PointCloudXYZI &cloud) {
  return cloud.empty();
}

void point_cloud_xyzi_clear(PointCloudXYZI &cloud) { cloud.clear(); }

void point_cloud_xyzi_reserve(PointCloudXYZI &cloud, size_t capacity) {
  cloud.reserve(capacity);
}

void point_cloud_xyzi_push_point(PointCloudXYZI &cloud, float x, float y,
                                 float z, float intensity) {
  pcl::PointXYZI point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  cloud.push_back(point);
}

Point4f point_cloud_xyzi_get_point(const PointCloudXYZI &cloud, size_t index) {
  if (index >= cloud.size()) {
    throw std::out_of_range("Point cloud index out of range");
  }
  const auto &point = cloud[index];
  return {point.x, point.y, point.z, point.intensity};
}

void point_cloud_xyzi_set_point(PointCloudXYZI &cloud, size_t index, float x,
                                float y, float z, float intensity) {
  if (index >= cloud.size()) {
    throw std::out_of_range("Point cloud index out of range");
  }
  cloud[index].x = x;
  cloud[index].y = y;
  cloud[index].z = z;
  cloud[index].intensity = intensity;
}

// === GICP Factory Functions ===
std::unique_ptr<FastGICP> create_fast_gicp() {
  return std::make_unique<FastGICP>();
}

std::unique_ptr<FastVGICP> create_fast_vgicp() {
  return std::make_unique<FastVGICP>();
}

std::unique_ptr<FastGICPI> create_fast_gicp_i() {
  return std::make_unique<FastGICPI>();
}

std::unique_ptr<FastVGICPI> create_fast_vgicp_i() {
  return std::make_unique<FastVGICPI>();
}

#ifdef BUILD_VGICP_CUDA
std::unique_ptr<FastVGICPCuda> create_fast_vgicp_cuda() {
  return std::make_unique<FastVGICPCuda>();
}

std::unique_ptr<NDTCuda> create_ndt_cuda() {
  return std::make_unique<NDTCuda>();
}
#else
std::unique_ptr<FastVGICPCuda> create_fast_vgicp_cuda() {
  throw std::runtime_error("CUDA support not available in this build");
}

std::unique_ptr<NDTCuda> create_ndt_cuda() {
  throw std::runtime_error("CUDA support not available in this build");
}
#endif

// === Registration Configuration ===
void fast_gicp_set_input_source(FastGICP &gicp, const PointCloudXYZ &cloud) {
  gicp.setInputSource(cloud.makeShared());
}

void fast_gicp_set_input_target(FastGICP &gicp, const PointCloudXYZ &cloud) {
  gicp.setInputTarget(cloud.makeShared());
}

void fast_gicp_set_max_iterations(FastGICP &gicp, int max_iterations) {
  gicp.setMaximumIterations(max_iterations);
}

void fast_gicp_set_transformation_epsilon(FastGICP &gicp, double eps) {
  gicp.setTransformationEpsilon(eps);
}

void fast_gicp_set_euclidean_fitness_epsilon(FastGICP &gicp, double eps) {
  gicp.setEuclideanFitnessEpsilon(eps);
}

void fast_gicp_set_max_correspondence_distance(FastGICP &gicp,
                                               double distance) {
  gicp.setMaxCorrespondenceDistance(distance);
}

void fast_vgicp_set_input_source(FastVGICP &vgicp, const PointCloudXYZ &cloud) {
  vgicp.setInputSource(cloud.makeShared());
}

void fast_vgicp_set_input_target(FastVGICP &vgicp, const PointCloudXYZ &cloud) {
  vgicp.setInputTarget(cloud.makeShared());
}

void fast_vgicp_set_max_iterations(FastVGICP &vgicp, int max_iterations) {
  vgicp.setMaximumIterations(max_iterations);
}

void fast_vgicp_set_transformation_epsilon(FastVGICP &vgicp, double eps) {
  vgicp.setTransformationEpsilon(eps);
}

void fast_vgicp_set_euclidean_fitness_epsilon(FastVGICP &vgicp, double eps) {
  vgicp.setEuclideanFitnessEpsilon(eps);
}

void fast_vgicp_set_max_correspondence_distance(FastVGICP &vgicp,
                                                double distance) {
  vgicp.setMaxCorrespondenceDistance(distance);
}

void fast_vgicp_set_resolution(FastVGICP &vgicp, double resolution) {
  vgicp.setResolution(resolution);
}

#ifdef BUILD_VGICP_CUDA
void fast_vgicp_cuda_set_input_source(FastVGICPCuda &cuda_vgicp,
                                      const PointCloudXYZ &cloud) {
  cuda_vgicp.setInputSource(cloud.makeShared());
}

void fast_vgicp_cuda_set_input_target(FastVGICPCuda &cuda_vgicp,
                                      const PointCloudXYZ &cloud) {
  cuda_vgicp.setInputTarget(cloud.makeShared());
}

void fast_vgicp_cuda_set_max_iterations(FastVGICPCuda &cuda_vgicp,
                                        int max_iterations) {
  cuda_vgicp.setMaximumIterations(max_iterations);
}

void fast_vgicp_cuda_set_transformation_epsilon(FastVGICPCuda &cuda_vgicp,
                                                double eps) {
  cuda_vgicp.setTransformationEpsilon(eps);
}

void fast_vgicp_cuda_set_euclidean_fitness_epsilon(FastVGICPCuda &cuda_vgicp,
                                                   double eps) {
  cuda_vgicp.setEuclideanFitnessEpsilon(eps);
}

void fast_vgicp_cuda_set_max_correspondence_distance(FastVGICPCuda &cuda_vgicp,
                                                     double distance) {
  cuda_vgicp.setMaxCorrespondenceDistance(distance);
}

void fast_vgicp_cuda_set_resolution(FastVGICPCuda &cuda_vgicp,
                                    double resolution) {
  cuda_vgicp.setResolution(resolution);
}

void fast_vgicp_cuda_set_neighbor_search_method(FastVGICPCuda &cuda_vgicp,
                                                int method) {
  cuda_vgicp.setNeighborSearchMethod(
      static_cast<fast_gicp::NeighborSearchMethod>(method));
}
#else
void fast_vgicp_cuda_set_input_source(FastVGICPCuda &, const PointCloudXYZ &) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_input_target(FastVGICPCuda &, const PointCloudXYZ &) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_max_iterations(FastVGICPCuda &, int) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_transformation_epsilon(FastVGICPCuda &, double) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_euclidean_fitness_epsilon(FastVGICPCuda &, double) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_max_correspondence_distance(FastVGICPCuda &, double) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_resolution(FastVGICPCuda &, double) {
  throw std::runtime_error("CUDA support not available in this build");
}

void fast_vgicp_cuda_set_neighbor_search_method(FastVGICPCuda &, int) {
  throw std::runtime_error("CUDA support not available in this build");
}
#endif

// === Registration Execution ===
Transform4f fast_gicp_align(FastGICP &gicp) {
  PointCloudXYZ output;
  gicp.align(output);
  return eigen_to_transform4f(gicp.getFinalTransformation());
}

Transform4f fast_gicp_align_with_guess(FastGICP &gicp,
                                       const Transform4f &guess) {
  PointCloudXYZ output;
  Eigen::Matrix4f guess_matrix = transform4f_to_eigen(guess);
  gicp.align(output, guess_matrix);
  return eigen_to_transform4f(gicp.getFinalTransformation());
}

Transform4f fast_vgicp_align(FastVGICP &vgicp) {
  PointCloudXYZ output;
  vgicp.align(output);
  return eigen_to_transform4f(vgicp.getFinalTransformation());
}

Transform4f fast_vgicp_align_with_guess(FastVGICP &vgicp,
                                        const Transform4f &guess) {
  PointCloudXYZ output;
  Eigen::Matrix4f guess_matrix = transform4f_to_eigen(guess);
  vgicp.align(output, guess_matrix);
  return eigen_to_transform4f(vgicp.getFinalTransformation());
}

#ifdef BUILD_VGICP_CUDA
Transform4f fast_vgicp_cuda_align(FastVGICPCuda &cuda_vgicp) {
  PointCloudXYZ output;
  cuda_vgicp.align(output);
  return eigen_to_transform4f(cuda_vgicp.getFinalTransformation());
}

Transform4f fast_vgicp_cuda_align_with_guess(FastVGICPCuda &cuda_vgicp,
                                             const Transform4f &guess) {
  PointCloudXYZ output;
  Eigen::Matrix4f guess_matrix = transform4f_to_eigen(guess);
  cuda_vgicp.align(output, guess_matrix);
  return eigen_to_transform4f(cuda_vgicp.getFinalTransformation());
}
#else
Transform4f fast_vgicp_cuda_align(FastVGICPCuda &) {
  throw std::runtime_error("CUDA support not available in this build");
}

Transform4f fast_vgicp_cuda_align_with_guess(FastVGICPCuda &, const Transform4f &) {
  throw std::runtime_error("CUDA support not available in this build");
}
#endif

// === Registration Status ===
bool fast_gicp_has_converged(const FastGICP &gicp) {
  return gicp.hasConverged();
}

double fast_gicp_get_fitness_score(const FastGICP &gicp) {
  return const_cast<FastGICP&>(gicp).getFitnessScore();
}

int fast_gicp_get_final_num_iterations(const FastGICP &gicp) {
  // Fast GICP doesn't expose final iteration count - return max iterations as approximation
  return const_cast<FastGICP&>(gicp).getMaximumIterations();
}

bool fast_vgicp_has_converged(const FastVGICP &vgicp) {
  return vgicp.hasConverged();
}

double fast_vgicp_get_fitness_score(const FastVGICP &vgicp) {
  return const_cast<FastVGICP&>(vgicp).getFitnessScore();
}

int fast_vgicp_get_final_num_iterations(const FastVGICP &vgicp) {
  // Fast VGICP doesn't expose final iteration count - return max iterations as approximation
  return const_cast<FastVGICP&>(vgicp).getMaximumIterations();
}

#ifdef BUILD_VGICP_CUDA
bool fast_vgicp_cuda_has_converged(const FastVGICPCuda &cuda_vgicp) {
  return cuda_vgicp.hasConverged();
}

double fast_vgicp_cuda_get_fitness_score(const FastVGICPCuda &cuda_vgicp) {
  return const_cast<FastVGICPCuda&>(cuda_vgicp).getFitnessScore();
}

int fast_vgicp_cuda_get_final_num_iterations(const FastVGICPCuda &cuda_vgicp) {
  // Fast VGICP CUDA doesn't expose final iteration count - return max iterations as approximation
  return const_cast<FastVGICPCuda&>(cuda_vgicp).getMaximumIterations();
}
#else
// Stub implementations when CUDA is not available - these should never be called
bool fast_vgicp_cuda_has_converged(const FastVGICPCuda &) {
  throw std::runtime_error("CUDA support not available in this build");
}

double fast_vgicp_cuda_get_fitness_score(const FastVGICPCuda &) {
  throw std::runtime_error("CUDA support not available in this build");
}

int fast_vgicp_cuda_get_final_num_iterations(const FastVGICPCuda &) {
  throw std::runtime_error("CUDA support not available in this build");
}
#endif

// === Transform Utilities ===
Transform4f transform_identity() {
  Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  return eigen_to_transform4f(identity);
}

Transform4f transform_from_translation(float x, float y, float z) {
  Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
  translation(0, 3) = x;
  translation(1, 3) = y;
  translation(2, 3) = z;
  return eigen_to_transform4f(translation);
}

Transform4f transform_multiply(const Transform4f &a, const Transform4f &b) {
  Eigen::Matrix4f matrix_a = transform4f_to_eigen(a);
  Eigen::Matrix4f matrix_b = transform4f_to_eigen(b);
  return eigen_to_transform4f(matrix_a * matrix_b);
}

Transform4f transform_inverse(const Transform4f &t) {
  Eigen::Matrix4f matrix = transform4f_to_eigen(t);
  return eigen_to_transform4f(matrix.inverse());
}
