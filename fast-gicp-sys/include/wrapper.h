#pragma once

#include <array>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <flann/util/params.h>
#include <memory>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#ifdef BUILD_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#endif

// Forward declarations of CXX-generated types
struct Transform4f;
struct Point3f;
struct Point4f;
struct Hessian6x6;

// Type aliases for clean instantiation
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;
using KdTreeType = pcl::search::KdTree<
    pcl::PointXYZ, pcl::KdTreeFLANN<pcl::PointXYZ, flann::L2_Simple<float>>>;
using KdTreeTypeI = pcl::search::KdTree<
    pcl::PointXYZI, pcl::KdTreeFLANN<pcl::PointXYZI, flann::L2_Simple<float>>>;

// Instantiated GICP types
using FastGICP =
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ, KdTreeType, KdTreeType>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastGICPI = fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI,
                                      KdTreeTypeI, KdTreeTypeI>;
using FastVGICPI = fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI>;

// Always declare CUDA types for CXX bridge compatibility
#ifdef BUILD_VGICP_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ>;
using NDTCuda = fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
#else
// Dummy types when CUDA is not available - these should never be instantiated
// but are needed for CXX bridge compilation
struct FastVGICPCuda {};
struct NDTCuda {};
#endif

// === Point Cloud Factory Functions ===
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz();
std::unique_ptr<PointCloudXYZI> create_point_cloud_xyzi();

// === Point Cloud Operations ===
size_t point_cloud_xyz_size(const PointCloudXYZ &cloud);
bool point_cloud_xyz_empty(const PointCloudXYZ &cloud);
void point_cloud_xyz_clear(PointCloudXYZ &cloud);
void point_cloud_xyz_reserve(PointCloudXYZ &cloud, size_t capacity);
void point_cloud_xyz_push_point(PointCloudXYZ &cloud, float x, float y,
                                float z);
Point3f point_cloud_xyz_get_point(const PointCloudXYZ &cloud, size_t index);
void point_cloud_xyz_set_point(PointCloudXYZ &cloud, size_t index, float x,
                               float y, float z);

size_t point_cloud_xyzi_size(const PointCloudXYZI &cloud);
bool point_cloud_xyzi_empty(const PointCloudXYZI &cloud);
void point_cloud_xyzi_clear(PointCloudXYZI &cloud);
void point_cloud_xyzi_reserve(PointCloudXYZI &cloud, size_t capacity);
void point_cloud_xyzi_push_point(PointCloudXYZI &cloud, float x, float y,
                                 float z, float intensity);
Point4f point_cloud_xyzi_get_point(const PointCloudXYZI &cloud, size_t index);
void point_cloud_xyzi_set_point(PointCloudXYZI &cloud, size_t index, float x,
                                float y, float z, float intensity);

// === GICP Factory Functions ===
std::unique_ptr<FastGICP> create_fast_gicp();
std::unique_ptr<FastVGICP> create_fast_vgicp();
std::unique_ptr<FastGICPI> create_fast_gicp_i();
std::unique_ptr<FastVGICPI> create_fast_vgicp_i();

#ifdef BUILD_VGICP_CUDA
std::unique_ptr<FastVGICPCuda> create_fast_vgicp_cuda();
#endif

// === Registration Configuration ===
void fast_gicp_set_input_source(FastGICP &gicp, const PointCloudXYZ &cloud);
void fast_gicp_set_input_target(FastGICP &gicp, const PointCloudXYZ &cloud);
void fast_gicp_set_max_iterations(FastGICP &gicp, int max_iterations);
void fast_gicp_set_transformation_epsilon(FastGICP &gicp, double eps);
void fast_gicp_set_euclidean_fitness_epsilon(FastGICP &gicp, double eps);
void fast_gicp_set_max_correspondence_distance(FastGICP &gicp, double distance);
void fast_gicp_set_num_threads(FastGICP &gicp, int num_threads);
void fast_gicp_set_correspondence_randomness(FastGICP &gicp, int k);
void fast_gicp_set_regularization_method(FastGICP &gicp, int method);
void fast_gicp_set_rotation_epsilon(FastGICP &gicp, double eps);

void fast_vgicp_set_input_source(FastVGICP &vgicp, const PointCloudXYZ &cloud);
void fast_vgicp_set_input_target(FastVGICP &vgicp, const PointCloudXYZ &cloud);
void fast_vgicp_set_max_iterations(FastVGICP &vgicp, int max_iterations);
void fast_vgicp_set_transformation_epsilon(FastVGICP &vgicp, double eps);
void fast_vgicp_set_euclidean_fitness_epsilon(FastVGICP &vgicp, double eps);
void fast_vgicp_set_max_correspondence_distance(FastVGICP &vgicp,
                                                double distance);
void fast_vgicp_set_resolution(FastVGICP &vgicp, double resolution);
void fast_vgicp_set_num_threads(FastVGICP &vgicp, int num_threads);
void fast_vgicp_set_regularization_method(FastVGICP &vgicp, int method);
void fast_vgicp_set_voxel_accumulation_mode(FastVGICP &vgicp, int mode);
void fast_vgicp_set_neighbor_search_method(FastVGICP &vgicp, int method);

#ifdef BUILD_VGICP_CUDA
void fast_vgicp_cuda_set_input_source(FastVGICPCuda &cuda_vgicp,
                                      const PointCloudXYZ &cloud);
void fast_vgicp_cuda_set_input_target(FastVGICPCuda &cuda_vgicp,
                                      const PointCloudXYZ &cloud);
void fast_vgicp_cuda_set_max_iterations(FastVGICPCuda &cuda_vgicp,
                                        int max_iterations);
void fast_vgicp_cuda_set_transformation_epsilon(FastVGICPCuda &cuda_vgicp,
                                                double eps);
void fast_vgicp_cuda_set_euclidean_fitness_epsilon(FastVGICPCuda &cuda_vgicp,
                                                   double eps);
void fast_vgicp_cuda_set_max_correspondence_distance(FastVGICPCuda &cuda_vgicp,
                                                     double distance);
void fast_vgicp_cuda_set_resolution(FastVGICPCuda &cuda_vgicp,
                                    double resolution);
void fast_vgicp_cuda_set_neighbor_search_method(FastVGICPCuda &cuda_vgicp,
                                                int method);
#endif

// === Registration Execution ===
Transform4f fast_gicp_align(FastGICP &gicp);
Transform4f fast_gicp_align_with_guess(FastGICP &gicp,
                                       const Transform4f &guess);

Transform4f fast_vgicp_align(FastVGICP &vgicp);
Transform4f fast_vgicp_align_with_guess(FastVGICP &vgicp,
                                        const Transform4f &guess);

#ifdef BUILD_VGICP_CUDA
Transform4f fast_vgicp_cuda_align(FastVGICPCuda &cuda_vgicp);
Transform4f fast_vgicp_cuda_align_with_guess(FastVGICPCuda &cuda_vgicp,
                                             const Transform4f &guess);
#endif

// === Registration Status ===
bool fast_gicp_has_converged(const FastGICP &gicp);
double fast_gicp_get_fitness_score(const FastGICP &gicp);
int fast_gicp_get_final_num_iterations(const FastGICP &gicp);

bool fast_vgicp_has_converged(const FastVGICP &vgicp);
double fast_vgicp_get_fitness_score(const FastVGICP &vgicp);
int fast_vgicp_get_final_num_iterations(const FastVGICP &vgicp);

#ifdef BUILD_VGICP_CUDA
bool fast_vgicp_cuda_has_converged(const FastVGICPCuda &cuda_vgicp);
double fast_vgicp_cuda_get_fitness_score(const FastVGICPCuda &cuda_vgicp);
int fast_vgicp_cuda_get_final_num_iterations(const FastVGICPCuda &cuda_vgicp);
#endif

// === NDTCuda Operations ===
#ifdef BUILD_VGICP_CUDA
std::unique_ptr<NDTCuda> create_ndt_cuda();
void ndt_cuda_set_input_source(NDTCuda &ndt_cuda, const PointCloudXYZ &cloud);
void ndt_cuda_set_input_target(NDTCuda &ndt_cuda, const PointCloudXYZ &cloud);
void ndt_cuda_set_max_iterations(NDTCuda &ndt_cuda, int max_iterations);
void ndt_cuda_set_transformation_epsilon(NDTCuda &ndt_cuda, double eps);
void ndt_cuda_set_euclidean_fitness_epsilon(NDTCuda &ndt_cuda, double eps);
void ndt_cuda_set_max_correspondence_distance(NDTCuda &ndt_cuda,
                                              double distance);
void ndt_cuda_set_resolution(NDTCuda &ndt_cuda, double resolution);
void ndt_cuda_set_distance_mode(NDTCuda &ndt_cuda, int mode);
void ndt_cuda_set_neighbor_search_method(NDTCuda &ndt_cuda, int method,
                                         double radius);

Transform4f ndt_cuda_align(NDTCuda &ndt_cuda);
Transform4f ndt_cuda_align_with_guess(NDTCuda &ndt_cuda,
                                      const Transform4f &guess);

bool ndt_cuda_has_converged(const NDTCuda &ndt_cuda);
double ndt_cuda_get_fitness_score(const NDTCuda &ndt_cuda);
int ndt_cuda_get_final_num_iterations(const NDTCuda &ndt_cuda);

// Hessian and cost evaluation (for covariance estimation)
Hessian6x6 ndt_cuda_get_hessian(const NDTCuda &ndt_cuda);
double ndt_cuda_evaluate_cost(const NDTCuda &ndt_cuda, const Transform4f &pose);
#else
// Stub declarations for compatibility
std::unique_ptr<NDTCuda> create_ndt_cuda();
void ndt_cuda_set_input_source(NDTCuda &ndt_cuda, const PointCloudXYZ &cloud);
void ndt_cuda_set_input_target(NDTCuda &ndt_cuda, const PointCloudXYZ &cloud);
void ndt_cuda_set_max_iterations(NDTCuda &ndt_cuda, int max_iterations);
void ndt_cuda_set_transformation_epsilon(NDTCuda &ndt_cuda, double eps);
void ndt_cuda_set_euclidean_fitness_epsilon(NDTCuda &ndt_cuda, double eps);
void ndt_cuda_set_max_correspondence_distance(NDTCuda &ndt_cuda,
                                              double distance);
void ndt_cuda_set_resolution(NDTCuda &ndt_cuda, double resolution);
void ndt_cuda_set_distance_mode(NDTCuda &ndt_cuda, int mode);
void ndt_cuda_set_neighbor_search_method(NDTCuda &ndt_cuda, int method,
                                         double radius);

Transform4f ndt_cuda_align(NDTCuda &ndt_cuda);
Transform4f ndt_cuda_align_with_guess(NDTCuda &ndt_cuda,
                                      const Transform4f &guess);

bool ndt_cuda_has_converged(const NDTCuda &ndt_cuda);
double ndt_cuda_get_fitness_score(const NDTCuda &ndt_cuda);
int ndt_cuda_get_final_num_iterations(const NDTCuda &ndt_cuda);

// Hessian and cost evaluation (stubs for non-CUDA build)
Hessian6x6 ndt_cuda_get_hessian(const NDTCuda &ndt_cuda);
double ndt_cuda_evaluate_cost(const NDTCuda &ndt_cuda, const Transform4f &pose);
#endif

// === Transform Utilities ===
Transform4f transform_identity();
Transform4f transform_from_translation(float x, float y, float z);
Transform4f transform_multiply(const Transform4f &a, const Transform4f &b);
Transform4f transform_inverse(const Transform4f &t);
