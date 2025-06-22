#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef FAST_GICP_ENABLE_CUDA
#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>
#include <fast_gicp/cuda/ndt_cuda.cuh>
#endif

#include "rust/cxx.h"

// Point cloud type aliases in global namespace for cxx
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

// Registration algorithm type aliases
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
using FastVGICP = fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;


#ifdef FAST_GICP_ENABLE_CUDA
using FastVGICPCuda = fast_gicp::FastVGICPCuda;
using NDTCuda = fast_gicp::NDTCuda;
#endif

// Note: Point3f, Point4f, and Transform3f structs are defined by cxx

// Point cloud factory functions in global namespace for cxx
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz();
std::unique_ptr<PointCloudXYZI> create_point_cloud_xyzi();

// Point cloud basic operations
size_t point_cloud_xyz_size(const PointCloudXYZ& cloud);
size_t point_cloud_xyzi_size(const PointCloudXYZI& cloud);

bool point_cloud_xyz_empty(const PointCloudXYZ& cloud);
bool point_cloud_xyzi_empty(const PointCloudXYZI& cloud);

void point_cloud_xyz_clear(PointCloudXYZ& cloud);
void point_cloud_xyzi_clear(PointCloudXYZI& cloud);

void point_cloud_xyz_reserve(PointCloudXYZ& cloud, size_t size);
void point_cloud_xyzi_reserve(PointCloudXYZI& cloud, size_t size);

void point_cloud_xyz_push_back(PointCloudXYZ& cloud, const Point3f& point);
void point_cloud_xyzi_push_back(PointCloudXYZI& cloud, const Point4f& point);

Point3f point_cloud_xyz_get_point(const PointCloudXYZ& cloud, size_t index);
Point4f point_cloud_xyzi_get_point(const PointCloudXYZI& cloud, size_t index);

// Batch operations
std::unique_ptr<PointCloudXYZ> point_cloud_xyz_from_points(rust::Slice<const Point3f> points);
std::unique_ptr<PointCloudXYZI> point_cloud_xyzi_from_points(rust::Slice<const Point4f> points);

rust::Vec<Point3f> point_cloud_xyz_to_points(const PointCloudXYZ& cloud);
rust::Vec<Point4f> point_cloud_xyzi_to_points(const PointCloudXYZI& cloud);

// Registration algorithm factory functions
std::unique_ptr<FastGICP> create_fast_gicp();
std::unique_ptr<FastVGICP> create_fast_vgicp();

#ifdef FAST_GICP_ENABLE_CUDA
std::unique_ptr<FastVGICPCuda> create_fast_vgicp_cuda();
std::unique_ptr<NDTCuda> create_ndt_cuda();
#endif

// FastGICP functions
void fast_gicp_set_input_source(FastGICP& gicp, const PointCloudXYZ& cloud);
void fast_gicp_set_input_target(FastGICP& gicp, const PointCloudXYZ& cloud);
void fast_gicp_set_max_iterations(FastGICP& gicp, int max_iterations);
void fast_gicp_set_transformation_epsilon(FastGICP& gicp, double epsilon);
void fast_gicp_set_euclidean_fitness_epsilon(FastGICP& gicp, double epsilon);
void fast_gicp_set_max_correspondence_distance(FastGICP& gicp, double distance);
// TODO: Fix enum function signatures
// void fast_gicp_set_lsq_nonlinear_optimization_algorithm(FastGICP& gicp, uint32_t algorithm);
// void fast_gicp_set_neighbor_search_method(FastGICP& gicp, uint32_t method);
void fast_gicp_set_neighbor_search_radius(FastGICP& gicp, double radius);
// void fast_gicp_set_regularization_method(FastGICP& gicp, uint32_t method);
void fast_gicp_set_num_threads(FastGICP& gicp, int num_threads);
bool fast_gicp_align(FastGICP& gicp, PointCloudXYZ& output, const Transform3f& guess);
Transform3f fast_gicp_get_final_transformation(const FastGICP& gicp);
double fast_gicp_get_fitness_score(const FastGICP& gicp);
bool fast_gicp_has_converged(const FastGICP& gicp);

// FastVGICP functions
void fast_vgicp_set_input_source(FastVGICP& vgicp, const PointCloudXYZ& cloud);
void fast_vgicp_set_input_target(FastVGICP& vgicp, const PointCloudXYZ& cloud);
void fast_vgicp_set_max_iterations(FastVGICP& vgicp, int max_iterations);
void fast_vgicp_set_transformation_epsilon(FastVGICP& vgicp, double epsilon);
void fast_vgicp_set_euclidean_fitness_epsilon(FastVGICP& vgicp, double epsilon);
void fast_vgicp_set_max_correspondence_distance(FastVGICP& vgicp, double distance);
void fast_vgicp_set_resolution(FastVGICP& vgicp, double resolution);
// TODO: Fix enum function signatures
// void fast_vgicp_set_neighbor_search_method(FastVGICP& vgicp, uint32_t method);
void fast_vgicp_set_neighbor_search_radius(FastVGICP& vgicp, double radius);
// void fast_vgicp_set_regularization_method(FastVGICP& vgicp, uint32_t method);
void fast_vgicp_set_num_threads(FastVGICP& vgicp, int num_threads);
bool fast_vgicp_align(FastVGICP& vgicp, PointCloudXYZ& output, const Transform3f& guess);
Transform3f fast_vgicp_get_final_transformation(const FastVGICP& vgicp);
double fast_vgicp_get_fitness_score(const FastVGICP& vgicp);
bool fast_vgicp_has_converged(const FastVGICP& vgicp);

// Helper functions for converting between Eigen and our Transform3f
Eigen::Isometry3f transform_from_rust(const Transform3f& transform);
Transform3f transform_to_rust(const Eigen::Isometry3f& transform);