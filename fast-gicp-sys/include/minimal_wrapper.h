#pragma once

#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <flann/util/params.h>
#include <fast_gicp/gicp/fast_gicp.hpp>

// Simple type aliases
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
// Use the exact template instantiation from the library
using KdTreeType = pcl::search::KdTree<pcl::PointXYZ, pcl::KdTreeFLANN<pcl::PointXYZ, flann::L2_Simple<float>>>;
using FastGICP = fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ, KdTreeType, KdTreeType>;

// Simple factory functions
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz();
std::unique_ptr<FastGICP> create_fast_gicp();

// Simple info functions
size_t point_cloud_xyz_size(const PointCloudXYZ& cloud);
bool point_cloud_xyz_empty(const PointCloudXYZ& cloud);

// Simple registration functions
void fast_gicp_set_input_source(FastGICP& gicp, const PointCloudXYZ& cloud);
void fast_gicp_set_input_target(FastGICP& gicp, const PointCloudXYZ& cloud);
void fast_gicp_set_max_iterations(FastGICP& gicp, int max_iterations);
bool fast_gicp_has_converged(const FastGICP& gicp);