#include "minimal_wrapper.h"

// Explicit template instantiation to ensure the right template is used
template class fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ, KdTreeType, KdTreeType>;

// Simple factory functions
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz() {
    return std::make_unique<PointCloudXYZ>();
}

std::unique_ptr<FastGICP> create_fast_gicp() {
    return std::make_unique<FastGICP>();
}

// Simple info functions
size_t point_cloud_xyz_size(const PointCloudXYZ& cloud) {
    return cloud.size();
}

bool point_cloud_xyz_empty(const PointCloudXYZ& cloud) {
    return cloud.empty();
}

// Simple registration functions
void fast_gicp_set_input_source(FastGICP& gicp, const PointCloudXYZ& cloud) {
    gicp.setInputSource(cloud.makeShared());
}

void fast_gicp_set_input_target(FastGICP& gicp, const PointCloudXYZ& cloud) {
    gicp.setInputTarget(cloud.makeShared());
}

void fast_gicp_set_max_iterations(FastGICP& gicp, int max_iterations) {
    gicp.setMaximumIterations(max_iterations);
}

bool fast_gicp_has_converged(const FastGICP& gicp) {
    return gicp.hasConverged();
}