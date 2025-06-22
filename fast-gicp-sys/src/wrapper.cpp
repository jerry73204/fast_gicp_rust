#include "fast_gicp_wrapper.h"
#include <iostream>

// Helper functions for converting between PCL and our point types
pcl::PointXYZ point_from_rust(const Point3f& point) {
    pcl::PointXYZ pcl_point;
    pcl_point.x = point.x;
    pcl_point.y = point.y;
    pcl_point.z = point.z;
    return pcl_point;
}

Point3f point_to_rust(const pcl::PointXYZ& point) {
    return Point3f{point.x, point.y, point.z};
}

pcl::PointXYZI point_from_rust(const Point4f& point) {
    pcl::PointXYZI pcl_point;
    pcl_point.x = point.x;
    pcl_point.y = point.y;
    pcl_point.z = point.z;
    pcl_point.intensity = point.intensity;
    return pcl_point;
}

Point4f point_to_rust(const pcl::PointXYZI& point) {
    return Point4f{point.x, point.y, point.z, point.intensity};
}

// Transform conversion helpers
Eigen::Isometry3f transform_from_rust(const Transform3f& transform) {
    Eigen::Matrix4f matrix;
    for (int i = 0; i < 16; i++) {
        matrix(i / 4, i % 4) = transform.matrix[i];
    }
    return Eigen::Isometry3f(matrix);
}

Transform3f transform_to_rust(const Eigen::Isometry3f& transform) {
    Transform3f result;
    const Eigen::Matrix4f& matrix = transform.matrix();
    for (int i = 0; i < 16; i++) {
        result.matrix[i] = matrix(i / 4, i % 4);
    }
    return result;
}

// Point cloud factory functions
std::unique_ptr<PointCloudXYZ> create_point_cloud_xyz() {
    return std::make_unique<PointCloudXYZ>();
}

std::unique_ptr<PointCloudXYZI> create_point_cloud_xyzi() {
    return std::make_unique<PointCloudXYZI>();
}

// Point cloud basic operations
size_t point_cloud_xyz_size(const PointCloudXYZ& cloud) {
    return cloud.size();
}

size_t point_cloud_xyzi_size(const PointCloudXYZI& cloud) {
    return cloud.size();
}

bool point_cloud_xyz_empty(const PointCloudXYZ& cloud) {
    return cloud.empty();
}

bool point_cloud_xyzi_empty(const PointCloudXYZI& cloud) {
    return cloud.empty();
}

void point_cloud_xyz_clear(PointCloudXYZ& cloud) {
    cloud.clear();
}

void point_cloud_xyzi_clear(PointCloudXYZI& cloud) {
    cloud.clear();
}

void point_cloud_xyz_reserve(PointCloudXYZ& cloud, size_t size) {
    cloud.reserve(size);
}

void point_cloud_xyzi_reserve(PointCloudXYZI& cloud, size_t size) {
    cloud.reserve(size);
}

void point_cloud_xyz_push_back(PointCloudXYZ& cloud, const Point3f& point) {
    cloud.push_back(point_from_rust(point));
}

void point_cloud_xyzi_push_back(PointCloudXYZI& cloud, const Point4f& point) {
    cloud.push_back(point_from_rust(point));
}

Point3f point_cloud_xyz_get_point(const PointCloudXYZ& cloud, size_t index) {
    if (index >= cloud.size()) {
        throw std::out_of_range("Point index out of range");
    }
    return point_to_rust(cloud[index]);
}

Point4f point_cloud_xyzi_get_point(const PointCloudXYZI& cloud, size_t index) {
    if (index >= cloud.size()) {
        throw std::out_of_range("Point index out of range");
    }
    return point_to_rust(cloud[index]);
}

// Batch operations
std::unique_ptr<PointCloudXYZ> point_cloud_xyz_from_points(rust::Slice<const Point3f> points) {
    auto cloud = std::make_unique<PointCloudXYZ>();
    cloud->reserve(points.size());
    for (const auto& point : points) {
        cloud->push_back(point_from_rust(point));
    }
    return cloud;
}

std::unique_ptr<PointCloudXYZI> point_cloud_xyzi_from_points(rust::Slice<const Point4f> points) {
    auto cloud = std::make_unique<PointCloudXYZI>();
    cloud->reserve(points.size());
    for (const auto& point : points) {
        cloud->push_back(point_from_rust(point));
    }
    return cloud;
}

rust::Vec<Point3f> point_cloud_xyz_to_points(const PointCloudXYZ& cloud) {
    rust::Vec<Point3f> points;
    points.reserve(cloud.size());
    for (const auto& point : cloud) {
        points.push_back(point_to_rust(point));
    }
    return points;
}

rust::Vec<Point4f> point_cloud_xyzi_to_points(const PointCloudXYZI& cloud) {
    rust::Vec<Point4f> points;
    points.reserve(cloud.size());
    for (const auto& point : cloud) {
        points.push_back(point_to_rust(point));
    }
    return points;
}

// Registration algorithm factory functions
std::unique_ptr<FastGICP> create_fast_gicp() {
    return std::make_unique<FastGICP>();
}

std::unique_ptr<FastVGICP> create_fast_vgicp() {
    return std::make_unique<FastVGICP>();
}

#ifdef FAST_GICP_ENABLE_CUDA
std::unique_ptr<FastVGICPCuda> create_fast_vgicp_cuda() {
    return std::make_unique<FastVGICPCuda>();
}

std::unique_ptr<NDTCuda> create_ndt_cuda() {
    return std::make_unique<NDTCuda>();
}
#endif

// FastGICP functions
void fast_gicp_set_input_source(FastGICP& gicp, const PointCloudXYZ& cloud) {
    gicp.setInputSource(cloud.makeShared());
}

void fast_gicp_set_input_target(FastGICP& gicp, const PointCloudXYZ& cloud) {
    gicp.setInputTarget(cloud.makeShared());
}

void fast_gicp_set_max_iterations(FastGICP& gicp, int max_iterations) {
    gicp.setMaximumIterations(max_iterations);
}

void fast_gicp_set_transformation_epsilon(FastGICP& gicp, double epsilon) {
    gicp.setTransformationEpsilon(epsilon);
}

void fast_gicp_set_euclidean_fitness_epsilon(FastGICP& gicp, double epsilon) {
    gicp.setEuclideanFitnessEpsilon(epsilon);
}

void fast_gicp_set_max_correspondence_distance(FastGICP& gicp, double distance) {
    gicp.setMaxCorrespondenceDistance(distance);
}

// TODO: Fix enum function signatures
// void fast_gicp_set_lsq_nonlinear_optimization_algorithm(FastGICP& gicp, uint32_t algorithm) {
//     gicp.setLSQNonlinearOptimizationAlgorithm(
//         static_cast<fast_gicp::LSQNonlinearOptimizationAlgorithm>(algorithm));
// }

// void fast_gicp_set_neighbor_search_method(FastGICP& gicp, uint32_t method) {
//     gicp.setNeighborSearchMethod(static_cast<fast_gicp::NeighborSearchMethod>(method));
// }

void fast_gicp_set_neighbor_search_radius(FastGICP& gicp, double radius) {
    gicp.setNeighborSearchRadius(radius);
}

// void fast_gicp_set_regularization_method(FastGICP& gicp, uint32_t method) {
//     gicp.setRegularizationMethod(static_cast<fast_gicp::RegularizationMethod>(method));
// }

void fast_gicp_set_num_threads(FastGICP& gicp, int num_threads) {
    gicp.setNumThreads(num_threads);
}

bool fast_gicp_align(FastGICP& gicp, PointCloudXYZ& output, const Transform3f& guess) {
    Eigen::Isometry3f initial_guess = transform_from_rust(guess);
    gicp.align(output, initial_guess.matrix());
    return gicp.hasConverged();
}

Transform3f fast_gicp_get_final_transformation(const FastGICP& gicp) {
    Eigen::Isometry3f transform(gicp.getFinalTransformation());
    return transform_to_rust(transform);
}

double fast_gicp_get_fitness_score(const FastGICP& gicp) {
    return gicp.getFitnessScore();
}

bool fast_gicp_has_converged(const FastGICP& gicp) {
    return gicp.hasConverged();
}

// FastVGICP functions
void fast_vgicp_set_input_source(FastVGICP& vgicp, const PointCloudXYZ& cloud) {
    vgicp.setInputSource(cloud.makeShared());
}

void fast_vgicp_set_input_target(FastVGICP& vgicp, const PointCloudXYZ& cloud) {
    vgicp.setInputTarget(cloud.makeShared());
}

void fast_vgicp_set_max_iterations(FastVGICP& vgicp, int max_iterations) {
    vgicp.setMaximumIterations(max_iterations);
}

void fast_vgicp_set_transformation_epsilon(FastVGICP& vgicp, double epsilon) {
    vgicp.setTransformationEpsilon(epsilon);
}

void fast_vgicp_set_euclidean_fitness_epsilon(FastVGICP& vgicp, double epsilon) {
    vgicp.setEuclideanFitnessEpsilon(epsilon);
}

void fast_vgicp_set_max_correspondence_distance(FastVGICP& vgicp, double distance) {
    vgicp.setMaxCorrespondenceDistance(distance);
}

void fast_vgicp_set_resolution(FastVGICP& vgicp, double resolution) {
    vgicp.setResolution(resolution);
}

// TODO: Fix enum function signatures
// void fast_vgicp_set_neighbor_search_method(FastVGICP& vgicp, uint32_t method) {
//     vgicp.setNeighborSearchMethod(static_cast<fast_gicp::NeighborSearchMethod>(method));
// }

void fast_vgicp_set_neighbor_search_radius(FastVGICP& vgicp, double radius) {
    vgicp.setNeighborSearchRadius(radius);
}

// void fast_vgicp_set_regularization_method(FastVGICP& vgicp, uint32_t method) {
//     vgicp.setRegularizationMethod(static_cast<fast_gicp::RegularizationMethod>(method));
// }

void fast_vgicp_set_num_threads(FastVGICP& vgicp, int num_threads) {
    vgicp.setNumThreads(num_threads);
}

bool fast_vgicp_align(FastVGICP& vgicp, PointCloudXYZ& output, const Transform3f& guess) {
    Eigen::Isometry3f initial_guess = transform_from_rust(guess);
    vgicp.align(output, initial_guess.matrix());
    return vgicp.hasConverged();
}

Transform3f fast_vgicp_get_final_transformation(const FastVGICP& vgicp) {
    Eigen::Isometry3f transform(vgicp.getFinalTransformation());
    return transform_to_rust(transform);
}

double fast_vgicp_get_fitness_score(const FastVGICP& vgicp) {
    return vgicp.getFitnessScore();
}

bool fast_vgicp_has_converged(const FastVGICP& vgicp) {
    return vgicp.hasConverged();
}