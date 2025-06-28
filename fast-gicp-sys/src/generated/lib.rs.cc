#include "wrapper.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace rust {
inline namespace cxxbridge1 {
// #include "rust/cxx.h"

#ifndef CXXBRIDGE1_IS_COMPLETE
#define CXXBRIDGE1_IS_COMPLETE
namespace detail {
namespace {
template <typename T, typename = std::size_t>
struct is_complete : std::false_type {};
template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};
} // namespace
} // namespace detail
#endif // CXXBRIDGE1_IS_COMPLETE

namespace {
template <bool> struct deleter_if {
  template <typename T> void operator()(T *) {}
};

template <> struct deleter_if<true> {
  template <typename T> void operator()(T *ptr) { ptr->~T(); }
};
} // namespace
} // namespace cxxbridge1
} // namespace rust

#if __cplusplus >= 201402L
#define CXX_DEFAULT_VALUE(value) = value
#else
#define CXX_DEFAULT_VALUE(value)
#endif

struct Transform4f;
struct Point3f;
struct Point4f;
using PointCloudXYZ = ::PointCloudXYZ;
using PointCloudXYZI = ::PointCloudXYZI;
using FastGICP = ::FastGICP;
using FastVGICP = ::FastVGICP;
using FastGICPI = ::FastGICPI;
using FastVGICPI = ::FastVGICPI;
using FastVGICPCuda = ::FastVGICPCuda;
using NDTCuda = ::NDTCuda;

#ifndef CXXBRIDGE1_STRUCT_Transform4f
#define CXXBRIDGE1_STRUCT_Transform4f
struct Transform4f final {
  ::std::array<float, 16> data;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_Transform4f

#ifndef CXXBRIDGE1_STRUCT_Point3f
#define CXXBRIDGE1_STRUCT_Point3f
struct Point3f final {
  float x CXX_DEFAULT_VALUE(0);
  float y CXX_DEFAULT_VALUE(0);
  float z CXX_DEFAULT_VALUE(0);

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_Point3f

#ifndef CXXBRIDGE1_STRUCT_Point4f
#define CXXBRIDGE1_STRUCT_Point4f
struct Point4f final {
  float x CXX_DEFAULT_VALUE(0);
  float y CXX_DEFAULT_VALUE(0);
  float z CXX_DEFAULT_VALUE(0);
  float intensity CXX_DEFAULT_VALUE(0);

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_Point4f

extern "C" {
::PointCloudXYZ *cxxbridge1$create_point_cloud_xyz() noexcept {
  ::std::unique_ptr<::PointCloudXYZ> (*create_point_cloud_xyz$)() = ::create_point_cloud_xyz;
  return create_point_cloud_xyz$().release();
}

::PointCloudXYZI *cxxbridge1$create_point_cloud_xyzi() noexcept {
  ::std::unique_ptr<::PointCloudXYZI> (*create_point_cloud_xyzi$)() = ::create_point_cloud_xyzi;
  return create_point_cloud_xyzi$().release();
}

::std::size_t cxxbridge1$point_cloud_xyz_size(::PointCloudXYZ const &cloud) noexcept {
  ::std::size_t (*point_cloud_xyz_size$)(::PointCloudXYZ const &) = ::point_cloud_xyz_size;
  return point_cloud_xyz_size$(cloud);
}

bool cxxbridge1$point_cloud_xyz_empty(::PointCloudXYZ const &cloud) noexcept {
  bool (*point_cloud_xyz_empty$)(::PointCloudXYZ const &) = ::point_cloud_xyz_empty;
  return point_cloud_xyz_empty$(cloud);
}

void cxxbridge1$point_cloud_xyz_clear(::PointCloudXYZ &cloud) noexcept {
  void (*point_cloud_xyz_clear$)(::PointCloudXYZ &) = ::point_cloud_xyz_clear;
  point_cloud_xyz_clear$(cloud);
}

void cxxbridge1$point_cloud_xyz_reserve(::PointCloudXYZ &cloud, ::std::size_t capacity) noexcept {
  void (*point_cloud_xyz_reserve$)(::PointCloudXYZ &, ::std::size_t) = ::point_cloud_xyz_reserve;
  point_cloud_xyz_reserve$(cloud, capacity);
}

void cxxbridge1$point_cloud_xyz_push_point(::PointCloudXYZ &cloud, float x, float y, float z) noexcept {
  void (*point_cloud_xyz_push_point$)(::PointCloudXYZ &, float, float, float) = ::point_cloud_xyz_push_point;
  point_cloud_xyz_push_point$(cloud, x, y, z);
}

void cxxbridge1$point_cloud_xyz_get_point(::PointCloudXYZ const &cloud, ::std::size_t index, ::Point3f *return$) noexcept {
  ::Point3f (*point_cloud_xyz_get_point$)(::PointCloudXYZ const &, ::std::size_t) = ::point_cloud_xyz_get_point;
  new (return$) ::Point3f(point_cloud_xyz_get_point$(cloud, index));
}

void cxxbridge1$point_cloud_xyz_set_point(::PointCloudXYZ &cloud, ::std::size_t index, float x, float y, float z) noexcept {
  void (*point_cloud_xyz_set_point$)(::PointCloudXYZ &, ::std::size_t, float, float, float) = ::point_cloud_xyz_set_point;
  point_cloud_xyz_set_point$(cloud, index, x, y, z);
}

::std::size_t cxxbridge1$point_cloud_xyzi_size(::PointCloudXYZI const &cloud) noexcept {
  ::std::size_t (*point_cloud_xyzi_size$)(::PointCloudXYZI const &) = ::point_cloud_xyzi_size;
  return point_cloud_xyzi_size$(cloud);
}

bool cxxbridge1$point_cloud_xyzi_empty(::PointCloudXYZI const &cloud) noexcept {
  bool (*point_cloud_xyzi_empty$)(::PointCloudXYZI const &) = ::point_cloud_xyzi_empty;
  return point_cloud_xyzi_empty$(cloud);
}

void cxxbridge1$point_cloud_xyzi_clear(::PointCloudXYZI &cloud) noexcept {
  void (*point_cloud_xyzi_clear$)(::PointCloudXYZI &) = ::point_cloud_xyzi_clear;
  point_cloud_xyzi_clear$(cloud);
}

void cxxbridge1$point_cloud_xyzi_reserve(::PointCloudXYZI &cloud, ::std::size_t capacity) noexcept {
  void (*point_cloud_xyzi_reserve$)(::PointCloudXYZI &, ::std::size_t) = ::point_cloud_xyzi_reserve;
  point_cloud_xyzi_reserve$(cloud, capacity);
}

void cxxbridge1$point_cloud_xyzi_push_point(::PointCloudXYZI &cloud, float x, float y, float z, float intensity) noexcept {
  void (*point_cloud_xyzi_push_point$)(::PointCloudXYZI &, float, float, float, float) = ::point_cloud_xyzi_push_point;
  point_cloud_xyzi_push_point$(cloud, x, y, z, intensity);
}

void cxxbridge1$point_cloud_xyzi_get_point(::PointCloudXYZI const &cloud, ::std::size_t index, ::Point4f *return$) noexcept {
  ::Point4f (*point_cloud_xyzi_get_point$)(::PointCloudXYZI const &, ::std::size_t) = ::point_cloud_xyzi_get_point;
  new (return$) ::Point4f(point_cloud_xyzi_get_point$(cloud, index));
}

void cxxbridge1$point_cloud_xyzi_set_point(::PointCloudXYZI &cloud, ::std::size_t index, float x, float y, float z, float intensity) noexcept {
  void (*point_cloud_xyzi_set_point$)(::PointCloudXYZI &, ::std::size_t, float, float, float, float) = ::point_cloud_xyzi_set_point;
  point_cloud_xyzi_set_point$(cloud, index, x, y, z, intensity);
}

::FastGICP *cxxbridge1$create_fast_gicp() noexcept {
  ::std::unique_ptr<::FastGICP> (*create_fast_gicp$)() = ::create_fast_gicp;
  return create_fast_gicp$().release();
}

::FastVGICP *cxxbridge1$create_fast_vgicp() noexcept {
  ::std::unique_ptr<::FastVGICP> (*create_fast_vgicp$)() = ::create_fast_vgicp;
  return create_fast_vgicp$().release();
}

::FastGICPI *cxxbridge1$create_fast_gicp_i() noexcept {
  ::std::unique_ptr<::FastGICPI> (*create_fast_gicp_i$)() = ::create_fast_gicp_i;
  return create_fast_gicp_i$().release();
}

::FastVGICPI *cxxbridge1$create_fast_vgicp_i() noexcept {
  ::std::unique_ptr<::FastVGICPI> (*create_fast_vgicp_i$)() = ::create_fast_vgicp_i;
  return create_fast_vgicp_i$().release();
}

void cxxbridge1$fast_gicp_set_input_source(::FastGICP &gicp, ::PointCloudXYZ const &cloud) noexcept {
  void (*fast_gicp_set_input_source$)(::FastGICP &, ::PointCloudXYZ const &) = ::fast_gicp_set_input_source;
  fast_gicp_set_input_source$(gicp, cloud);
}

void cxxbridge1$fast_gicp_set_input_target(::FastGICP &gicp, ::PointCloudXYZ const &cloud) noexcept {
  void (*fast_gicp_set_input_target$)(::FastGICP &, ::PointCloudXYZ const &) = ::fast_gicp_set_input_target;
  fast_gicp_set_input_target$(gicp, cloud);
}

void cxxbridge1$fast_gicp_set_max_iterations(::FastGICP &gicp, ::std::int32_t max_iterations) noexcept {
  void (*fast_gicp_set_max_iterations$)(::FastGICP &, ::std::int32_t) = ::fast_gicp_set_max_iterations;
  fast_gicp_set_max_iterations$(gicp, max_iterations);
}

void cxxbridge1$fast_gicp_set_transformation_epsilon(::FastGICP &gicp, double eps) noexcept {
  void (*fast_gicp_set_transformation_epsilon$)(::FastGICP &, double) = ::fast_gicp_set_transformation_epsilon;
  fast_gicp_set_transformation_epsilon$(gicp, eps);
}

void cxxbridge1$fast_gicp_set_euclidean_fitness_epsilon(::FastGICP &gicp, double eps) noexcept {
  void (*fast_gicp_set_euclidean_fitness_epsilon$)(::FastGICP &, double) = ::fast_gicp_set_euclidean_fitness_epsilon;
  fast_gicp_set_euclidean_fitness_epsilon$(gicp, eps);
}

void cxxbridge1$fast_gicp_set_max_correspondence_distance(::FastGICP &gicp, double distance) noexcept {
  void (*fast_gicp_set_max_correspondence_distance$)(::FastGICP &, double) = ::fast_gicp_set_max_correspondence_distance;
  fast_gicp_set_max_correspondence_distance$(gicp, distance);
}

void cxxbridge1$fast_gicp_set_num_threads(::FastGICP &gicp, ::std::int32_t num_threads) noexcept {
  void (*fast_gicp_set_num_threads$)(::FastGICP &, ::std::int32_t) = ::fast_gicp_set_num_threads;
  fast_gicp_set_num_threads$(gicp, num_threads);
}

void cxxbridge1$fast_gicp_set_correspondence_randomness(::FastGICP &gicp, ::std::int32_t k) noexcept {
  void (*fast_gicp_set_correspondence_randomness$)(::FastGICP &, ::std::int32_t) = ::fast_gicp_set_correspondence_randomness;
  fast_gicp_set_correspondence_randomness$(gicp, k);
}

void cxxbridge1$fast_gicp_set_regularization_method(::FastGICP &gicp, ::std::int32_t method) noexcept {
  void (*fast_gicp_set_regularization_method$)(::FastGICP &, ::std::int32_t) = ::fast_gicp_set_regularization_method;
  fast_gicp_set_regularization_method$(gicp, method);
}

void cxxbridge1$fast_gicp_set_rotation_epsilon(::FastGICP &gicp, double eps) noexcept {
  void (*fast_gicp_set_rotation_epsilon$)(::FastGICP &, double) = ::fast_gicp_set_rotation_epsilon;
  fast_gicp_set_rotation_epsilon$(gicp, eps);
}

void cxxbridge1$fast_vgicp_set_input_source(::FastVGICP &vgicp, ::PointCloudXYZ const &cloud) noexcept {
  void (*fast_vgicp_set_input_source$)(::FastVGICP &, ::PointCloudXYZ const &) = ::fast_vgicp_set_input_source;
  fast_vgicp_set_input_source$(vgicp, cloud);
}

void cxxbridge1$fast_vgicp_set_input_target(::FastVGICP &vgicp, ::PointCloudXYZ const &cloud) noexcept {
  void (*fast_vgicp_set_input_target$)(::FastVGICP &, ::PointCloudXYZ const &) = ::fast_vgicp_set_input_target;
  fast_vgicp_set_input_target$(vgicp, cloud);
}

void cxxbridge1$fast_vgicp_set_max_iterations(::FastVGICP &vgicp, ::std::int32_t max_iterations) noexcept {
  void (*fast_vgicp_set_max_iterations$)(::FastVGICP &, ::std::int32_t) = ::fast_vgicp_set_max_iterations;
  fast_vgicp_set_max_iterations$(vgicp, max_iterations);
}

void cxxbridge1$fast_vgicp_set_transformation_epsilon(::FastVGICP &vgicp, double eps) noexcept {
  void (*fast_vgicp_set_transformation_epsilon$)(::FastVGICP &, double) = ::fast_vgicp_set_transformation_epsilon;
  fast_vgicp_set_transformation_epsilon$(vgicp, eps);
}

void cxxbridge1$fast_vgicp_set_euclidean_fitness_epsilon(::FastVGICP &vgicp, double eps) noexcept {
  void (*fast_vgicp_set_euclidean_fitness_epsilon$)(::FastVGICP &, double) = ::fast_vgicp_set_euclidean_fitness_epsilon;
  fast_vgicp_set_euclidean_fitness_epsilon$(vgicp, eps);
}

void cxxbridge1$fast_vgicp_set_max_correspondence_distance(::FastVGICP &vgicp, double distance) noexcept {
  void (*fast_vgicp_set_max_correspondence_distance$)(::FastVGICP &, double) = ::fast_vgicp_set_max_correspondence_distance;
  fast_vgicp_set_max_correspondence_distance$(vgicp, distance);
}

void cxxbridge1$fast_vgicp_set_resolution(::FastVGICP &vgicp, double resolution) noexcept {
  void (*fast_vgicp_set_resolution$)(::FastVGICP &, double) = ::fast_vgicp_set_resolution;
  fast_vgicp_set_resolution$(vgicp, resolution);
}

void cxxbridge1$fast_vgicp_set_num_threads(::FastVGICP &vgicp, ::std::int32_t num_threads) noexcept {
  void (*fast_vgicp_set_num_threads$)(::FastVGICP &, ::std::int32_t) = ::fast_vgicp_set_num_threads;
  fast_vgicp_set_num_threads$(vgicp, num_threads);
}

void cxxbridge1$fast_vgicp_set_regularization_method(::FastVGICP &vgicp, ::std::int32_t method) noexcept {
  void (*fast_vgicp_set_regularization_method$)(::FastVGICP &, ::std::int32_t) = ::fast_vgicp_set_regularization_method;
  fast_vgicp_set_regularization_method$(vgicp, method);
}

void cxxbridge1$fast_vgicp_set_voxel_accumulation_mode(::FastVGICP &vgicp, ::std::int32_t mode) noexcept {
  void (*fast_vgicp_set_voxel_accumulation_mode$)(::FastVGICP &, ::std::int32_t) = ::fast_vgicp_set_voxel_accumulation_mode;
  fast_vgicp_set_voxel_accumulation_mode$(vgicp, mode);
}

void cxxbridge1$fast_vgicp_set_neighbor_search_method(::FastVGICP &vgicp, ::std::int32_t method) noexcept {
  void (*fast_vgicp_set_neighbor_search_method$)(::FastVGICP &, ::std::int32_t) = ::fast_vgicp_set_neighbor_search_method;
  fast_vgicp_set_neighbor_search_method$(vgicp, method);
}

::Transform4f cxxbridge1$fast_gicp_align(::FastGICP &gicp) noexcept {
  ::Transform4f (*fast_gicp_align$)(::FastGICP &) = ::fast_gicp_align;
  return fast_gicp_align$(gicp);
}

::Transform4f cxxbridge1$fast_gicp_align_with_guess(::FastGICP &gicp, ::Transform4f const &guess) noexcept {
  ::Transform4f (*fast_gicp_align_with_guess$)(::FastGICP &, ::Transform4f const &) = ::fast_gicp_align_with_guess;
  return fast_gicp_align_with_guess$(gicp, guess);
}

::Transform4f cxxbridge1$fast_vgicp_align(::FastVGICP &vgicp) noexcept {
  ::Transform4f (*fast_vgicp_align$)(::FastVGICP &) = ::fast_vgicp_align;
  return fast_vgicp_align$(vgicp);
}

::Transform4f cxxbridge1$fast_vgicp_align_with_guess(::FastVGICP &vgicp, ::Transform4f const &guess) noexcept {
  ::Transform4f (*fast_vgicp_align_with_guess$)(::FastVGICP &, ::Transform4f const &) = ::fast_vgicp_align_with_guess;
  return fast_vgicp_align_with_guess$(vgicp, guess);
}

bool cxxbridge1$fast_gicp_has_converged(::FastGICP const &gicp) noexcept {
  bool (*fast_gicp_has_converged$)(::FastGICP const &) = ::fast_gicp_has_converged;
  return fast_gicp_has_converged$(gicp);
}

double cxxbridge1$fast_gicp_get_fitness_score(::FastGICP const &gicp) noexcept {
  double (*fast_gicp_get_fitness_score$)(::FastGICP const &) = ::fast_gicp_get_fitness_score;
  return fast_gicp_get_fitness_score$(gicp);
}

::std::int32_t cxxbridge1$fast_gicp_get_final_num_iterations(::FastGICP const &gicp) noexcept {
  ::std::int32_t (*fast_gicp_get_final_num_iterations$)(::FastGICP const &) = ::fast_gicp_get_final_num_iterations;
  return fast_gicp_get_final_num_iterations$(gicp);
}

bool cxxbridge1$fast_vgicp_has_converged(::FastVGICP const &vgicp) noexcept {
  bool (*fast_vgicp_has_converged$)(::FastVGICP const &) = ::fast_vgicp_has_converged;
  return fast_vgicp_has_converged$(vgicp);
}

double cxxbridge1$fast_vgicp_get_fitness_score(::FastVGICP const &vgicp) noexcept {
  double (*fast_vgicp_get_fitness_score$)(::FastVGICP const &) = ::fast_vgicp_get_fitness_score;
  return fast_vgicp_get_fitness_score$(vgicp);
}

::std::int32_t cxxbridge1$fast_vgicp_get_final_num_iterations(::FastVGICP const &vgicp) noexcept {
  ::std::int32_t (*fast_vgicp_get_final_num_iterations$)(::FastVGICP const &) = ::fast_vgicp_get_final_num_iterations;
  return fast_vgicp_get_final_num_iterations$(vgicp);
}

::Transform4f cxxbridge1$transform_identity() noexcept {
  ::Transform4f (*transform_identity$)() = ::transform_identity;
  return transform_identity$();
}

::Transform4f cxxbridge1$transform_from_translation(float x, float y, float z) noexcept {
  ::Transform4f (*transform_from_translation$)(float, float, float) = ::transform_from_translation;
  return transform_from_translation$(x, y, z);
}

::Transform4f cxxbridge1$transform_multiply(::Transform4f const &a, ::Transform4f const &b) noexcept {
  ::Transform4f (*transform_multiply$)(::Transform4f const &, ::Transform4f const &) = ::transform_multiply;
  return transform_multiply$(a, b);
}

::Transform4f cxxbridge1$transform_inverse(::Transform4f const &t) noexcept {
  ::Transform4f (*transform_inverse$)(::Transform4f const &) = ::transform_inverse;
  return transform_inverse$(t);
}

static_assert(::rust::detail::is_complete<::PointCloudXYZ>::value, "definition of PointCloudXYZ is required");
static_assert(sizeof(::std::unique_ptr<::PointCloudXYZ>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::PointCloudXYZ>) == alignof(void *), "");
void cxxbridge1$unique_ptr$PointCloudXYZ$null(::std::unique_ptr<::PointCloudXYZ> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::PointCloudXYZ>();
}
void cxxbridge1$unique_ptr$PointCloudXYZ$raw(::std::unique_ptr<::PointCloudXYZ> *ptr, ::PointCloudXYZ *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::PointCloudXYZ>(raw);
}
::PointCloudXYZ const *cxxbridge1$unique_ptr$PointCloudXYZ$get(::std::unique_ptr<::PointCloudXYZ> const &ptr) noexcept {
  return ptr.get();
}
::PointCloudXYZ *cxxbridge1$unique_ptr$PointCloudXYZ$release(::std::unique_ptr<::PointCloudXYZ> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$PointCloudXYZ$drop(::std::unique_ptr<::PointCloudXYZ> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::PointCloudXYZ>::value>{}(ptr);
}

static_assert(::rust::detail::is_complete<::PointCloudXYZI>::value, "definition of PointCloudXYZI is required");
static_assert(sizeof(::std::unique_ptr<::PointCloudXYZI>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::PointCloudXYZI>) == alignof(void *), "");
void cxxbridge1$unique_ptr$PointCloudXYZI$null(::std::unique_ptr<::PointCloudXYZI> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::PointCloudXYZI>();
}
void cxxbridge1$unique_ptr$PointCloudXYZI$raw(::std::unique_ptr<::PointCloudXYZI> *ptr, ::PointCloudXYZI *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::PointCloudXYZI>(raw);
}
::PointCloudXYZI const *cxxbridge1$unique_ptr$PointCloudXYZI$get(::std::unique_ptr<::PointCloudXYZI> const &ptr) noexcept {
  return ptr.get();
}
::PointCloudXYZI *cxxbridge1$unique_ptr$PointCloudXYZI$release(::std::unique_ptr<::PointCloudXYZI> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$PointCloudXYZI$drop(::std::unique_ptr<::PointCloudXYZI> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::PointCloudXYZI>::value>{}(ptr);
}

static_assert(::rust::detail::is_complete<::FastGICP>::value, "definition of FastGICP is required");
static_assert(sizeof(::std::unique_ptr<::FastGICP>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::FastGICP>) == alignof(void *), "");
void cxxbridge1$unique_ptr$FastGICP$null(::std::unique_ptr<::FastGICP> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastGICP>();
}
void cxxbridge1$unique_ptr$FastGICP$raw(::std::unique_ptr<::FastGICP> *ptr, ::FastGICP *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastGICP>(raw);
}
::FastGICP const *cxxbridge1$unique_ptr$FastGICP$get(::std::unique_ptr<::FastGICP> const &ptr) noexcept {
  return ptr.get();
}
::FastGICP *cxxbridge1$unique_ptr$FastGICP$release(::std::unique_ptr<::FastGICP> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$FastGICP$drop(::std::unique_ptr<::FastGICP> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::FastGICP>::value>{}(ptr);
}

static_assert(::rust::detail::is_complete<::FastVGICP>::value, "definition of FastVGICP is required");
static_assert(sizeof(::std::unique_ptr<::FastVGICP>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::FastVGICP>) == alignof(void *), "");
void cxxbridge1$unique_ptr$FastVGICP$null(::std::unique_ptr<::FastVGICP> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastVGICP>();
}
void cxxbridge1$unique_ptr$FastVGICP$raw(::std::unique_ptr<::FastVGICP> *ptr, ::FastVGICP *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastVGICP>(raw);
}
::FastVGICP const *cxxbridge1$unique_ptr$FastVGICP$get(::std::unique_ptr<::FastVGICP> const &ptr) noexcept {
  return ptr.get();
}
::FastVGICP *cxxbridge1$unique_ptr$FastVGICP$release(::std::unique_ptr<::FastVGICP> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$FastVGICP$drop(::std::unique_ptr<::FastVGICP> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::FastVGICP>::value>{}(ptr);
}

static_assert(::rust::detail::is_complete<::FastGICPI>::value, "definition of FastGICPI is required");
static_assert(sizeof(::std::unique_ptr<::FastGICPI>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::FastGICPI>) == alignof(void *), "");
void cxxbridge1$unique_ptr$FastGICPI$null(::std::unique_ptr<::FastGICPI> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastGICPI>();
}
void cxxbridge1$unique_ptr$FastGICPI$raw(::std::unique_ptr<::FastGICPI> *ptr, ::FastGICPI *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastGICPI>(raw);
}
::FastGICPI const *cxxbridge1$unique_ptr$FastGICPI$get(::std::unique_ptr<::FastGICPI> const &ptr) noexcept {
  return ptr.get();
}
::FastGICPI *cxxbridge1$unique_ptr$FastGICPI$release(::std::unique_ptr<::FastGICPI> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$FastGICPI$drop(::std::unique_ptr<::FastGICPI> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::FastGICPI>::value>{}(ptr);
}

static_assert(::rust::detail::is_complete<::FastVGICPI>::value, "definition of FastVGICPI is required");
static_assert(sizeof(::std::unique_ptr<::FastVGICPI>) == sizeof(void *), "");
static_assert(alignof(::std::unique_ptr<::FastVGICPI>) == alignof(void *), "");
void cxxbridge1$unique_ptr$FastVGICPI$null(::std::unique_ptr<::FastVGICPI> *ptr) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastVGICPI>();
}
void cxxbridge1$unique_ptr$FastVGICPI$raw(::std::unique_ptr<::FastVGICPI> *ptr, ::FastVGICPI *raw) noexcept {
  ::new (ptr) ::std::unique_ptr<::FastVGICPI>(raw);
}
::FastVGICPI const *cxxbridge1$unique_ptr$FastVGICPI$get(::std::unique_ptr<::FastVGICPI> const &ptr) noexcept {
  return ptr.get();
}
::FastVGICPI *cxxbridge1$unique_ptr$FastVGICPI$release(::std::unique_ptr<::FastVGICPI> &ptr) noexcept {
  return ptr.release();
}
void cxxbridge1$unique_ptr$FastVGICPI$drop(::std::unique_ptr<::FastVGICPI> *ptr) noexcept {
  ::rust::deleter_if<::rust::detail::is_complete<::FastVGICPI>::value>{}(ptr);
}
} // extern "C"
