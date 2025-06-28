#pragma once
#include "wrapper.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

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
