/**************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename set_benchmark_label.hpp
 *
 **************************************************************************/

#ifndef COMMON_SET_BENCHMARK_LABEL
#define COMMON_SET_BENCHMARK_LABEL

#include <map>
#include <string>

extern bool const computecpp_available;
extern char const* const computecpp_version;
extern char const* const computecpp_edition;

extern const char* commit_date;
extern const char* commit_hash;

namespace blas_benchmark {
namespace utils {

namespace opencl_info {
/**
 * Add device info from the provided SYCL device to the benchmark label.
 *
 * \param [in] device    SYCL device to query for info to add to the label.
 * \param [out] key_value_map The benchmark key value pair to hold the info.
 */
inline void add_opencl_device_info(
    cl::sycl::device const& device,
    std::map<std::string, std::string>& key_value_map) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls.
  // On some OpenCL implementations this results in strings that behave
  // unexpectedly when appended to. This lambda trims those strings.
  auto trim = [](std::string s) -> std::string {
    s.resize(strlen(s.c_str()));
    return s;
  };
  auto device_name = device.get_info<cl::sycl::info::device::name>();
  auto device_version = device.get_info<cl::sycl::info::device::version>();
  auto vendor_name = device.get_info<cl::sycl::info::device::vendor>();
  auto driver_version =
      device.get_info<cl::sycl::info::device::driver_version>();

  key_value_map["device_name"] = trim(device_name);
  key_value_map["device_version"] = trim(device_version);
  key_value_map["vendor_name"] = trim(vendor_name);
  key_value_map["driver_version"] = trim(driver_version);
}

}  // namespace opencl_info

namespace computecpp_info {

/**
 * Add ComputeCpp meta-data (if available) to the benchmark label. The
 * version of compute++ is tied to the version of ComputeCpp, so the associated
 * meta-data of compute++ will be the same.
 *
 * SYCL-BLAS benchmarks will include these attributes only if ComputeCpp info is
 * available. Benchmarks from other libraries such as cublas etc. will not
 * include them.
 *
 * \param [out] key_value_map The benchmark key value pair to hold the info.
 */
inline void add_computecpp_version(
    std::map<std::string, std::string>& key_value_map) {
  if (computecpp_available) {
    key_value_map["@computecpp-version"] = computecpp_version;
    key_value_map["@computecpp-edition"] = computecpp_edition;
  }
}

}  // namespace computecpp_info

namespace datatype_info {
/**
 * Add the datatype used to the benchmark label.
 *
 * \param [out] key_value_map The benchmark key value pair to hold the info.
 */
template <typename scalar_t>
inline void add_datatype_info(
    std::map<std::string, std::string>& key_value_map);

template <>
inline void add_datatype_info<float>(
    std::map<std::string, std::string>& key_value_map) {
  key_value_map["@datatype"] = "float";
}

#ifdef BLAS_DATA_TYPE_DOUBLE
template <>
inline void add_datatype_info<double>(
    std::map<std::string, std::string>& key_value_map) {
  key_value_map["@datatype"] = "double";
}
#endif

#ifdef BLAS_DATA_TYPE_HALF
template <>
inline void add_datatype_info<cl::sycl::half>(
    std::map<std::string, std::string>& key_value_map) {
  key_value_map["@datatype"] = "half";
}
#endif  // BLAS_DATA_TYPE_HALF

}  // namespace datatype_info

enum class BackendType { SYCLBLAS, CUBLAS, ROCBLAS };

inline void set_label(benchmark::State& state,
                      const std::map<std::string, std::string>& key_value_map) {
  std::string label;
  for (auto& kv : key_value_map) {
    if (label.size()) {
      label += ",";
    }

    label += kv.first + "=" + kv.second;
  }
  state.SetLabel(label);
}

template <typename scalar_t>
inline void set_benchmark_label(benchmark::State& state,
                                const cl::sycl::queue& q, BackendType backend) {
  std::map<std::string, std::string> key_value_map;
  auto dev = q.get_device();
  opencl_info::add_opencl_device_info(dev, key_value_map);
  computecpp_info::add_computecpp_version(key_value_map);
  datatype_info::add_datatype_info<scalar_t>(key_value_map);

  key_value_map["@library"] = "SYCL-BLAS";
  key_value_map["git_hash"] = commit_hash;
  key_value_map["git_hash_date"] = commit_date;

  const auto backend_label = "@backend";
  switch (backend) {
    case BackendType::SYCLBLAS:
      key_value_map[backend_label] = "sycl-blas";
      break;
    case BackendType::CUBLAS:
      key_value_map[backend_label] = "cublas";
      break;
    case BackendType::ROCBLAS:
      key_value_map[backend_label] = "rocblas";
      break;
  }
  set_label(state, key_value_map);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_SET_BENCHMARK_LABEL
