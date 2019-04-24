include(CMakeFindDependencyMacro)
find_dependency(ComputeCpp REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/sycl-blas-targets.cmake")
