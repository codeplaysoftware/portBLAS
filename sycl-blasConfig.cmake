include(CMakeFindDependencyMacro)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_dependency(OpenCL)
find_dependency(ComputeCpp)
find_dependency(OpenBLAS)

include("${CMAKE_CURRENT_LIST_DIR}/sycl-blas.cmake")