
add_library(sycl_blas_common STATIC "src/clwrap.cpp")
target_include_directories(sycl_blas_common PUBLIC "include")

# ComputeCpp is required so we can use cl::sycl::half
target_link_libraries(sycl_blas_common PUBLIC OpenCL::OpenCL ComputeCpp::ComputeCpp)
