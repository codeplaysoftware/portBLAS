cmake_minimum_required (VERSION 3.4.3)
project(sycl-blas VERSION 0.1)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake OPTIONAL)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")
find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# We add some flags to workaround OpenCL platform bugs, see ComputeCpp documentation
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -no-serial-memop -sycl-compress-name)

add_library(sycl-blas INTERFACE)
target_include_directories(sycl-blas INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sycl-blas INTERFACE ComputeCpp::ComputeCpp)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/sycl-blasConfigVersion.cmake"
	VERSION ${sycl-blas_VERSION_MAJOR}
	COMPATIBILITY ExactVersion
)

install(DIRECTORY include DESTINATION .)
install(FILES
	sycl-blasConfig.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/sycl-blasConfigVersion.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/Modules/FindComputeCpp.cmake"
	DESTINATION lib/sycl-blas
)
configure_file(sycl-blasConfig.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/sycl-blasConfig.cmake"
	COPYONLY
)

install(TARGETS sycl-blas EXPORT sycl-blas
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
export(EXPORT sycl-blas
	FILE "${CMAKE_CURRENT_BINARY_DIR}/sycl-blas.cmake"
	NAMESPACE sycl::
)
install(EXPORT sycl-blas
	DESTINATION lib/sycl-blas
	NAMESPACE sycl::
)