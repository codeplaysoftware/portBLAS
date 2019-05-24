configure_file(include/config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/include/config.hpp")

add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE
    include
    "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(test_common INTERFACE
    codeplay::sycl-blas
    SystemBLAS::BLAS
    GTest::gtest
    GTest::gmock
)
