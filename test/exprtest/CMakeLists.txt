set(test_sources blas1_iface_test.cpp)

add_executable(exprtest ${test_sources} main.cpp)
target_link_libraries(exprtest PRIVATE sycl-blas ${GTEST_LIBRARIES} Threads::Threads)
target_include_directories(exprtest PRIVATE ${GTEST_INCLUDE_DIRS})
gtest_add_tests(exprtest "" AUTO)
add_sycl_to_target(TARGET exprtest SOURCES ${test_sources})
