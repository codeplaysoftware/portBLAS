set(sources
    blas1_iface_test.cpp
)

add_executable(exprtest main.cpp ${sources})
target_link_libraries(exprtest PRIVATE test_common)
add_sycl_to_target(TARGET exprtest SOURCES ${sources})
