set(sources
    blas1_asum_test.cpp
    blas1_axpy_test.cpp
    blas1_copy_test.cpp
    blas1_dot_test.cpp
    blas1_iamax_test.cpp
    blas1_iamin_test.cpp
    blas1_nrm2_test.cpp
    blas1_rotg_test.cpp
    blas1_scal_test.cpp
    blas1_swap_test.cpp
    blas2_gemv_test.cpp
    blas2_ger_test.cpp
    blas3_gemm_test.cpp
    sycl_buffer_test.cpp
)
add_executable(unittests main.cpp ${sources})
target_link_libraries(unittests PRIVATE test_common)
add_sycl_to_target(TARGET unittests SOURCES ${sources})
