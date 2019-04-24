find_package(BLAS REQUIRED)
find_package(GTest REQUIRED CONFIG)

if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_property(TARGET BLAS::BLAS PROPERTY
        INTERFACE_LINK_LIBRARIES
            ${BLAS_LIBRARIES}
            ${BLAS_LINKER_FLAGS}
    )
endif()


if(SYCL_DEVICE)
  add_compile_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

add_subdirectory(common)
add_subdirectory(unittest)
add_subdirectory(exprtest)
