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

add_subdirectory(common)
add_subdirectory(unittest)
add_subdirectory(exprtest)
