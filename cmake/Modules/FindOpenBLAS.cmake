include(FindPackageHandleStandardArgs)

if(TARGET OpenBLAS::OpenBLAS)
    return()
endif()

find_path(OPENBLAS_INCLUDE_DIR cblas.h)
find_library(OPENBLAS_LIBRARY openblas)

find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OPENBLAS_LIBRARY OPENBLAS_INCLUDE_DIR)

add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDE_DIR}
    IMPORTED_LOCATION ${OPENBLAS_LIBRARY})
