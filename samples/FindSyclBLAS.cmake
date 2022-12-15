# Try to find the SyclBLAS library.
#
# If the library is found then the `SyclBLAS::SyclBLAS` target will be exported
# with the required include directories.
#
# Sets the following variables:
#   SyclBLAS_FOUND        - whether the system has SyclBLAS
#   SyclBLAS_INCLUDE_DIRS - the SyclBLAS include directory

find_path(SyclBLAS_INCLUDE_DIR
  NAMES sycl_blas.h
  PATH_SUFFIXES include
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS include directory"
)

find_path(SyclBLAS_SRC_DIR
  NAMES sycl_blas.hpp
  PATH_SUFFIXES src
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS source directory"
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SyclBLAS
  FOUND_VAR SyclBLAS_FOUND
  REQUIRED_VARS SyclBLAS_INCLUDE_DIR
                SyclBLAS_SRC_DIR
)

mark_as_advanced(SyclBLAS_FOUND
                 SyclBLAS_SRC_DIR
                 SyclBLAS_INCLUDE_DIR
)

if(SyclBLAS_FOUND)
  set(SyclBLAS_INCLUDE_DIRS
    ${SyclBLAS_INCLUDE_DIR}
    ${SyclBLAS_SRC_DIR}
  )
endif()

if(SyclBLAS_FOUND AND NOT TARGET SyclBLAS::SyclBLAS)
  add_library(SyclBLAS::SyclBLAS INTERFACE IMPORTED)
  set_target_properties(SyclBLAS::SyclBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SyclBLAS_INCLUDE_DIRS}"
  )
endif()
