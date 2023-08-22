# Try to find the PORTBLAS library.
#
# If the library is found then the `PORTBLAS::PORTBLAS` target will be exported
# with the required include directories.
#
# Sets the following variables:
#   PORTBLAS_FOUND        - whether the system has PORTBLAS
#   PORTBLAS_INCLUDE_DIRS - the PORTBLAS include directory

find_path(PORTBLAS_INCLUDE_DIR
  NAMES portblas.h
  PATH_SUFFIXES include
  HINTS ${PORTBLAS_DIR}
  DOC "The PORTBLAS include directory"
)

find_path(PORTBLAS_SRC_DIR
  NAMES portblas.hpp
  PATH_SUFFIXES src
  HINTS ${PORTBLAS_DIR}
  DOC "The PORTBLAS source directory"
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PORTBLAS
  FOUND_VAR PORTBLAS_FOUND
  REQUIRED_VARS PORTBLAS_INCLUDE_DIR
                PORTBLAS_SRC_DIR
)

mark_as_advanced(PORTBLAS_FOUND
                 PORTBLAS_SRC_DIR
                 PORTBLAS_INCLUDE_DIR
)

if(PORTBLAS_FOUND)
  set(PORTBLAS_INCLUDE_DIRS
    ${PORTBLAS_INCLUDE_DIR}
    ${PORTBLAS_SRC_DIR}
  )
endif()

if(PORTBLAS_FOUND AND NOT TARGET PORTBLAS::PORTBLAS)
  add_library(PORTBLAS::PORTBLAS INTERFACE IMPORTED)
  set_target_properties(PORTBLAS::PORTBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PORTBLAS_INCLUDE_DIRS}"
  )
endif()
