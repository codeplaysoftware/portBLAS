find_path(NPY_INCLUDE_DIRS "libnpy/npy.hpp"
  HINTS ${ACL_ROOT}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(npy REQUIRED_VARS NPY_INCLUDE_DIRS)

if(npy_FOUND AND NOT TARGET npy::npy)
    add_library(npy::npy INTERFACE IMPORTED)
    set_target_properties(npy::npy PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NPY_INCLUDE_DIRS}
    )
endif()
