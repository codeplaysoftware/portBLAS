find_path(CLHPP_INCLUDE_DIRS CL/cl2.hpp
  HINTS ${OpenCL_INCLUDE_DIR}
        ${ACL_ROOT}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLHPP REQUIRED_VARS CLHPP_INCLUDE_DIRS)

if(CLHPP_FOUND AND NOT TARGET CLHPP::CLHPP)
    add_library(CLHPP::CLHPP INTERFACE IMPORTED)
    set_target_properties(CLHPP::CLHPP PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CLHPP_INCLUDE_DIRS}
    )
endif()
