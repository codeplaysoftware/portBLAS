find_path(CLARA_INCLUDE_DIRS clara.hpp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Clara REQUIRED_VARS CLARA_INCLUDE_DIRS)
if(Clara_FOUND AND NOT TARGET Clara::Clara)
    add_library(Clara::Clara INTERFACE IMPORTED)
    set_target_properties(Clara::Clara PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CLARA_INCLUDE_DIRS}
    )
endif()
