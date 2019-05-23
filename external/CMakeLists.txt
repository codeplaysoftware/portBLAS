set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/computecpp-sdk/cmake/Modules" PARENT_SCOPE)

add_library(codeplay::computecpp-sdk INTERFACE IMPORTED GLOBAL)
set_target_properties(codeplay::computecpp-sdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/computecpp-sdk/include"
)
