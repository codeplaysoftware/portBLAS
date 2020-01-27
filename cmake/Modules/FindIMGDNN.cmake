# FindIMGDNN.cmake
# Expects either to find IMGDNN in system directories, or at the location
# specified in IMGDNN_DIR. Outputs variables IMGDNN_INCLUDE_DIRS,
# IMGDNN_LIBRARIES and IMGDNN_ROOT_DIR. It also creates the target
# IMGDNN::IMGDNN, which can be linked against in the usual way.
cmake_minimum_required(VERSION 3.2.2)
include(FindPackageHandleStandardArgs)

find_library(IMGDNN_LIBRARY
  NAMES IMGDNN
  HINTS ${IMGDNN_DIR}
  PATH_SUFFIXES lib
  DOC "The Imagination DNN library")

find_path(IMGDNN_INCLUDE_DIR
  NAMES "imgdnn/imgdnn.h"
  HINTS ${IMGDNN_DIR}
  DOC "The directory with imgdnn.h and cl.h")

get_filename_component(imgdnn_canonical_dir "${IMGDNN_INCLUDE_DIR}/.." ABSOLUTE)
set(IMGDNN_ROOT_DIR "${imgdnn_canonical_dir}" CACHE PATH
  "The IMGDNN library root")
mark_as_advanced(IMGDNN_ROOT_DIR
                 IMGDNN_INCLUDE_DIR
                 IMGDNN_LIBRARY)

set(IMGDNN_INCLUDE_DIRS "${IMGDNN_INCLUDE_DIR}")
set(IMGDNN_LIBRARIES "${IMGDNN_LIBRARY}")
find_package_handle_standard_args(IMGDNN
  REQUIRED_VARS IMGDNN_ROOT_DIR
                IMGDNN_INCLUDE_DIRS
                IMGDNN_LIBRARIES)

if(IMGDNN_FOUND AND NOT TARGET IMGDNN::IMGDNN)
  add_library(IMGDNN::IMGDNN UNKNOWN IMPORTED)
  set_target_properties(IMGDNN::IMGDNN PROPERTIES
    IMPORTED_LOCATION                "${IMGDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES    "${IMGDNN_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES         "-Wl,--allow-shlib-undefined"
)
endif()
