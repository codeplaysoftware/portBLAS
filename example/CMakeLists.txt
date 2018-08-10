cmake_minimum_required(VERSION 3.2.2)

project(syclblas_example)

set(SYCLBLAS_EXAMPLE ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_EXAMPLE_INCLUDE "${SYCLBLAS_EXAMPLE}/include")

if(SYCL_DEVICE)
  add_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

file(GLOB SYCLBLAS_EXAMPLE_SRCS ./*.cpp)

foreach(example ${SYCLBLAS_EXAMPLE_SRCS})

  message(STATUS " Getting ${example} ")

  get_filename_component(SOURCE_NAME ${example} NAME_WE)
  message(STATUS " Adding ${SOURCE_NAME} ")

#   include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

  add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp  )
  target_compile_options(${SOURCE_NAME} PUBLIC ${HOST_COMPILER_OPTIONS})
  target_link_libraries(${SOURCE_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

  add_sycl_to_target(${SOURCE_NAME} ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp )

endforeach()
