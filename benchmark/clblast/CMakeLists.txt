find_package(CLBlast REQUIRED)

add_library(benchmark-clblast-common OBJECT
    clwrap.h clwrap.cpp
    utils.hpp
    main.cpp
)
target_link_libraries(benchmark-clblast-common PUBLIC clblast benchmark-common)
target_include_directories(benchmark-clblast-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
