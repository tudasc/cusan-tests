cmake_minimum_required(VERSION 3.20)

project(MPIMemUsage VERSION 1.0 LANGUAGES C)

set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)

include_directories(${MPI_C_INCLUDE_DIRS})

add_library(cutest_memusage SHARED MemUsage.c)

target_link_libraries(cutest_memusage MPI::MPI_C)

set_target_properties(cutest_memusage PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
