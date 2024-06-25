cmake_minimum_required(VERSION 3.20)

project(cucorr-tests)


set(CUCORR_PATH "" CACHE STRING "Path to cucorr installation")

if(NOT IS_DIRECTORY ${CUCORR_PATH})
  if(DEFINED ENV{CUCORR_PATH})
    set(CUCORR_PATH $ENV{CUCORR_PATH} CACHE STRING "" FORCE)
  endif()
endif()

if(NOT DEFINED CUCORR_PATH OR NOT IS_DIRECTORY ${CUCORR_PATH})
  message(FATAL_ERROR "Path to cucorr installation not set correctly (${CUCORR_PATH})")
endif()

message(STATUS "Path to cucorr installation is set to ${CUCORR_PATH}")

find_package(MPI REQUIRED)

# Not so portable, but suffices
find_program(cucorr_mpicxx
    cucorr-mpic++
    PATHS ${CUCORR_PATH} ${CUCORR_PATH}/bin
    REQUIRED.
)
find_program(cucorr_mpicc
    cucorr-mpicc
    PATHS ${CUCORR_PATH} ${CUCORR_PATH}/bin
    REQUIRED
)
find_library(lib_cucorr_runtime
    NAMES CucorrRuntime
    PATHS ${CUCORR_PATH}/lib ${CUCORR_PATH}
    REQUIRED
    NO_DEFAULT_PATH
)

find_library(lib_cucorr_interceptor
    NAMES CucorrMPIInterceptor
    PATHS ${CUCORR_PATH}/lib ${CUCORR_PATH}

    REQUIRED
    NO_DEFAULT_PATH
)

find_program(must_run
    mustrun
    PATHS ${MUST_PATH} ${MUST_PATH}/bin
)

message(STATUS "cucorr mpicc wrapper is set to ${cucorr_mpicc}")
message(STATUS "cucorr mpicxx wrapper is set to ${cucorr_mpicxx}")
message(STATUS "cucorr lib is set to ${lib_cucorr_runtime}")
message(STATUS "cucorr interceptor lib is set to ${lib_cucorr_interceptor}")
message(STATUS "MUST mustrun is set to ${must_run}")

enable_testing()

add_subdirectory(support)
add_subdirectory(testsuite)

add_subdirectory(jacobi/scripts)
add_subdirectory(tealeaf/scripts)
