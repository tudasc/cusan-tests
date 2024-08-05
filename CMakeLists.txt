cmake_minimum_required(VERSION 3.20)

project(cusan-tests)


set(CUSAN_PATH "${PROJECT_SOURCE_DIR}/cusan/install" CACHE STRING "Path to CuSan installation")

if(NOT IS_DIRECTORY ${CUSAN_PATH})
  if(DEFINED ENV{CUSAN_PATH})
    set(CUSAN_PATH $ENV{CUSAN_PATH} CACHE STRING "" FORCE)
  endif()
endif()

if(NOT DEFINED CUSAN_PATH OR NOT IS_DIRECTORY ${CUSAN_PATH})
  message(WARNING "Path to cusan installation not set correctly (${CUSAN_PATH})")
endif()

message(STATUS "Path to cusan installation is set to ${CUSAN_PATH}")

find_package(MPI REQUIRED)

# Not so portable, but suffices
find_program(cusan_mpicxx
    NAMES cusan-mpic++
    PATHS ${CUSAN_PATH} ${CUSAN_PATH}/bin
    NO_DEFAULT_PATH
)
find_program(cusan_mpicc
    NAME cusan-mpicc
    PATHS ${CUSAN_PATH} ${CUSAN_PATH}/bin
    NO_DEFAULT_PATH
)
find_library(lib_cusan_runtime
    NAMES CusanRuntime
    PATHS ${CUSAN_PATH}/lib ${CUSAN_PATH}
    NO_DEFAULT_PATH
)

find_library(lib_cusan_interceptor
    NAMES CusanMPIInterceptor
    PATHS ${CUSAN_PATH}/lib ${CUSAN_PATH}
    NO_DEFAULT_PATH
)

find_program(must_run
    mustrun
    PATHS ${MUST_PATH} ${MUST_PATH}/bin ${PROJECT_SOURCE_DIR}/must-tsan/install/bin
    NO_DEFAULT_PATH
)

message(STATUS "cusan mpicc wrapper is set to ${cusan_mpicc}")
message(STATUS "cusan mpicxx wrapper is set to ${cusan_mpicxx}")
message(STATUS "cusan lib is set to ${lib_cusan_runtime}")
message(STATUS "cusan interceptor lib is set to ${lib_cusan_interceptor}")
message(STATUS "MUST mustrun is set to ${must_run}")

enable_testing()
add_subdirectory(support)
add_subdirectory(testsuite)
add_subdirectory(jacobi/scripts)
add_subdirectory(tealeaf/scripts)
