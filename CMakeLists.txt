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
#set(cucorr_mpicxx ${CUCORR_PATH}/bin/cucorr-mpic++)
#set(cucorr_mpicc ${CUCORR_PATH}/bin/cucorr-mpicc)

message(STATUS "cucorr mpicc wrapper is set to ${cucorr_mpicc}")
message(STATUS "cucorr mpicxx wrapper is set to ${cucorr_mpicxx}")

enable_testing()
add_subdirectory(testsuite)
