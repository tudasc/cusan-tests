cmake_minimum_required(VERSION 3.20)

project(MPIMemUsage VERSION 1.0 LANGUAGES C)

set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)

add_library(cutest_memusage SHARED MemUsage.c)

target_include_directories(cutest_memusage SYSTEM PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(cutest_memusage MPI::MPI_C)
set_target_properties(cutest_memusage PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)

get_filename_component(MUST_BASE_PATH "${PROJECT_SOURCE_DIR}/.." ABSOLUTE)
add_custom_target(must-bootstrap
        COMMAND "./must-bootstrap.sh" ${MUST_BASE_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_custom_target(cusan-bootstrap
    COMMAND "./cusan-bootstrap.sh" ${MUST_BASE_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

function(configure_must_cfg input output)
  file(READ ${input} contents)
  string(CONFIGURE "${contents}" contents @ONLY)
  file(GENERATE
    OUTPUT
      ${output}
    CONTENT
      "${contents}"
    FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ
      WORLD_READ
  )
endfunction()

set(MUST_LAYOUT_PROCESS_COUNT 2)
configure_must_cfg(${CMAKE_CURRENT_SOURCE_DIR}/must-mpi-layout-bench.xml.in ${CMAKE_CURRENT_SOURCE_DIR}/must-mpi-layout-2.xml)
set(MUST_LAYOUT_PROCESS_COUNT 4)
configure_must_cfg(${CMAKE_CURRENT_SOURCE_DIR}/must-mpi-layout-bench.xml.in ${CMAKE_CURRENT_SOURCE_DIR}/must-mpi-layout-4.xml)
