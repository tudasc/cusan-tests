function(configure_tealeaf_cfg input output)
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

function(add_build_clean_target name dir)
    add_custom_target(${name}-build
        COMMAND ./build.sh
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )
    
    add_custom_target(${name}-vanilla-build
        COMMAND ./build.sh CUSAN_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )

    add_custom_target(${name}-tsan-vanilla-build
        COMMAND ./build.sh CUSAN_WRAPPER=OFF -fsanitize=thread
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )

    add_custom_target(${name}-all-build)

    add_dependencies(${name}-all-build
        ${name}-build
        ${name}-vanilla-build
        ${name}-tsan-vanilla-build
    )

    add_custom_target(${name}-clean
        WORKING_DIRECTORY ${dir}
    )

    add_custom_target(${name}-run
        COMMAND ./run.sh
        WORKING_DIRECTORY ${dir}
    )
    
    add_custom_target(${name}-vanilla-run
        COMMAND ./run.sh CUSAN_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
    )
endfunction()

set(CUTEST_NP 2)

set(CUSAN_YAML_DIR ${PROJECT_SOURCE_DIR}/tealeaf)
set(CUSAN_SUPPRESSION_DIR ${PROJECT_SOURCE_DIR}/tealeaf/scripts)

set(CUTEST_SUPPORT_DIR ${PROJECT_SOURCE_DIR}/support)
get_filename_component(MUST_BIN_DIR "${must_run}" DIRECTORY ABSOLUTE)
set(CUTEST_MUST_LAYOUT_FILE ${PROJECT_SOURCE_DIR}/support/must-mpi-layout-2.xml)

configure_tealeaf_cfg(${CMAKE_CURRENT_SOURCE_DIR}/build.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../build.sh
  )

configure_tealeaf_cfg(${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../run.sh
  )

function (make_sbatch)
  set(CUTEST_TEALEAF_CUSAN_BUILD_BASE ${PROJECT_SOURCE_DIR}/tealeaf)
  set(CUTEST_TEALEAF_CUSAN ${CUTEST_TEALEAF_CUSAN_BUILD_BASE}/build_cusan/cuda-tealeaf)
  set(CUTEST_TEALEAF_VANILLA ${CUTEST_TEALEAF_CUSAN_BUILD_BASE}/build_vanilla/cuda-tealeaf)
  set(CUTEST_TEALEAF_TSAN_VANILLA ${CUTEST_TEALEAF_CUSAN_BUILD_BASE}/build_tsan_vanilla/cuda-tealeaf)
  set(CUTEST_TEALEAF_ARGS "")

  configure_tealeaf_cfg(${CMAKE_CURRENT_SOURCE_DIR}/sbatch.sh.in
    ${CMAKE_CURRENT_SOURCE_DIR}/../sbatch.sh
    )
  add_custom_target(tealeaf-sbatch
    COMMAND sbatch sbatch.sh
    COMMAND squeue
    COMMAND date +%T
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tealeaf
  )
endfunction()

make_sbatch()

add_build_clean_target(tealeaf
    ${PROJECT_SOURCE_DIR}/tealeaf
  )