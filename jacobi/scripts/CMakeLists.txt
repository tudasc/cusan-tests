function(configure_jacobi_cfg input output)
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
        COMMAND "make" clean
        COMMAND ./build.sh
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )
    
    add_custom_target(${name}-vanilla-build
        COMMAND "make" clean
        COMMAND ./build.sh CUSAN_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )

    add_custom_target(${name}-tsan-vanilla-build
        COMMAND "make" clean
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
        COMMAND "make" clean
        WORKING_DIRECTORY ${dir}
    )

    add_custom_target(${name}-veryclean
        COMMAND "make" veryclean
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
set(CUSAN_YAML_DIR ${PROJECT_SOURCE_DIR}/jacobi)
set(CUSAN_SUPPRESSION_DIR ${PROJECT_SOURCE_DIR}/jacobi/scripts)
set(CUTEST_SUPPORT_DIR ${PROJECT_SOURCE_DIR}/support)
set(CUTEST_MUST_LAYOUT_FILE ${PROJECT_SOURCE_DIR}/support/must-mpi-layout-${CUTEST_NP}.xml)
get_filename_component(MUST_BIN_DIR "${must_run}" DIRECTORY ABSOLUTE)
set(CUTEST_JACOBI_ARGS "-t ${CUTEST_NP} 1 -d 2048")

configure_jacobi_cfg(${CMAKE_CURRENT_SOURCE_DIR}/build.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../build.sh
  )

configure_jacobi_cfg(${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../run.sh
  )

function (make_sbatch target file_in file_out)
  set(CUTEST_JACOBI_BASEEXE ${PROJECT_SOURCE_DIR}/jacobi/bin/jacobi_cuda_aware_mpi)
  set(CUTEST_JACOBI_CUSAN ${CUTEST_JACOBI_BASEEXE}_cusan)
  set(CUTEST_JACOBI_VANILLA ${CUTEST_JACOBI_BASEEXE}_vanilla)
  set(CUTEST_JACOBI_TSAN_VANILLA ${CUTEST_JACOBI_BASEEXE}_tsan_vanilla)

  configure_jacobi_cfg(${file_in}
    ${file_out}
    )
  add_custom_target(jacobi-${target}
    COMMAND sbatch ${file_out}
    COMMAND squeue
    COMMAND date +%T
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/jacobi
  )
endfunction()

make_sbatch(sbatch ${CMAKE_CURRENT_SOURCE_DIR}/sbatch.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/../sbatch.sh)

set(CUTEST_JACOBI_ARGS "-t ${CUTEST_NP} 1 -d $d")
make_sbatch(scale ${CMAKE_CURRENT_SOURCE_DIR}/sbatch-scale.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/../sbatch-scale.sh)

add_build_clean_target(jacobi
    ${PROJECT_SOURCE_DIR}/jacobi
  )
