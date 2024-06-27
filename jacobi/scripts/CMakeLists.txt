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
        COMMAND ./build.sh CUCORR_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
        DEPENDS cutest_memusage
    )

    add_custom_target(${name}-tsan-vanilla-build
        COMMAND "make" clean
        COMMAND ./build.sh CUCORR_WRAPPER=OFF -fsanitize=thread
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
        COMMAND ./run.sh CUCORR_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
    )
endfunction()


set(CUCORR_YAML_DIR ${PROJECT_SOURCE_DIR}/jacobi)
set(CUCORR_SUPPRESSION_DIR ${PROJECT_SOURCE_DIR}/jacobi/scripts)

configure_jacobi_cfg(${CMAKE_CURRENT_SOURCE_DIR}/build.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../build.sh
  )

configure_jacobi_cfg(${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../run.sh
  )

function (make_sbatch)
  set(CUTEST_JACOBI_BASEEXE ${PROJECT_SOURCE_DIR}/jacobi/bin/jacobi_cuda_aware_mpi)
  set(CUTEST_JACOBI_CUCORR ${CUTEST_JACOBI_BASEEXE}_cucorr)
  set(CUTEST_JACOBI_VANILLA ${CUTEST_JACOBI_BASEEXE}_vanilla)
  set(CUTEST_JACOBI_TSAN_VANILLA ${CUTEST_JACOBI_BASEEXE}_tsan_vanilla)
  set(CUTEST_JACOBI_ARGS "-t 2 1")

  get_filename_component(MUST_BIN_DIR "${must_run}" DIRECTORY ABSOLUTE)

  configure_jacobi_cfg(${CMAKE_CURRENT_SOURCE_DIR}/sbatch.sh.in
    ${CMAKE_CURRENT_SOURCE_DIR}/../sbatch.sh
    )
  add_custom_target(jacobi-sbatch
    COMMAND sbatch sbatch.sh
    COMMAND squeue
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/jacobi
  )
endfunction()

make_sbatch()

add_build_clean_target(jacobi
    ${PROJECT_SOURCE_DIR}/jacobi
  )
