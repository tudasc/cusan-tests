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
    add_custom_target(build-${name}
        COMMAND ./build.sh
        WORKING_DIRECTORY ${dir}
    )
    
    add_custom_target(clean-${name}
        COMMAND "make" clean
        WORKING_DIRECTORY ${dir}
    )

    add_custom_target(run-${name}
        COMMAND ./run.sh
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

add_build_clean_target(jacobi
    ${PROJECT_SOURCE_DIR}/jacobi
  )