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
    )
    
    add_custom_target(${name}-vanilla-build
        COMMAND ./build.sh CUCORR_WRAPPER=OFF
        WORKING_DIRECTORY ${dir}
    )

    add_custom_target(${name}-clean
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


set(CUCORR_YAML_DIR ${PROJECT_SOURCE_DIR}/tealeaf)
set(CUCORR_SUPPRESSION_DIR ${PROJECT_SOURCE_DIR}/tealeaf/scripts)

configure_tealeaf_cfg(${CMAKE_CURRENT_SOURCE_DIR}/build.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../build.sh
  )

configure_tealeaf_cfg(${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../run.sh
  )

add_build_clean_target(tealeaf
    ${PROJECT_SOURCE_DIR}/tealeaf
  )