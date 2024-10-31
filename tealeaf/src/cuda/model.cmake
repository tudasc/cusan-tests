
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_required(CMAKE_CUDA_COMPILER
        "Path to the CUDA nvcc compiler")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_required(CUDA_ARCH
        "Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")


register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")


macro(setup)
    set(CMAKE_CXX_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()

    # add -forward-unknown-to-host-compiler for compatibility reasons
    # add -std=c++17 manually as older CMake seems to omit this (source gets treated as C otherwise)
    # FIXME cusan: unsupported by clang's cuda: -use_fast_math -restrict -keep -forward-unknown-to-host-compiler 
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CUSAN_ARCH_FLAG --cuda-gpu-arch=${CUDA_ARCH})
    else()
        set(CUSAN_ARCH_FLAG -arch=${CUDA_ARCH})
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 ${CUSAN_ARCH_FLAG} ${CUDA_EXTRA_FLAGS}")

    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    if(NOT CMAKE_CUDA_FLAGS_${BUILD_TYPE})
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS)
    else()
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})
    endif()

    # For Clang-18, we need the above flags to bet set, in order to avoid, e.g., __fp128 issues when try_compile is used
    enable_language(CUDA)

    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()

    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()


macro(setup_target NAME)
    # Treat everything as CUDA source
    get_target_property(PROJECT_SRC "${NAME}" SOURCES)
    foreach (SRC ${PROJECT_SRC})
        set_source_files_properties("${SRC}" PROPERTIES LANGUAGE CUDA)
    endforeach ()
endmacro()
