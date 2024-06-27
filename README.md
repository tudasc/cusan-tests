# cucorr-tests

Test environment for CUDA-aware MPI race detection.

## Overview

- [testsuite](testsuite/): Unit tests with data races
- [jacobi Solver mini-app](jacobi/): CUDA-aware MPI C implementation by [NVidia](https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/cuda-aware-mpi-example/src)
- [tealeaf mini-app](tealeaf/): CUDA-aware MPI C++ heat conduction, [see TeaLeaf](https://github.com/UoB-HPC/TeaLeaf)

### Requirements

[CMakeLists](CMakeLists.txt) facilitates the execution of our tests by generating appropriate targets.

[CMakeLists](CMakeLists.txt) will require the following environment variables:
- `CUCORR_PATH`: Path to the cucorr installation, to detect the compiler wrappers/libs
- `MUST_PATH`: PATH to the MUST installation, to detect `mustrun`

#### Software
- System software: `gcc/11 cuda/11.5 openmpi/4.0.7 git/2.40 python/3.10 clang/14`
- Cucorr: [bootstrap](support/cucorr-bootstrap.sh)
- MUST: [bootstrap](support/must-bootstrap.sh)
- testsuite: llvm-lit, FileCheck

### testsuite Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](testsuite/CMakeLists.txt): 
    - `make check-cutests`: All
    - `make check-cutests-mpi-to-cuda`: Only MPI to cuda races
    - `make check-cutests-cuda-to-mpi`: Only cuda to MPI races

### Jacobi Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](jacobi/scripts/CMakeLists.txt): 
    - `make jacobi-all-build`: builds vanilla, vanilla-tsan, cucorr
    - `make jacobi-run` or `make jacobi-vanilla-run`: Run on current node
- [`sbatch.sh`](jacobi/scripts/sbatch.sh.in): To run on compute node (requires `make jacobi-all-build` to be run)

### TeaLeaf Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](tealeaf/scripts/CMakeLists.txt): 
    - `make tealeaf-all-build`: builds vanilla, vanilla-tsan, cucorr
    - `make tealeaf-run` or `make tealeaf-vanilla-run`: Run on current node
- [`sbatch.sh`](tealeaf/scripts/sbatch.sh.in): To run on compute node (requires `make tealeaf-all-build` to be run)
