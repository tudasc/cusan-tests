# cusan-tests &middot; [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Test environment for CUDA-aware MPI race detection.

## Overview

- [testsuite](testsuite/): Unit tests with data races
- [jacobi Solver mini-app](jacobi/): CUDA-aware MPI C implementation by [NVidia](https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/cuda-aware-mpi-example/src)
- [tealeaf mini-app](tealeaf/): CUDA-aware MPI C++ heat conduction, [see TeaLeaf](https://github.com/UoB-HPC/TeaLeaf)

### Requirements

[CMakeLists](CMakeLists.txt) facilitates the execution of our tests by generating appropriate targets.

To setup, execute:

```shell
$ cd cusan-tests
$ mkdir build && cd build
$ cmake ..
```

[CMakeLists](CMakeLists.txt) will require the following environment variables:
- `CUSAN_PATH`: Path to the cusan installation, to detect the compiler wrappers/libs
- `MUST_PATH`: PATH to the MUST installation, to detect `mustrun`

#### Software & GPU
- System modules: `1) gcc/11.2.0 2) cuda/11.8 3) openmpi/4.1.6 4) git/2.40.0 5) python/3.10.10 6) clang/14.0.6`
- CuSan: [bootstrap](support/cusan-bootstrap.sh)
- MUST: [bootstrap](support/must-bootstrap.sh)
- testsuite: llvm-lit, FileCheck
- GPU: Tesla T4 and Tesla V100 (arch=sm_70)

### testsuite Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](testsuite/CMakeLists.txt): 
    - `make check-cutests`: All
    - `make check-cutests-mpi-to-cuda`: Only MPI to CUDA races
    - `make check-cutests-cuda-to-mpi`: Only CUDA to MPI races
    - `make check-cutests-cuda-only`: Only CUDA races

### Jacobi Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](jacobi/scripts/CMakeLists.txt): 
    - `make jacobi-all-build`: builds vanilla, vanilla-tsan, cusan
    - `make jacobi-run` or `make jacobi-vanilla-run`: Run on current node
- [`sbatch.sh`](jacobi/scripts/sbatch.sh.in): To run on compute node (requires `make jacobi-all-build` to be run)
    - `make jacobi-sbatch`: Slurm commit generated as sbatch

### TeaLeaf Targets

In [root level](./) created `build` folder `cmake ..` generates:
- Targets to build, and run on the current node, see [CMakeLists](tealeaf/scripts/CMakeLists.txt): 
    - `make tealeaf-all-build`: builds vanilla, vanilla-tsan, cusan
    - `make tealeaf-run` or `make tealeaf-vanilla-run`: Run on current node
- [`sbatch.sh`](tealeaf/scripts/sbatch.sh.in): To run on compute node (requires `make tealeaf-all-build` to be run)
    - `make tealeaf-sbatch`: Slurm commit generated as sbatch
