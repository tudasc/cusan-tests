set -xe

export CUCORR_BUILD_DIR="/home/tz91sono/cucorr_main/cucorr/install/cucorr/"
export TYPEART_TYPE_FILE=$PWD"/BUILD-cucorr/types.yaml"

export CUCORR_MPICXX_EXEC=$CUCORR_BUILD_DIR"bin/cucorr-mpic++-d"
export CUCORR_MPICC_EXEC=$CUCORR_BUILD_DIR"bin/cucorr-mpicc-d"
export CUCORR_MPI_PRELOAD=$CUCORR_BUILD_DIR"lib64/libCucorrMPIInterceptor-d.so"



TYPEART_WRAPPER=OFF cmake . \
                    -DENABLE_MPI=ON -BBUILD-cucorr -DMODEL=cuda -DCUDA_ARCH=sm_70 -DCMAKE_BUILD_TYPE=Debug\
                    -DMODEL=cuda -DCMAKE_CUDA_COMPILER=$CUCORR_MPICXX_EXEC -DCMAKE_CXX_COMPILER=$CUCORR_MPICXX_EXEC  -DCMAKE_C_COMPILER=$CUCORR_MPICC_EXEC

cmake --build BUILD-cucorr -- -j1

mpirun -np 2 ./BUILD-cucorr/cuda-tealeaf 