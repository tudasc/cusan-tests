
export CUCORR_BUILD_DIR="../../cucorr/build/"

export CUCORR_MPICXX_EXEC=$CUCORR_BUILD_DIR"scripts/cucorr-mpic++-test"
export CUCORR_MPICC_EXEC=$CUCORR_BUILD_DIR"scripts/cucorr-mpicc-test"
export CUCORR_MPI_PRELOAD=$CUCORR_BUILD_DIR"lib/runtime/libCucorrMPIInterceptor-d.so"

make all

LD_PRELOAD=$CUCORR_MPI_PRELOAD mpiexec -np 2 ./bin/jacobi_cuda_normal_mpi -t 2 1