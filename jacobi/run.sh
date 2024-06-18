
export CUCORR_BUILD_DIR="/home/tz91sono/cucorr_main/cucorr/install/cucorr/"
export TYPEART_TYPE_FILE=$PWD"/types.yaml"

export CUCORR_MPICXX_EXEC=$CUCORR_BUILD_DIR"bin/cucorr-mpic++-d"
export CUCORR_MPICC_EXEC=$CUCORR_BUILD_DIR"bin/cucorr-mpicc-d"
export CUCORR_MPI_PRELOAD=$CUCORR_BUILD_DIR"lib64/libCucorrMPIInterceptor-d.so"

make all

LD_PRELOAD=$CUCORR_MPI_PRELOAD mpiexec -np 2 ./bin/jacobi_cuda_normal_mpi -t 2 1