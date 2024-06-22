#!/bin/bash

export TYPEART_TYPE_FILE="/home/a_h_ck/workspace/cucorr-tests/jacobi/types-jacobi.yaml"
export CUCORR_KERNEL_DATA_FILE="/home/a_h_ck/workspace/cucorr-tests/jacobi/kernel-jacobi.yaml"

export TSAN_OPTIONS="exitcode=0 suppressions=/home/a_h_ck/workspace/cucorr-tests/jacobi/scripts/suppression-lb2.txt"

LD_PRELOAD="/home/a_h_ck/workspace/cucorr/install/cucorr/lib64/libCucorrMPIInterceptor.so" mpiexec -np 2 ./bin/jacobi_cuda_normal_mpi -t 2 1