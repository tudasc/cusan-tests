#!/bin/bash

export TYPEART_TYPE_FILE="/home/a_h_ck/workspace/cucorr-tests/tealeaf/types-tealeaf.yaml"
export CUCORR_KERNEL_DATA_FILE="/home/a_h_ck/workspace/cucorr-tests/tealeaf/kernel-tealeaf.yaml"

export TSAN_OPTIONS="exitcode=0 suppressions=/home/a_h_ck/workspace/cucorr-tests/tealeaf/scripts/suppression-lb2.txt"


build_dir="build_cucorr"
preflag=$1
if [ ! -z "$preflag" ]; then
  export $preflag
  echo "Exported : $preflag"
  build_dir="build_vanilla"
fi

if [ -z "$preflag" ]; then
    LD_PRELOAD="/home/a_h_ck/workspace/cucorr/install/cucorr/lib64/libCucorrMPIInterceptor.so" mpiexec -np 2 ./${build_dir}/cuda-tealeaf 
else
    mpiexec -np 2 ./${build_dir}/cuda-tealeaf 
fi