#!/bin/bash

module purge
ml gcc/11 openmpi git python
ml clang/14

export CC=clang
export CXX=clang++

if [ -z "$1" ]; then
    base_must_f=$(pwd)
else
    base_must_f="$1"
fi

build_f="$base_must_f"/must-tsan/build
install_f="$base_must_f"/must-tsan/install

echo "Base folder $base_must_f"
echo "Build to $build_f ; Install to $install_f"

function must_fetch() {
    cd "$base_must_f"
    git clone -b feature/integrate-handle-shim-with-fiber --single-branch --depth 1 https://git-ce.rwth-aachen.de/hpc-research/correctness/MUST.git must-tsan
    cd "$base_must_f"/must-tsan
    git submodule update --recursive --init
}

function must_config() {
    mkdir -p "$build_f"
    cd "$build_f"
    cmake ../ \
        -DCMAKE_INSTALL_PREFIX="$install_f" \
        -DCMAKE_BUILD_TYPE=Release
}

function must_install() {
    cd "$build_f"
    make -j32 install install-prebuilds
}

must_fetch
must_config
must_install

echo "##########"
echo "Execute: export PATH="$install_f"/bin:\$PATH"
