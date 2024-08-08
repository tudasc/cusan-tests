#!/bin/bash

function must_modules() {
    module purge
    ml gcc/11 cuda openmpi git python
    ml clang/14
}

export CC=clang
export CXX=clang++

script_dir=$(pwd)

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
    git clone -b feature/integrate-handle-shim-with-fiber --single-branch https://git-ce.rwth-aachen.de/hpc-research/correctness/MUST.git must-tsan
    cd "$base_must_f"/must-tsan
    git submodule update --recursive --init
}

function must_patch() {
    cd "$base_must_f"/must-tsan
    git apply "${script_dir}"/must-changes.patch
}

function must_download() {
    cd "$base_must_f"
    wget http://hpc.rwth-aachen.de/must/files/MUST-v1.10.0-fiber-preview.tar.gz
    tar -xzvf MUST-v1.10.0-fiber-preview.tar.gz
    mv MUST-v1.10.0-beta must-tsan
}

function must_config() {
    mkdir -p "$build_f"
    cd "$build_f"
    cmake ../ \
        -DCMAKE_INSTALL_PREFIX="$install_f" \
        -DENABLE_TYPEART=OFF \
        -DCMAKE_BUILD_TYPE=Release
}

function must_install() {
    cd "$build_f"
    make -j32 install install-prebuilds
}

#must_fetch
#must_patch
must_download
must_config
must_install

echo "##########"
echo "Execute: export PATH="$install_f"/bin:\$PATH"
echo "Execute: export MUST_PATH="$install_f""
