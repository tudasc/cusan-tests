#!/bin/bash

module purge
ml gcc/11 cuda openmpi git python
ml clang/14

export CC=clang
export CXX=clang++

if [ -z "$1" ]; then
    base_cucorr_f=$(pwd)
else
    base_cucorr_f="$1"
fi

build_f="$base_cucorr_f"/cucorr/build
install_f="$base_cucorr_f"/cucorr/install

echo "Base folder $base_cucorr_f"
echo "Build to $build_f ; Install to $install_f"

function cucorr_fetch() {
    cd "$base_cucorr_f"
    git clone -b devel --single-branch git@github.com:ahueck/cucorr.git cucorr
}

function cucorr_config() {
    mkdir -p "$build_f"
    cd "$build_f"
    cmake ../ \
        -DCMAKE_INSTALL_PREFIX="$install_f" \
        -DCMAKE_BUILD_TYPE=Release
}

function cucorr_install() {
    cd "$build_f"
    make -j16 install
}

cucorr_fetch
cucorr_config
cucorr_install

echo "##########"
echo "Execute: export PATH="$install_f"/bin:\$PATH"
echo "Execute: export CUCORR_PATH="$install_f""
