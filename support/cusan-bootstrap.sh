#!/bin/bash

function cusan_modules() {
    module purge
    ml gcc/11 cuda openmpi git python
    ml clang/14
}

export CC=clang
export CXX=clang++

if [ -z "$1" ]; then
    base_cusan_f=$(pwd)
else
    base_cusan_f="$1"
fi

build_f="$base_cusan_f"/cusan/build
install_f="$base_cusan_f"/cusan/install

echo "Base folder $base_cusan_f"
echo "Build to $build_f ; Install to $install_f"

function cusan_fetch() {
    cd "$base_cusan_f"
    git clone https://github.com/ahueck/cusan.git cusan
}

function cusan_config() {
    mkdir -p "$build_f"
    cd "$build_f"
    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$install_f" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUSAN_FIBERPOOL=OFF \
        -DCUSAN_SOFTCOUNTER=OFF \
        -DCUSAN_SYNC_DETAIL_LEVEL=ON \
        -DTYPEART_SOFTCOUNTERS=OFF \
        -DTYPEART_SHOW_STATS=OFF  \
        -DTYPEART_MPI_INTERCEPT_LIB=OFF
}

function cusan_install() {
    cd "$build_f"
    make -j16 install
}

cusan_fetch
cusan_config
cusan_install

echo "##########"
echo "Execute: export PATH="$install_f"/bin:\$PATH"
echo "Execute: export CUSAN_PATH="$install_f""
