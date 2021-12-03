#!/bin/bash

mode="debug"
if [ $# -eq 1 ]; then
    mode=$1
fi

pwd="$(echo $PWD | sed -n 's/.*\/\(.*\)/\1/p')"
if [[ $pwd == "build" ]]; then
    cd ..
fi

if [ -d build ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

mkdir build

cd build

if [[ $mode == "debug" ]]; then
    cmake -DCMAKE_BUILD_TYPE="Debug" ..
elif [[ $mode == "release" ]]; then
    cmake -DCMAKE_BUILD_TYPE="Release" ..
else
    echo "Error: Valid options are 'release' or 'debug'"
    cd ..
    exit 1
fi

cd ..

if [[ $pwd == "build" ]]; then
    cd build
    exec $(cd .)
fi
