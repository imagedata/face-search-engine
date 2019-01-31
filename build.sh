#!/bin/bash

export CONFIG_DIR=`readlink -e $PWD`

pushd donkey_src/src
./setup.py build
popd
cp donkey_src/src/build/lib.linux-x86_64-*/donkey.cpython-*.so ./

