#!/bin/bash

export CONFIG_DIR=`readlink -e $PWD`
if [ ! -d donkey ]
then
    git clone https://github.com/aaalgo/donkey
fi
pushd donkey/src
./setup.py build
popd
cp donkey/src/build/lib.linux-x86_64-*/donkey.cpython-*.so ./


