#!/bin/bash

docker run -it --rm -p 18877:8888 \
    -v $PWD:/src \
    -v /u07/wei/face_search_engine/faces:/src/data \
    -v /u07/wei/face_search_engine/db:/src/db \
    aaalgo/face_search_engine /bin/bash
