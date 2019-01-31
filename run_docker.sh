#!/bin/bash

docker run --rm -p 18877:8888 \
    -v /u07/wei/face_search_engine/faces:/face-search-engine/data \
    -v /u07/wei/face_search_engine/db:/face-search-engine/db \
    aaalgo/face_search_engine $*
