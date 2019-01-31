#!/bin/bash

docker run -it --rm -p 18877:8888 \
    -v /u07/wei/face_search_engine/faces:/face_search_engine/data \
    -v /u07/wei/face_search_engine/db:/face_search_engine/db \
    aaalgo/face_search_engine 
