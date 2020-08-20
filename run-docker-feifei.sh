#!/bin/bash

docker run --rm -it --name face_feature_extractor -p 18881:8888 -v /home/frong/face-feature-extractor:/face-search-engine face-feature-extractor /bin/bash
