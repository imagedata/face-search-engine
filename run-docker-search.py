#!/bin/bash

docker run --rm -it --name face_query_api -p 18882:8888 -v /home/frong/face-query-api:/face-search-engine face-feature-extractor /bin/bash
