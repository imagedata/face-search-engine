#!/bin/bash

docker run --rm -it --name face_search_engine_production -p 18880:8888 -v /home/frong/face-search-engine-production:/face-search-engine face-search-engine-production /bin/bash
