Use `make docker` to build the docker `aaalgo/face_search_engine`.
Run docker by mounting the following:
``
face data -> /face-search-engine/data
feature db -> /face-search-engine/db
``
The server runs at port 8888.  See `run_docker.sh` for example.

For development purpose, one can mount `$PWD` inside docker.
After that, run `./build.sh` in the source directory to produce
the donkey python module.
