FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y build-essential python3-dev libboost-all-dev python3-pip python3-dev git cmake
RUN pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install opencv-python-headless simplejson tqdm Flask
RUN git clone https://github.com/TuXiaokang/pyseeta && cd pyseeta && git submodule update --init --recursive && cd SeetaFaceEngine/ && mkdir Release && cd Release && cmake .. && make && cd ../../ &&  python3 setup.py build && python3 setup.py install
RUN apt-get install -y wget
RUN cd / && wget http://www.aaalgo.com/public_models/face_models.tar.bz2 && tar jxf face_models.tar.bz2 && rm face_models.tar.bz2
#RUN git clone https://github.com/aaalgo/face-search-engine && cd face-search-engine && git clone https://github.com/aaalgo/donkey
#RUN cd face-search-engine && ./build.sh
ENV FACE_MODELS_DIR /face_models
