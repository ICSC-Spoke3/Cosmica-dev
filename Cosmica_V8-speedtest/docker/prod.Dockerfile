FROM sdegno-dev

ARG ARCH

COPY . /home/sdegno/Cosmica_V8/

WORKDIR /home/sdegno/Cosmica_V8

RUN rm -rf ./build
RUN cmake -S . ./build -DCMAKE_CUDA_ARCHITECTURES=$ARCH
RUN cmake --build ./build --target Cosmica -- -j 10