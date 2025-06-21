FROM sdegno-dev

ARG SSH_KEY
ARG ARCH=80

RUN mkdir $HOME/.ssh/
RUN echo "$SSH_KEY" > $HOME/.ssh/id_rsa
RUN chmod 600 $HOME/.ssh/id_rsa
RUN touch $HOME/.ssh/known_hosts
RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts

RUN git clone git@github.com:SDEGnOHub/Cosmica-dev.git

WORKDIR /home/sdegno/Cosmica-dev/Cosmica_V8-speedtest

RUN cmake -S . ./build -DCMAKE_CUDA_ARCHITECTURES=$ARCH
RUN cmake --build ./build --target Cosmica -- -j 10