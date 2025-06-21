FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    curl \
    git \
    python3 \
    python3-pip \
    python3-pandas \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and extract CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.30.0-linux-x86_64.tar.gz -C /opt && \
    rm cmake-3.30.0-linux-x86_64.tar.gz

# Add CMake to PATH (without moving its structure)
ENV PATH="/opt/cmake-3.30.0-linux-x86_64/bin:${PATH}"

# Fix Git "dubious ownership" warning
RUN git config --system --add safe.directory '*'

WORKDIR /home/sdegno