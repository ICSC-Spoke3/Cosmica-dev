docker build -t sdegno-dev -f docker/dev.Dockerfile .

if [ -n "$1" ]; then
    ARCH="$1"
    echo "ARCH provided as argument: $ARCH"
else
    GPU_NAME=$(docker run --rm --gpus all sdegno-dev \
      bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1' | tail -1)


    case "$GPU_NAME" in
        *A30*) ARCH=80 ;;
        *A40*) ARCH=86 ;;
        *A100*) ARCH=80 ;;
        *H100*) ARCH=90 ;;
        *L4*)   ARCH=89 ;;
        *RTX\ 40*) ARCH=89 ;;
        *RTX\ 30*) ARCH=86 ;;
        *RTX\ 20*) ARCH=75 ;;
        *T4*)   ARCH=75 ;;
        *V100*) ARCH=70 ;;
        *P100*) ARCH=60 ;;
        *)      ARCH=80 ;;
    esac

    echo "GPU name: '$GPU_NAME', Architecture: '$ARCH'"
fi

docker build -t sdegno-prod -f docker/prod.Dockerfile --build-arg ARCH=$ARCH .