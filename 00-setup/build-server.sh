#!/usr/bin/env bash
# Build the docker image from the PrismML llama.cpp fork at a given tag.
#
# Two build arguments for GB10 (see README.md):
#   CUDA_VERSION=13.0.1         CUDA of the image; must not be newer than the driver supports
#   CUDA_DOCKER_ARCH=121a-real  compile for GB10 only
#
# Usage:  ./build-server.sh
#
set -euo pipefail

FORK=https://github.com/PrismML-Eng/llama.cpp
TAG=prism-b10735-842b188
COMMIT=842b1880415d6f508f03b789e5ce70194def7bfd

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/.build}
CUDA_VERSION=${CUDA_VERSION:-13.0.1}
CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH:-121a-real}
IMAGE=${IMAGE:-prism-llamacpp:b10735}

# What this machine has
if command -v nvidia-smi >/dev/null; then
  echo "GPU:  $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)"
  echo "      (name, compute capability, driver); $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*')"
fi
echo "Building $IMAGE with CUDA_VERSION=$CUDA_VERSION CUDA_DOCKER_ARCH=$CUDA_DOCKER_ARCH"

# Get the llama.cpp fork at the pinned tag, and check the tag still points at the same commit.
src=$BUILD_DIR/llama.cpp
if [ ! -d "$src" ]; then
  mkdir -p "$BUILD_DIR"
  git -c advice.detachedHead=false clone --filter=blob:none --branch "$TAG" "$FORK" "$src"
fi
head=$(git -C "$src" rev-parse HEAD)
if [ "$head" != "$COMMIT" ] || [ "$(git -C "$src" rev-parse --is-shallow-repository)" = true ]; then
  echo "error: $src is not a full checkout of $TAG ($COMMIT)." >&2
  echo "Delete $src and run this script again." >&2
  exit 1
fi

log=$BUILD_DIR/build-$CUDA_DOCKER_ARCH.log
docker build -f "$src/.devops/cuda.Dockerfile" --target server \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg CUDA_DOCKER_ARCH="$CUDA_DOCKER_ARCH" \
  --build-arg APP_VERSION="$TAG" \
  --build-arg APP_REVISION="$COMMIT" \
  --progress=plain \
  -t "$IMAGE" "$src" 2>&1 | tee "$log"

# report the architecture the image is compiled for
if grep -q "Using CMAKE_CUDA_ARCHITECTURES=$CUDA_DOCKER_ARCH " "$log"; then
  echo "OK: compiled for $CUDA_DOCKER_ARCH."
elif grep -q "Using CMAKE_CUDA_ARCHITECTURES=" "$log"; then
  echo "warning: compiled for a different architecture than $CUDA_DOCKER_ARCH:" >&2
  grep "Using CMAKE_CUDA_ARCHITECTURES=" "$log" >&2
else
  echo "Note: the compile step came from Docker's cache, so the log doesn't show the architecture."
fi
echo "Built $IMAGE (log: $log)"
