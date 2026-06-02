# parakeet.cpp container image.
#
# Multi-stage build: a fat build stage compiles parakeet-cli (and the ggml
# backends it links against), then a slim runtime stage carries only the
# binary plus the ggml shared libraries.
#
# The same Dockerfile produces the CPU and CUDA variants. Select with build
# args:
#
#   CPU (default):
#     docker build -t parakeet.cpp:cpu .
#
#   CUDA:
#     docker build -t parakeet.cpp:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:12.6.2-devel-ubuntu24.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:12.6.2-runtime-ubuntu24.04 \
#       --build-arg CMAKE_EXTRA_ARGS=-DPARAKEET_GGML_CUDA=ON .
#
# The build context must be a checkout with the ggml submodule populated
# (git clone --recursive, or actions/checkout with submodules: recursive).
# Models are not bundled: mount a pre-converted .gguf at runtime.

ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

# ---------------------------------------------------------------------------
# build: configure + compile parakeet-cli and the ggml backends.
# ---------------------------------------------------------------------------
FROM ${BUILD_BASE} AS build

# Extra cmake flags appended verbatim (e.g. -DPARAKEET_GGML_CUDA=ON).
ARG CMAKE_EXTRA_ARGS=""

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# CMake auto-applies the in-tree ggml patches during configure via
# scripts/apply_ggml_patches.sh, which uses `git apply` and therefore needs
# third_party/ggml to be a git repo. Re-init it as a throwaway repo so this
# works regardless of how the submodule arrived in the build context.
RUN rm -rf third_party/ggml/.git && git -C third_party/ggml init -q

# GGML_NATIVE=OFF keeps the binary portable across the CPUs that will pull the
# published image (no host-specific ISA extensions baked in). GGML_LLAMAFILE
# stays on (forced by CMakeLists) for the tinyBLAS SGEMM speedup.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DPARAKEET_BUILD_CLI=ON \
        -DPARAKEET_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS} \
    && cmake --build build -j"$(nproc)"

# Stage the binary and every backend shared library (CPU, and CUDA when built)
# into a clean prefix the runtime stage can copy wholesale.
RUN mkdir -p /install/bin /install/lib \
    && cp build/examples/cli/parakeet-cli /install/bin/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

# ---------------------------------------------------------------------------
# runtime: slim image with just the binary and its shared libraries.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /install/bin/ /usr/local/bin/
COPY --from=build /install/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /work
ENTRYPOINT ["parakeet-cli"]
CMD ["--help"]
