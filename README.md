# parakeet.cpp

parakeet.cpp is a C++/ggml inference port of NVIDIA NeMo Parakeet ASR models, providing fast, dependency-light automatic speech recognition on CPU and GPU via the ggml tensor library. See `docs/superpowers/specs/` for the full design.

Supports all offline Parakeet families — CTC, RNNT, TDT, and hybrid TDT-CTC (0.6B/1.1B/110M, EN + multilingual v3) — validated at WER 0 vs NeMo. Cache-aware streaming + EOU is future work.

## Build (placeholder)

```sh
git submodule update --init --recursive
cmake -B build -DPARAKEET_BUILD_TESTS=ON
cmake --build build -j
```
