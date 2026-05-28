# AGENTS.md

Durable reference for humans and agents maintaining parakeet.cpp.

## What this project is

parakeet.cpp is a C++17/ggml inference port of NVIDIA NeMo Parakeet ASR.
It targets CPU (GPU backends are wired but not exercised) and is designed for
parity with the NeMo reference: a Python converter turns a NeMo checkpoint into
a metadata-driven GGUF, and a C++ model loader + conformer inference engine
run the same computation natively.

Current status: Phase 3.5 complete.  Supports all offline Parakeet families —
CTC, RNNT, TDT, and hybrid TDT-CTC (0.6B/1.1B/110M, EN + multilingual v3) —
validated at WER 0 vs NeMo on every published checkpoint.  Cache-aware
streaming + EOU decoding (parakeet_realtime_eou_120m) is Phase 5 (future work).

## Repository layout

```
include/             public C/C++ API (parakeet.h)
src/                 libparakeet implementation
                       parakeet.cpp    — version()
                       common.hpp/cpp  — logging helpers
                       audio_io.hpp/cpp — dr_wav load + linear resample to 16k
                       model_loader.hpp/cpp — GGUF → ParakeetConfig + name→tensor
examples/cli/        parakeet-cli binary (info subcommand)
scripts/             Python tooling
                       convert_parakeet_to_gguf.py — .nemo/.hf → GGUF
                       gen_nemo_baseline.py         — NeMo intermediates → baseline.gguf
                       requirements.txt             — nemo_toolkit[asr] + gguf
tests/               ctest targets
                       test_smoke.cpp       — version string (model-independent)
                       test_audio_io.cpp    — wav load + resample (model-independent)
                       test_model_loader.cpp — config + tensor map (model-dependent)
                       python/check_convert.py   — converter round-trip (model-dependent)
                       python/check_baseline.py  — baseline dumper (model-dependent)
                       fixtures/clip.wav     — 2s 16k mono wav for baseline tests
third_party/         vendored deps
                       ggml/      — submodule pinned at v0.13.0
                       dr_wav.h   — vendored single header
docs/
  conversion.md      — full GGUF schema reference
  superpowers/       — spec and phase plans for agentic workers
```

## Build

```
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
```

### CMake options

| Option                   | Default | Purpose                                    |
| ------------------------ | ------- | ------------------------------------------ |
| `PARAKEET_BUILD_TESTS`   | OFF     | Compile and register ctest targets         |
| `PARAKEET_BUILD_CLI`     | ON      | Build `parakeet-cli`                       |
| `PARAKEET_SHARED`        | OFF     | Build libparakeet as a shared library      |
| `PARAKEET_GGML_CUDA`     | OFF     | Forward GGML_CUDA to the submodule         |
| `PARAKEET_GGML_METAL`    | OFF     | Forward GGML_METAL to the submodule        |
| `PARAKEET_GGML_VULKAN`   | OFF     | Forward GGML_VULKAN to the submodule       |
| `PARAKEET_GGML_HIPBLAS`  | OFF     | Forward GGML_HIPBLAS to the submodule      |

Use `-DGGML_NATIVE=OFF` when building for CI or portable binaries.

## Running tests

### Model-independent (run anywhere, no checkpoint needed)

```
ctest --test-dir build --output-on-failure -LE model
```

Expected: `test_smoke`, `test_audio_io` PASS (2/2).

### Model-dependent (need Python venv + cached checkpoint)

```
# Convert the checkpoint first (see "Converting a model" below)
PARAKEET_TEST_GGUF=/path/to/model.gguf ctest --test-dir build -R test_model_loader --output-on-failure
ctest --test-dir build -L model --output-on-failure   # includes check_convert + check_baseline
```

Tests return exit code 77 (ctest SKIP) when the venv or checkpoint is absent,
so they never break a CI environment that lacks them.

### Test labels

| Label   | Tests                                            | Needs         |
| ------- | ------------------------------------------------ | ------------- |
| (none)  | `test_smoke`, `test_audio_io`                    | nothing       |
| `model` | `test_model_loader`, `check_convert`, `check_baseline` | venv + checkpoint |

## Converting a model

Set up the Python venv once:

```
uv venv .venv --python 3.12
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r scripts/requirements.txt   # nemo_toolkit[asr] + gguf
```

NeMo 2.7.3 is the validated version.  The anchor checkpoint is
`nvidia/parakeet-tdt_ctc-110m` (~440 MB, auto-downloaded by NeMo on first use).

Convert (HuggingFace id or local `.nemo`):

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --output models/parakeet-tdt_ctc-110m.gguf
```

Featurizer window and filterbank are lifted from the checkpoint at runtime;
mel/fft parameters do not need to be specified manually.

## Dumping NeMo baselines

Used by Phase 1 parity tests.  Requires the venv and a 16k mono WAV.

```
.venv/bin/python scripts/gen_nemo_baseline.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --audio tests/fixtures/clip.wav \
    --output /tmp/baseline.gguf
```

## CLI

The binary is at `build/examples/cli/parakeet-cli` (no `build/bin`).

```
build/examples/cli/parakeet-cli info models/parakeet-tdt_ctc-110m.gguf
```

Prints the full config block: arch, encoder dims, mel parameters, vocab size,
TDT durations, etc.

## GGUF schema

See `docs/conversion.md` for the authoritative schema.  Quick summary:

- `general.architecture = "parakeet"`
- All metadata keys use the `parakeet.*` prefix (e.g.
  `parakeet.arch`, `parakeet.encoder.d_model`, `parakeet.preprocessor.n_mels`).
- **Tensor names are verbatim NeMo `state_dict` keys** — no remapping, no
  prefix stripping.  The two featurizer buffers (`preprocessor.featurizer.fb`,
  `preprocessor.featurizer.window`) are included explicitly; all other
  preprocessor internals are skipped.
- This verbatim convention is load-bearing: the C++ model loader maps
  `name → ggml_tensor*` by exact string, and the Phase 1 inference code calls
  `ml.tensor("encoder.layers.0.norm_feed_forward1.weight")` etc.  Never remap
  tensor names at conversion time.

## ggml submodule

Pinned at v0.13.0 in `third_party/ggml`.  No local patches.  To bump:
1. Update the submodule SHA.
2. Run `ctest --test-dir build --output-on-failure`.
3. Fix any API breakage in `src/model_loader.cpp` (gguf/ggml C API).

## Spec and phase plans

- Design spec: `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md`
- Phase plans: `docs/superpowers/plans/`

Fresh agents should read the spec (especially §2.1 config, §7 GGUF schema,
§8 Python env) and the current phase plan before starting work.  Each phase
plan has an Orientation section and an entry-gate checklist.
