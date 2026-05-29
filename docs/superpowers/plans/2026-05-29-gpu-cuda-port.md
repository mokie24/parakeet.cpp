# GPU (Multi-Backend: CUDA / Metal / Vulkan / HIP) Support — Full Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the entire parakeet.cpp inference pipeline (encoder, joint, AND the transducer prediction-net) on ANY ggml GPU backend (CUDA, Metal, Vulkan, HIP/ROCm, SYCL — whichever the build enables), selected automatically at runtime; and benchmark it on the GB10 Grace-Blackwell host `dgx.casa`.

**Architecture:** The work is backend-agnostic by design. Generalize the process-global `pk::Backend` to discover a GPU device via the **ggml backend registry** (`ggml_backend_dev_*`) and init it — falling back to CPU — so one code path covers every backend compiled in, with no per-backend `#ifdef`s. Upload weights into a device backend buffer (`ggml_backend_alloc_ctx_tensors`, backend-agnostic) instead of the CPU zero-copy host wrap; and replace the hand-written C++ LSTM prediction net with a ggml graph so no inference component reads host-resident weights. The decode-loop orchestration (greedy loops, argmax, token logic) stays on the host and drives per-step ggml graphs, exactly as today — only the compute backend changes. Everything is developed and parity-tested on the **CPU backend first** (device-buffer upload and the ggml LSTM both run on CPU), so enabling any GPU backend is a build flag + a registry pick.

**Backend support matrix:**
| Backend | Build flag (ours → ggml) | Test target in this plan |
|---|---|---|
| CUDA | `PARAKEET_GGML_CUDA` → `GGML_CUDA` | **dgx.casa GB10 (sm_121, CUDA 13)** — primary |
| Vulkan | `PARAKEET_GGML_VULKAN` → `GGML_VULKAN` | dgx.casa GB10 (NVIDIA Vulkan driver) — secondary, if loader present |
| Metal | `PARAKEET_GGML_METAL` → `GGML_METAL` | macOS only — **enable + document, untested here** |
| HIP/ROCm | `PARAKEET_GGML_HIP` → `GGML_HIP` | AMD GPU only — **enable + document, untested here** |

**Tech Stack:** C++17, ggml (submodule; backends via `GGML_{CUDA,METAL,VULKAN,HIP}`), the ggml backend registry, CUDA 13.0 (`/usr/local/cuda` on `dgx.casa`), GB10 Blackwell (`sm_121`), ctest parity suite vs NeMo baselines.

---

## Orientation + Entry gate

Read this whole plan. Verify the starting state:
- On `master`, `cmake --build build` is green and the parity suite passes (35 tests) with the fixture env vars set (see below).
- Confirm the facts this plan relies on (already verified, but re-check if the tree moved):
  - `src/backend.cpp` constructs the backend with `ggml_backend_cpu_init()` (hardcoded) in `Backend::Backend(int)`.
  - `src/model_loader.cpp::realize_weights(backend)` wraps the loader's host ctx memory via `ggml_backend_cpu_buffer_from_ptr` (zero-copy, CPU-only).
  - `src/prediction.cpp` implements the LSTM in plain C++ (`lstm_cell`, `step`, `forward`) reading weights through `tensor_data()` (host f32 pointers).
  - `CMakeLists.txt` already has `option(PARAKEET_GGML_CUDA ...)` forwarding to `GGML_CUDA` (so the build flag exists).
  - ggml exposes `ggml_sigmoid`, `ggml_tanh`, `ggml_get_rows` (confirmed in `third_party/ggml/include/ggml.h`).
  - `src/mel.cpp` is plain C++ (preprocessing) — it stays on the host; its output `feats` is already fed to the encoder as a graph input.

**Fixture env vars for the parity suite** (GGUFs live on `/tmp` from prior runs; regenerate via `scripts/convert_parakeet_to_gguf.py` + `scripts/gen_nemo_baseline.py` if missing):
```bash
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
export PARAKEET_TEST_BASELINE_SPEECH=/tmp/baseline_speech.gguf PARAKEET_TEST_BASELINE_TS=/tmp/baseline_ts.gguf
export PARAKEET_TEST_GGUF_EOU=/tmp/eou.gguf PARAKEET_TEST_BASELINE_EOU=/tmp/baseline_eou.gguf
export PARAKEET_TEST_BASELINE_EOU_STREAM=/tmp/baseline_eou_stream.gguf PARAKEET_TEST_GGUF_RNNT=/tmp/rnnt06.gguf
```

**Branch:** `feat/gpu-cuda`. **Conventions:** parity-first (every task keeps the CPU parity suite byte-identical green); commit per green step; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## Task G0: Backend selection via the ggml registry (any GPU backend, CPU fallback)

**Files:**
- Modify: `src/backend.cpp` (the `Backend::Backend(int)` constructor; registry-based device pick)
- Modify: `src/backend.hpp` (add `const char* device_name() const;` + a `std::string device_name_` member)
- Modify: `CMakeLists.txt` (fix the stale HIP forward; it currently sets `GGML_HIPBLAS`, but ggml's option is `GGML_HIP`)

**Why:** The backend is the single chokepoint for all compute. If it returns a GPU `ggml_backend_t`, every graph (encoder, joint, and the new LSTM) runs on the GPU with no change to the graph-building code, because `Backend::compute` is already backend-agnostic (`ggml_gallocr` + `ggml_backend_graph_compute`). Using the registry (not per-backend `#ifdef`s) means the SAME code picks CUDA, Metal, Vulkan, HIP, or SYCL — whichever was compiled in registers itself.

- [ ] **Step 1: Fix the CMake HIP forwarding (and keep the others).**

In `CMakeLists.txt`, the HIP option is stale (`GGML_HIPBLAS` no longer exists; ggml uses `GGML_HIP`). Replace:
```cmake
option(PARAKEET_GGML_HIPBLAS "Forward GGML_HIPBLAS" OFF)
...
set(GGML_HIPBLAS ${PARAKEET_GGML_HIPBLAS} CACHE BOOL "" FORCE)
```
with:
```cmake
option(PARAKEET_GGML_HIP "Forward GGML_HIP (ROCm)" OFF)
...
set(GGML_HIP ${PARAKEET_GGML_HIP} CACHE BOOL "" FORCE)
```
Leave the CUDA/METAL/VULKAN forwards as-is (they match ggml's option names).

- [ ] **Step 2: Registry-based backend selection in the constructor.**

In `src/backend.cpp`, add `#include "ggml-backend.h"` (if not already pulled in) and replace the backend-creation in the constructor. No backend-specific headers are needed — the registry is populated by whatever backends are compiled in:
```cpp
Backend::Backend(int n_threads) : impl_(new Impl()) {
    // Optional override: PARAKEET_DEVICE=cpu forces the CPU backend (used for the
    // CPU baseline on a GPU box without rebuilding).
    const char* force = std::getenv("PARAKEET_DEVICE");
    const bool force_cpu = force && std::string(force) == "cpu";

    if (!force_cpu) {
        // Pick the first GPU device the registry reports (CUDA/Metal/Vulkan/HIP/SYCL).
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                impl_->backend = ggml_backend_dev_init(dev, nullptr);
                if (impl_->backend) {
                    device_name_ = ggml_backend_dev_name(dev);
                    PK_LOG("pk::Backend using GPU device: %s", device_name_.c_str());
                    break;
                }
            }
        }
    }
    if (!impl_->backend) {                       // CPU fallback (or CPU-only build)
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
    }
    if (!impl_->backend) { PK_LOG("backend init returned null"); return; }
    set_n_threads(n_threads);   // effective only on CPU; guarded below
}
```
Add to `backend.hpp` private members: `std::string device_name_ = "cpu";` and the accessor:
```cpp
const char* device_name() const { return device_name_.c_str(); }
```
(Also add `#include <string>` and `#include <cstdlib>` to `backend.cpp` if not present.)

- [ ] **Step 3: Guard `set_n_threads` to the CPU backend.**

`ggml_backend_cpu_set_n_threads` is CPU-only; calling it on a GPU backend is invalid. In `src/backend.cpp`:
```cpp
void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : 1;
    if (impl_ && impl_->backend && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
}
```

- [ ] **Step 4: Build CPU-only (no GPU backend enabled) and confirm no behavior change.**

Run:
```bash
cmake --build build -j"$(nproc)" 2>&1 | tail -2
```
Expected: clean. On a CPU-only build the registry reports only the CPU device (no GPU device type), so the loop finds none and `device_name()` returns `"cpu"`. (The CPU backend itself is type CPU, not GPU, so it is never picked by the GPU loop — the fallback handles it.)

- [ ] **Step 5: Run the parity suite (CPU) — must stay byte-identical green.**

Run (with the fixture env vars exported):
```bash
cd build && ctest 2>&1 | grep -E "tests passed|tests failed"
```
Expected: `100% tests passed ... 35`.

- [ ] **Step 6: Commit.**
```bash
git add src/backend.cpp src/backend.hpp CMakeLists.txt
git commit -m "feat(gpu): registry-based backend selection (any GPU backend) + fix GGML_HIP forward"
```

---

## Task G1: Device-resident weights (upload when backend is not CPU)

**Files:**
- Modify: `src/model_loader.cpp` (`realize_weights`)
- Modify: `src/model_loader.hpp` (no API change; comment update)

**Why:** Today `realize_weights` wraps the host loader buffer zero-copy (`ggml_backend_cpu_buffer_from_ptr`). That only works when the compute backend reads host memory (CPU). For CUDA the weights must live in a device buffer. Keep the fast CPU zero-copy path; add a generic upload path for any non-CPU backend.

**State check vs G0:** `Backend` may now return a CUDA backend; `realize_weights(backend)` is called via `ensure_weights_realized` with `global_backend().handle()`.

- [ ] **Step 1: Branch realize_weights on backend type.**

The GGUF is loaded `no_alloc=false`, so every tensor's `->data` already points into the host `ctx_` mem buffer. For CPU keep the zero-copy wrap. For a device backend, allocate the ctx's tensors on the backend and upload from the host data. Capture host pointers BEFORE allocation (allocation reassigns `->data`).

In `src/model_loader.cpp`, replace the body of `realize_weights`:
```cpp
bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_) return true;                       // idempotent
    if(!backend || !ctx_){ PK_LOG("realize_weights: null backend/ctx"); return false; }

    if (ggml_backend_is_cpu(backend)) {
        // Fast path: borrow the host ctx memory directly (no copy).
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!weights_buf_){ PK_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for (ggml_tensor* t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t))
            t->buffer = weights_buf_;
        return true;
    }

    // Device path (e.g. CUDA): snapshot host data, allocate all ctx tensors on the
    // backend buffer (this reassigns ->data to device pointers), then upload.
    std::vector<std::pair<ggml_tensor*, const void*>> host_src;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t))
        host_src.emplace_back(t, t->data);
    weights_buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if(!weights_buf_){ PK_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& [t, src] : host_src)
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
    return true;
}
```
Add `#include "ggml-alloc.h"` and `#include <utility>`/`#include <vector>` to `model_loader.cpp` if not already present (it uses `ggml_backend_alloc_ctx_tensors`).

> Implementer note: `ggml_backend_alloc_ctx_tensors` allocates a buffer sized for all tensors in `ctx_` and assigns each tensor a slice (setting `->buffer` and `->data`). It requires the tensors to be leaves with no pre-existing backend buffer — which holds here (they were created by `gguf_init_from_file`, `->buffer == NULL`). Capturing `src=t->data` first preserves the host bytes (they live in the ctx mem buffer, untouched by the device allocation).

- [ ] **Step 2: Build + parity suite on CPU (unchanged path).**

Run:
```bash
cmake --build build -j"$(nproc)" 2>&1 | tail -2
cd build && ctest 2>&1 | grep -E "tests passed|tests failed"
```
Expected: clean build; `100% tests passed ... 35` (CPU still takes the zero-copy branch — no behavioral change).

- [ ] **Step 3: Commit.**
```bash
git add src/model_loader.cpp src/model_loader.hpp
git commit -m "feat(gpu): upload weights to a device backend buffer for non-CPU backends"
```

---

## Task G2: Port the prediction-net LSTM to a ggml graph

**Files:**
- Modify: `src/prediction.cpp` (`PredictionNet::step` and `PredictionNet::forward` to build ggml graphs; `lstm_cell` becomes a graph builder)
- Modify: `src/prediction.hpp` (no public API change; the `step`/`forward` signatures stay identical so `rnnt.cpp`/`tdt.cpp` and the pred-net cache are untouched)
- Tests: `tests/test_prediction.cpp`, `tests/test_prediction_step.cpp` (existing, run as the guard)

**Why:** With weights device-resident (G1), the C++ LSTM can no longer read `tensor->data` on the host. Reimplement the stacked-LSTM single step as a ggml graph on the persistent backend (it then runs wherever the backend runs). The greedy decode and the prediction-net output cache call `step()` unchanged.

**LSTM math (must match the current C++ exactly):** per layer `l`, given input `x[H]`, state `h_in[H]`, `c_in[H]`:
```
z = W_ih · x + b_ih + W_hh · h_in + b_hh         # z is [4H]
i = sigmoid(z[0:H]); f = sigmoid(z[H:2H]); g = tanh(z[2H:3H]); o = sigmoid(z[3H:4H])
c' = f * c_in + i * g
h' = o * tanh(c')
```
Weight tensor `weight_ih_l<l>` has ggml shape `ne = [H, 4H]` (ne[0]=in=H, ne[1]=out=4H) so `ggml_mul_mat(W_ih, x)` → `[4H]`. Same for `weight_hh_l<l>`. Biases `bias_ih_l<l>`, `bias_hh_l<l>` are `[4H]`. The embedding `decoder.prediction.embed.weight` has ne `[H, vocab+1]`.

- [ ] **Step 1: Implement `PredictionNet::step` as a ggml graph.**

Replace the C++ body of `step` (keep the signature `void step(int32_t token_id, bool is_sos, const PredState& in, std::vector<float>& g, PredState& out_state) const`). Build one graph that runs all `L` layers and outputs the top-layer `h'` plus every layer's `h'`/`c'` (captured for the next step's state). Use `pk::run_graph` + `pk::capture_graph_output` for the per-layer state, mirroring how the fused encoder captures intermediates.

```cpp
void PredictionNet::step(int32_t token_id, bool is_sos, const PredState& in,
                         std::vector<float>& g, PredState& out_state) const {
    const int H = H_, L = n_layers_;
    out_state.h.assign(L, std::vector<float>(H));
    out_state.c.assign(L, std::vector<float>(H));

    // Host-side input embedding for layer 0 (SOS -> zeros).
    std::vector<float> x0(H, 0.0f);
    if (!is_sos) {
        // embed row lookup done on host to keep the graph input small & simple.
        const ggml_tensor* emb = ml_.tensor("decoder.prediction.embed.weight");
        // emb is device-resident now; read the row via a tiny ggml get_rows graph
        // OR (simpler) keep a host copy of the embedding (see Step 1b).
    }

    bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                              (int64_t[]){H}, x0.data(), (size_t)H*sizeof(float));
        ggml_tensor* layer_in = x;
        ggml_tensor* top_h = nullptr;
        for (int l = 0; l < L; ++l) {
            const std::string s = "_l" + std::to_string(l);
            ggml_tensor* Wih = pk::clone_weight(ctx, ml_, ("decoder.prediction.dec_rnn.lstm.weight_ih"+s).c_str());
            ggml_tensor* Whh = pk::clone_weight(ctx, ml_, ("decoder.prediction.dec_rnn.lstm.weight_hh"+s).c_str());
            ggml_tensor* bih = pk::clone_weight(ctx, ml_, ("decoder.prediction.dec_rnn.lstm.bias_ih"+s).c_str());
            ggml_tensor* bhh = pk::clone_weight(ctx, ml_, ("decoder.prediction.dec_rnn.lstm.bias_hh"+s).c_str());
            ggml_tensor* h_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                    (int64_t[]){H}, in.h[l].data(), (size_t)H*sizeof(float));
            ggml_tensor* c_in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                    (int64_t[]){H}, in.c[l].data(), (size_t)H*sizeof(float));
            ggml_tensor* z = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, Wih, layer_in), bih),
                                          ggml_add(ctx, ggml_mul_mat(ctx, Whh, h_in),     bhh)); // [4H]
            ggml_tensor* zi = ggml_view_1d(ctx, z, H, 0);
            ggml_tensor* zf = ggml_view_1d(ctx, z, H, (size_t)H*sizeof(float));
            ggml_tensor* zg = ggml_view_1d(ctx, z, H, (size_t)2*H*sizeof(float));
            ggml_tensor* zo = ggml_view_1d(ctx, z, H, (size_t)3*H*sizeof(float));
            ggml_tensor* i = ggml_sigmoid(ctx, ggml_cont(ctx, zi));
            ggml_tensor* f = ggml_sigmoid(ctx, ggml_cont(ctx, zf));
            ggml_tensor* gg= ggml_tanh   (ctx, ggml_cont(ctx, zg));
            ggml_tensor* o = ggml_sigmoid(ctx, ggml_cont(ctx, zo));
            ggml_tensor* c_out = ggml_add(ctx, ggml_mul(ctx, f, c_in), ggml_mul(ctx, i, gg));
            ggml_tensor* h_out = ggml_mul(ctx, o, ggml_tanh(ctx, c_out));
            pk::capture_graph_output(c_out, &out_state.c[l]);
            pk::capture_graph_output(h_out, &out_state.h[l]);
            layer_in = h_out; top_h = h_out;
        }
        return top_h;   // g
    }, g, /*graph_nodes*/ 256);
    assert(ok && "pred-net step graph failed");
}
```

- [ ] **Step 1b: Resolve the embedding lookup.**

The embedding is one row per token. Two options — pick the simpler that keeps parity:
  - **Option A (recommended): host copy of the embedding.** In the `PredictionNet` constructor, copy `decoder.prediction.embed.weight` into a `std::vector<float> embed_host_` ONCE (it is small: `(vocab+1)*H` f32). Then `step` fills `x0` by `memcpy` from `embed_host_` (host), and the graph just consumes `x0`. This avoids a device round-trip for a trivial gather and is the least-risk for parity. Add `std::vector<float> embed_host_;` to `prediction.hpp` and populate it in the ctor with `ggml_backend_tensor_get` (works for both CPU and device backends) — but the ctor runs before `realize_weights`, so read it from the host ctx data directly: copy from `tensor("...embed.weight")->data` BEFORE weights are uploaded. Since `PredictionNet` is constructed per-transcribe AFTER weights may be realized, instead copy via `ggml_backend_tensor_get(emb, embed_host_.data(), 0, nbytes)` which is backend-safe.
  - Option B: `ggml_get_rows(emb, ids)` inside the graph. More "pure" but adds an int input tensor and a device gather for one row.

Use Option A.

- [ ] **Step 2: Reimplement `PredictionNet::forward` (used by `test_prediction`) on top of `step`.**

`forward` builds the full sequence output. Reimplement it as a loop over `step` carrying state (it already has the same recurrence), so there is ONE LSTM implementation:
```cpp
void PredictionNet::forward(const std::vector<int32_t>& ids, bool add_sos,
                            std::vector<float>& out, int& U_out, int& hidden) const {
    const int H = H_;
    const int seq = add_sos ? (int)ids.size() + 1 : (int)ids.size();
    U_out = seq; hidden = H;
    out.assign((size_t)seq * H, 0.0f);
    PredState st = zero_state();
    std::vector<float> g; PredState nxt;
    for (int t = 0; t < seq; ++t) {
        bool is_sos = add_sos && t == 0;
        int32_t tok = is_sos ? -1 : ids[add_sos ? t - 1 : t];
        step(tok, is_sos, st, g, nxt);
        std::memcpy(&out[(size_t)t * H], g.data(), (size_t)H * sizeof(float));
        st = nxt;
    }
}
```

- [ ] **Step 3: Build + run the prediction parity tests (CPU).**

Run:
```bash
cmake --build build -j"$(nproc)" 2>&1 | tail -2
cd build && ctest -R "test_prediction|test_prediction_step|test_joint|test_transducer_core" --output-on-failure 2>&1 | tail -15
```
Expected: all PASS. These compare the prediction-net output against the NeMo baseline; the ggml LSTM must match within the test's f32 tolerance. If a test fails on tolerance, check the gate order (i,f,g,o slices) and that `weight_ih`/`weight_hh` ne dims map as `[in, out]`.

- [ ] **Step 4: Run the FULL transducer parity suite (the real guard).**

Run:
```bash
cd build && ctest -R "test_transcribe_rnnt|test_transcribe_tdt|test_transcribe_eou|test_tdt_greedy|test_streaming_decode|test_transcribe_0_6b" --output-on-failure 2>&1 | tail -20
```
Expected: all PASS (byte-identical transcripts vs NeMo). This proves the ggml LSTM reproduces the decode exactly.

- [ ] **Step 5: CPU performance sanity check (no major regression).**

The pred-net output cache keeps `step` calls low (~U per utterance), so per-step ggml overhead should be negligible. Confirm:
```bash
./build/examples/cli/parakeet-cli bench --model /tmp/rnnt06.gguf --manifest benchmarks/librispeech_manifest.tsv --threads 8 --json /tmp/g2.json
.venv/bin/python -c "import json;d=json.load(open('/tmp/g2.json'));fs=d['files'];print('RTFx',sum(f['audio_sec'] for f in fs)/(sum(f['proc_ms'] for f in fs)/1000.0))"
```
Expected: RTFx within ~10% of the pre-G2 number (rnnt-0.6b ≈ 34–35 on the x86 box). If it regressed materially, keep the C++ `lstm_cell` as a CPU-only fast path selected when `global_backend().device_name()=="cpu"`, and use the ggml path otherwise.

- [ ] **Step 6: Commit.**
```bash
git add src/prediction.cpp src/prediction.hpp
git commit -m "feat(gpu): port prediction-net LSTM to a ggml graph (device-resident weights)"
```

---

## Task G3: GPU builds (CUDA on dgx.casa GB10; document Metal/Vulkan/HIP)

**Files:** none in-repo (build invocation only). The CUDA build runs ON `dgx.casa`.

**State check:** G0–G2 are merged to `master` (or pushed on `feat/gpu-cuda`) and the CPU parity suite is green. Because backend selection is registry-based, NO source differs per backend — only the build flag.

**Per-backend build flags (for reference / docs):**
```bash
# CUDA  (NVIDIA)         — tested here on GB10
cmake -B build-cuda   -DPARAKEET_GGML_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES=121 ...
# Vulkan (cross-vendor)  — secondary test on GB10 if the Vulkan loader is present
cmake -B build-vulkan -DPARAKEET_GGML_VULKAN=ON ...
# Metal  (Apple)         — build on macOS only
cmake -B build-metal  -DPARAKEET_GGML_METAL=ON ...
# HIP    (AMD ROCm)      — build on an AMD/ROCm host only
cmake -B build-hip    -DPARAKEET_GGML_HIP=ON   -DCMAKE_HIP_ARCHITECTURES=<gfxNNNN> ...
```
Only the CUDA build is exercised in this plan (that's the hardware we have). Metal/HIP are enable-and-document (verify they at least configure where the SDK exists; full testing is out of scope without the hardware). If the GB10 Vulkan loader is present, do a quick Vulkan parity smoke as a bonus (Step 6).

- [ ] **Step 1: Clone the repo and its submodule on dgx.casa.**
```bash
ssh dgx.casa 'git clone --recurse-submodules git@github.com:mudler/parakeet.cpp.git ~/_git/parakeet.cpp || (cd ~/_git/parakeet.cpp && git fetch && git checkout feat/gpu-cuda && git pull && git submodule update --init --recursive)'
```

- [ ] **Step 2: Configure with CUDA 13 + GB10 arch.**
```bash
ssh dgx.casa 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/_git/parakeet.cpp && cmake -B build-cuda -DPARAKEET_GGML_CUDA=ON -DPARAKEET_BUILD_CLI=ON -DPARAKEET_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=121 2>&1 | tail -25'
```
Expected: configures, finds CUDA 13, `GGML_CUDA` ON. (`121` = GB10 Blackwell sm_121; if nvcc rejects it, try `native` or `120`.)

- [ ] **Step 3: Build.**
```bash
ssh dgx.casa 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/_git/parakeet.cpp && cmake --build build-cuda -j"$(nproc)" 2>&1 | tail -20'
```
Expected: builds `parakeet-cli` and tests with the CUDA backend. Fix any aarch64/CUDA-13 compile errors (likely arch flags or a host-compiler flag).

- [ ] **Step 4: Smoke test on GPU (confirm it actually uses CUDA).**

Convert one model on the box (CUDA 13 toolkit + a Python env with the converter deps, OR scp a GGUF from the dev box), then:
```bash
ssh dgx.casa 'cd ~/_git/parakeet.cpp && ./build-cuda/examples/cli/parakeet-cli info /tmp/pk110m.gguf 2>&1 | head; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```
Add a one-line stderr log of `global_backend().device_name()` in the CLI startup (or rely on the `PK_LOG` from G0) to confirm `cuda:0`. Run a transcription and watch `nvidia-smi` show GPU memory/util.

- [ ] **Step 5: Commit any build fixes** (CMake/source) and push.
```bash
git add -A && git commit -m "build(gpu): CUDA 13 build on GB10 (sm_121) fixes" && git push
```

- [ ] **Step 6 (bonus): Vulkan smoke on GB10, if a Vulkan loader is present.**

Vulkan is cross-vendor and the NVIDIA driver ships a Vulkan ICD, so GB10 can double as a Vulkan test. Check + build + transcribe:
```bash
ssh dgx.casa 'vulkaninfo --summary 2>/dev/null | head || echo "no vulkan loader"'
ssh dgx.casa 'cd ~/_git/parakeet.cpp && cmake -B build-vulkan -DPARAKEET_GGML_VULKAN=ON -DPARAKEET_BUILD_CLI=ON 2>&1 | tail -5 && cmake --build build-vulkan -j"$(nproc)" 2>&1 | tail -3'
ssh dgx.casa 'cd ~/_git/parakeet.cpp && ./build-vulkan/examples/cli/parakeet-cli bench --model /tmp/pk110m.gguf --manifest <manifest> --json /tmp/vk.json 2>&1 | tail -3'
```
Expected: `device_name()` reports the Vulkan device and a transcript matches the CPU/CUDA output. If the loader is absent or it fails, note it and move on (Vulkan is a bonus, not a gate).

---

## Task G4: GPU benchmark on dgx.casa (free the GPU first)

**Files:** none in-repo. Runs on `dgx.casa`.

**State check:** G3 produced a working CUDA `parakeet-cli` at `~/_git/parakeet.cpp/build-cuda`.

- [ ] **Step 1: Free the GPU — stop the LocalAI worker (REVERSIBLE).**
```bash
ssh dgx.casa 'docker stop local-ai && docker ps --format "{{.Names}} {{.Status}}"'
```
Expected: `local-ai` no longer listed as running. (`restart=always` does not resurrect a manual stop; `docker start local-ai` brings it back in Step 4.)

- [ ] **Step 2: Prepare audio + GGUFs on the box.**

Copy the LibriSpeech manifest + audio and the diverse clips from the dev box (or regenerate via `scripts/bench_data.py`), and convert the model GGUFs for the dtypes to benchmark (reuse `scripts/benchmark.py`'s conversion, or `scp` the f32 GGUFs and quantize on-box with `parakeet-cli quantize`). Keep paths matching a manifest TSV.

- [ ] **Step 3: Run our engine on the GPU.**

Use the same `bench` subcommand; it auto-selects the CUDA backend (built in). For a same-box CPU-vs-GPU comparison, ALSO build a CPU `build/` on the GB10 and run both:
```bash
ssh dgx.casa 'cd ~/_git/parakeet.cpp && for m in <gguf list>; do ./build-cuda/examples/cli/parakeet-cli bench --model $m --manifest <manifest> --json results_gpu/$(basename $m).json; done'
```
Record RTFx per model/dtype. (Threads are irrelevant on CUDA; the Grace CPU run uses `--threads` for the CPU baseline.)

- [ ] **Step 4: Restore the LocalAI worker.**
```bash
ssh dgx.casa 'docker start local-ai && docker ps --format "{{.Names}} {{.Status}}"'
```
Expected: `local-ai` back `Up ... (healthy)`. **Do this even if the benchmark fails.**

- [ ] **Step 5: Pull the GPU results back to the dev box** (`scp` the JSONs into a new `benchmarks/results_gpu/` or a `device:"cuda"` field) and commit the raw numbers.

---

## Task G5: Update benchmarks with the GPU dimension

**Files:**
- Modify: `scripts/plot_benchmark.py`, `scripts/gen_benchmark_md.py` (add a device dimension / GPU section)
- Modify: `benchmarks/BENCHMARK.md` (regenerated)
- Add: `benchmarks/results_gpu/*.json` (or extend the existing JSONs with a `cuda` block)

**Why:** Present GB10 GPU vs Grace-CPU vs the x86-CPU/NeMo reference so the GPU win is visible.

- [ ] **Step 1: Decide the comparison axis.** Recommended: a same-box GB10 plot — **ours-GPU vs ours-CPU (Grace)** RTFx per model (clean apples-to-apples on one machine), plus a note that the x86 table/NeMo numbers are a separate host. If NeMo installs cleanly on dgx.casa, add **NeMo-GPU** as a stretch (otherwise omit — do not mix NeMo-CPU-x86 with ours-GPU-GB10 as a "speedup").

- [ ] **Step 2: Add a `plot_gpu.py` chart (or extend `plot_benchmark.py`)** that reads `benchmarks/results_gpu/*.json` and renders ours-GPU vs ours-CPU RTFx per model. Keep the existing CPU plots unchanged.

- [ ] **Step 3: Add a "GPU (GB10 Grace-Blackwell)" section to BENCHMARK.md** via `gen_benchmark_md.py`: a table (model, ours-CPU-Grace RTFx, ours-GPU RTFx, GPU speedup) + the new plot + the machine spec (GB10, CUDA 13, unified memory). Note the decode is autoregressive/latency-bound and that CUDA-graph capture is the future lever.

- [ ] **Step 4: Commit.**
```bash
git add scripts/ benchmarks/
git commit -m "bench(gpu): add GB10 GPU vs CPU results, plots, and write-up"
```

---

## Done-when
- `pk::Backend` selects a GPU backend via the ggml registry when one is built in (CUDA/Metal/Vulkan/HIP/SYCL), honors `PARAKEET_DEVICE=cpu`, and falls back to CPU otherwise; CPU build + parity suite unchanged (35/35 byte-identical). The HIP forward (`GGML_HIP`) is fixed.
- Weights upload to a device buffer on non-CPU backends; the prediction-net LSTM, encoder, and joint all run on the active backend — one code path, no per-backend `#ifdef`s.
- A CUDA build runs on GB10 (`dgx.casa`) and transcribes using the GPU (confirmed via `nvidia-smi` + `device_name()` reporting the GPU). Vulkan smoke on GB10 is a bonus if the loader is present.
- Metal and HIP are enabled in the build system and documented (configure-checked where an SDK exists); full Metal/HIP testing is out of scope without the hardware.
- GPU benchmark numbers collected with `local-ai` stopped then restarted; BENCHMARK.md gains a GB10 GPU-vs-CPU section with plots.

## Risks / notes
- **Parity tolerance for the ggml LSTM (G2):** f32 ggml matmul vs the C++ scalar loop can differ in the last bits; the transducer greedy is argmax-based so it should be byte-identical, but if a token flips, widen the per-frame check or keep the CPU C++ path for CPU (Step 5 fallback).
- **Per-step latency on GPU:** the decode launches a tiny graph per step (pred + joint). On GB10 unified memory the H2D/D2H of small state/logits is cheap, but kernel-launch overhead dominates batch-1 greedy. The real GPU lever (future work) is CUDA-graph capture/replay of the per-step graph — out of scope here.
- **NeMo-GPU comparison** is optional; do not present a cross-host ratio as a speedup.
- **Always restart `local-ai`** (G4 Step 4) even on failure.
