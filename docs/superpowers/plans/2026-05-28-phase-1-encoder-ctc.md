# parakeet.cpp Phase 1 — Preprocessing + FastConformer encoder + CTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the deterministic inference path — log-mel preprocessing → dw_striding subsampling → FastConformer encoder → CTC head → greedy CTC decode → detokenized text — with per-stage tensor parity against the NeMo baseline, ending in correct end-to-end CTC transcription via `parakeet-cli transcribe`.

**Architecture:** Each component is a focused translation unit with a public forward function and a ctest that diffs its output against the matching tensor in `baseline.gguf` (produced by Phase 0's `gen_nemo_baseline.py`). FFT and feature normalization run in plain C++ outside the ggml graph; everything from subsampling onward is a ggml compute graph (B=1). Tensor names in the GGUF are the verbatim NeMo `state_dict` keys, so weights are fetched by their NeMo names.

**Tech Stack:** C++17, ggml, the `.venv` NeMo env (for regenerating baselines), ctest. Anchor checkpoint `nvidia/parakeet-tdt_ctc-110m` (CTC head). Reference math source: `/home/mudler/_git/NeMo`.

**Reference (read before starting):**
- Spec `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md` (§2 architecture, §2.1 validated config, §9 parity strategy).
- Phase 0 plan `docs/superpowers/plans/2026-05-28-phase-0-foundation.md` — defines `pk::ModelLoader`, `pk::ParakeetConfig`, `pk::Audio`, the GGUF schema, and `baseline.gguf` contents.
- NeMo math (trace exact lines as you implement):
  - Preprocessing: `nemo/collections/asr/parts/preprocessing/features.py` → `FilterbankFeatures.forward`
  - Subsampling: `nemo/collections/asr/parts/submodules/subsampling.py` → `ConvSubsampling` (`dw_striding`)
  - Conformer layer: `nemo/collections/asr/parts/submodules/conformer_modules.py` → `ConformerLayer.forward`, `ConformerConvolution`, `ConformerFeedForward`
  - Attention: `nemo/collections/asr/parts/submodules/multi_head_attention.py` → `RelPositionMultiHeadAttention`, `RelPositionalEncoding`, `rel_shift`
  - CTC head: `nemo/collections/asr/modules/conv_asr.py` → `ConvASRDecoder`

**Conventions:** same as Phase 0 (ctest returns 0/77/fail, commit per green step, CPU-only, no skipped hooks).

---

## Orientation — run BEFORE any task (fresh sub-agent, zero context)

1. **Read the spec and the Phase 0 plan** (paths above). The spec is the source of truth; this plan implements its Phase 1.
2. **Verify the env** (prerequisites, not tasks): `git rev-parse --is-inside-work-tree`→true; `.venv/bin/python -c "import nemo,gguf,torch"` works; `/home/mudler/_git/NeMo` exists.
3. **Find your resume point:** `git log --oneline -25`. Each task ends in a commit with a recognizable message. Start at the first incomplete task; re-run its verification first.

## Entry gate — Phase 0 must be complete (do not start otherwise)

Run and confirm all succeed (skips with 77 are only acceptable where a model is genuinely unavailable — here the model IS available, so expect real passes):
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure -LE model            # test_smoke, test_audio_io
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf ctest --test-dir build -R test_model_loader --output-on-failure
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
```
If any fails, STOP and finish Phase 0 first. Export for this session:
```bash
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf
export PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
```

---

## File structure created in this phase

```
src/
  fft.hpp / fft.cpp               # radix-2 real FFT (n_fft=512)
  mel.hpp / mel.cpp               # log-mel front end (plain C++)
  ggml_graph.hpp / ggml_graph.cpp # tiny helper: build+alloc+compute a graph on CPU
  subsampling.hpp / subsampling.cpp
  relpos_attention.hpp / relpos_attention.cpp
  conformer.hpp / conformer.cpp
  encoder.hpp / encoder.cpp
  ctc_decoder.hpp / ctc_decoder.cpp
  tokenizer.hpp / tokenizer.cpp
  search.hpp / search.cpp         # CTC greedy collapse
  parakeet.cpp                    # add pk::transcribe() orchestrator (extend public API)
include/parakeet.h                # add transcribe entry point
examples/cli/main.cpp             # add `transcribe` subcommand
tests/
  parity.hpp                      # shared baseline-compare helper
  test_mel.cpp
  test_subsampling.cpp
  test_relpos_attention.cpp
  test_conformer.cpp
  test_encoder.cpp
  test_ctc.cpp
  test_tokenizer.cpp
  test_transcribe.cpp
scripts/gen_nemo_baseline.py      # EXTEND: dump pos_emb + layer-0 sub-tensors + tokenizer probe
```

---

## Task 1: Shared parity helper + ggml graph helper

**Files:** Create `tests/parity.hpp`, `src/ggml_graph.hpp`, `src/ggml_graph.cpp`. Modify `CMakeLists.txt`.

- [ ] **Step 1: `tests/parity.hpp` — load a baseline tensor + compare**

```cpp
#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace pktest {
// Load an f32 tensor (flattened, row-major) by name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ false, &ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) { gguf_free(g); ggml_free(ctx); return false; }
    shape.clear();
    for (int i = ggml_n_dims(t)-1; i >= 0; --i) shape.push_back(t->ne[i]); // report outer..inner
    size_t n = ggml_nelements(t);
    out.resize(n);
    memcpy(out.data(), t->data, n*sizeof(float));
    gguf_free(g); ggml_free(ctx);
    return true;
}
// Compare; returns true if within tolerance. Prints max/mean abs diff.
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n", label, got.size(), ref.size());
        return false;
    }
    double maxabs=0, sumabs=0; size_t worst=0;
    for (size_t i=0;i<got.size();++i){ double d=std::fabs((double)got[i]-ref[i]);
        sumabs+=d; if(d>maxabs){maxabs=d; worst=i;} }
    double mean = sumabs/(got.size()?got.size():1);
    bool ok = true;
    for (size_t i=0;i<got.size() && ok;++i){ double tol=atol+rtol*std::fabs((double)ref[i]);
        if (std::fabs((double)got[i]-ref[i])>tol) ok=false; }
    std::fprintf(stderr, "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, got.size(), maxabs, mean, worst, got[worst], ref[worst], ok?"OK":"FAIL");
    return ok;
}
} // namespace pktest
```

- [ ] **Step 2: `src/ggml_graph.hpp` — one-shot CPU graph runner**

```cpp
#pragma once
#include <functional>
struct ggml_context; struct ggml_tensor; struct ggml_cgraph;
namespace pk {
// Allocates a compute context of mem_bytes, lets `build` create the graph and
// return the output tensor, runs it on CPU with n_threads, and copies the
// output into `out` (caller sizes it). Returns false on failure.
bool run_graph(size_t mem_bytes, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out);
}
```
Implement in `src/ggml_graph.cpp` using `ggml_init` (no_alloc=false), `ggml_new_graph`, `ggml_build_forward_expand`, `ggml_graph_compute_with_ctx` (or the backend-cpu scheduler — match whatever the installed ggml exposes; check `third_party/ggml/include/ggml-cpu.h`). Copy `ggml_get_data_f32(out_tensor)` into `out`.

- [ ] **Step 3: Build (no test yet — these are libraries used by later tasks)**

Add `src/ggml_graph.cpp` to `PARAKEET_SRC`. Build to confirm it compiles:
```bash
cmake --build build -j
```
Expected: compiles clean.

- [ ] **Step 4: Commit**

```bash
git add tests/parity.hpp src/ggml_graph.* CMakeLists.txt
git commit -m "feat: parity-compare helper + one-shot ggml CPU graph runner"
```

---

## Task 2: FFT (radix-2, real input, n_fft=512)

**Files:** Create `src/fft.hpp`, `src/fft.cpp`, `tests/test_fft.cpp`. Modify CMake.

- [ ] **Step 1: Write the failing test (FFT of a known signal)**

`tests/test_fft.cpp`:
```cpp
#include "fft.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
int main() {
    const int N=512;
    std::vector<float> x(N);
    // pure cosine at bin k=8: magnitude should peak at bin 8
    for (int i=0;i<N;++i) x[i]=std::cos(2.0*M_PI*8*i/N);
    std::vector<float> re(N/2+1), im(N/2+1);
    pk::rfft(x, re, im);
    // find peak bin
    int peak=0; double best=-1;
    for (int k=0;k<=N/2;++k){ double m=re[k]*re[k]+im[k]*im[k]; if(m>best){best=m;peak=k;} }
    if (peak!=8){ std::fprintf(stderr,"peak at %d, expected 8\n",peak); return 1; }
    std::printf("fft ok: peak bin=%d\n", peak);
    return 0;
}
```

- [ ] **Step 2: Interface**

`src/fft.hpp`:
```cpp
#pragma once
#include <vector>
namespace pk {
// Real FFT of length n (n must be power of 2). Fills re/im for bins [0, n/2].
void rfft(const std::vector<float>& in, std::vector<float>& re, std::vector<float>& im);
}
```

- [ ] **Step 3: Implementation (iterative radix-2 Cooley-Tukey)**

`src/fft.cpp`: standard in-place complex radix-2 FFT on a complex buffer built from the real input (imag=0), bit-reversal permutation + butterfly stages, then copy bins `0..n/2` to `re/im`. Reference any standard iterative FFT; verify against the test. Keep it self-contained (no FFTW).

- [ ] **Step 4: Build + run**

```bash
cmake --build build -j && ctest --test-dir build -R test_fft --output-on-failure
```
Expected: PASS, `fft ok: peak bin=8`.

- [ ] **Step 5: Commit** — `git add src/fft.* tests/test_fft.cpp CMakeLists.txt tests/CMakeLists.txt && git commit -m "feat: radix-2 real FFT"`

---

## Task 3: Log-mel preprocessing (parity vs baseline `mel`)

**Files:** Create `src/mel.hpp`, `src/mel.cpp`, `tests/test_mel.cpp`. Modify CMake.

**Exact NeMo math to match** (trace `FilterbankFeatures.forward`; values from GGUF KV):
1. `dither=0` at inference (skip).
2. **Preemphasis** (if `preemph>0`): `x[t] = x[t] - preemph * x[t-1]` for t≥1, `x[0]` unchanged.
3. **STFT** with `center=True`: reflect-pad the signal by `n_fft//2` on both ends, then frame with `win_length=400`, `hop_length=160`, apply the **Hann window** (lift `preprocessor.featurizer.window` from the GGUF — do NOT recompute), FFT `n_fft=512` → keep bins `0..256` (257 bins).
4. **Power spectrum**: `|X|^mag_power` with `mag_power=2.0` → `re^2+im^2`.
5. **Mel projection**: `mel = fb @ power`, where `fb` is `preprocessor.featurizer.fb` from the GGUF, shape `[n_mels=80, 257]`. Result `[80, T]`.
6. **Log**: `log(mel + log_zero_guard)` (guard add type; `log_zero_guard≈2^-24`). Match NeMo: it uses `torch.log(x + guard)`.
7. **Per-feature normalization** (`normalize=="per_feature"`): for each mel bin (row), subtract its mean over time and divide by its std (NeMo uses unbiased std with the seq mask; B=1 full length → plain mean/std over T). Match `normalize_batch` in features.py exactly (it adds a tiny epsilon to std).

- [ ] **Step 1: Write the failing parity test**

`tests/test_mel.cpp`:
```cpp
#include "mel.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "parity.hpp"
#include <cstdlib>
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if(!gguf||!base){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    pk::Audio a; if(!pk::load_audio_16k_mono("tests/fixtures/clip.wav", a)) return 1;
    pk::MelFrontend mel(ml);            // pulls fb/window/config from loader
    std::vector<float> feats; int n_mels=0, T=0;
    mel.compute(a.samples, feats, n_mels, T);   // feats row-major [n_mels, T]
    std::vector<float> ref; std::vector<int64_t> shape;
    if(!pktest::load_baseline(base, "mel", ref, shape)) return 1;
    // baseline "mel" is [n_mels, T] (squeezed). Sizes must match.
    bool ok = pktest::compare(feats, ref, "mel", /*atol*/1e-2f, /*rtol*/1e-2f);
    return ok?0:1;
}
```

- [ ] **Step 2: Interface**

`src/mel.hpp`:
```cpp
#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {
class MelFrontend {
public:
    explicit MelFrontend(const ModelLoader& ml);
    // samples: 16k mono. Out: feats row-major [n_mels, T] (frame-major inner = T).
    void compute(const std::vector<float>& samples, std::vector<float>& feats, int& n_mels, int& T) const;
private:
    int n_fft_, win_, hop_, n_mels_, n_bins_;
    float preemph_, mag_power_, log_guard_;
    bool per_feature_;
    std::vector<float> window_;  // [win]
    std::vector<float> fb_;      // [n_mels, n_bins]
};
}
```

- [ ] **Step 3: Implementation** — follow the 7 steps above using `pk::rfft`. Read `window_`/`fb_` from `ml.tensor("preprocessor.featurizer.window")` / `...fb"` (copy their f32 data). Read scalar params from `ml.config()`.

- [ ] **Step 4: Build + run**

```bash
cmake --build build -j && PARAKEET_TEST_GGUF=$PARAKEET_TEST_GGUF PARAKEET_TEST_BASELINE=$PARAKEET_TEST_BASELINE ctest --test-dir build -R test_mel --output-on-failure
```
Expected: `[mel] … OK`, PASS. If FAIL: the printed worst index + first-divergence usually points to a specific step. Most common culprits: center-padding (frame count off by 1), window not lifted from GGUF, normalization formula, or `mag_power`. Add temporary stage dumps and compare against an ad-hoc Python dump of `FilterbankFeatures` internals if needed.

- [ ] **Step 5: Commit** — `git add src/mel.* tests/test_mel.cpp CMakeLists.txt tests/CMakeLists.txt && git commit -m "feat: log-mel front end with parity vs NeMo"`

---

## Task 4: Extend baseline dumper for fine-grained encoder parity

**Files:** Modify `scripts/gen_nemo_baseline.py`.

To make the attention/conformer parity tests debuggable, dump more tensors.

- [ ] **Step 1: Add hooks** for: the relative positional encoding output `pos_emb` (hook `m.encoder.pos_enc`), and layer-0 sub-module outputs by hooking `m.encoder.layers[0].self_attn` (→ `l0_attn_out`), `m.encoder.layers[0].conv` (→ `l0_conv_out`). Also dump the **subsampling-then-pos-encode input** to layer 0 as `enc_pre_layers` (the tensor fed into `layers[0]`, captured via a forward pre-hook on `m.encoder.layers[0]`). Store all squeezed f32.

- [ ] **Step 2: Regenerate + verify** the new tensors appear:
```bash
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
.venv/bin/python -c "import gguf;print(sorted(t.name for t in gguf.GGUFReader('/tmp/baseline.gguf').tensors))"
```
Expected: includes `pos_emb`, `l0_attn_out`, `l0_conv_out`, `enc_pre_layers` alongside the Phase 0 tensors.

- [ ] **Step 3: Commit** — `git add scripts/gen_nemo_baseline.py && git commit -m "feat: dump pos_emb + layer-0 sub-tensors for encoder parity"`

---

## Task 5: Subsampling (dw_striding ÷8) — parity vs `subsampling_out`

**Files:** Create `src/subsampling.{hpp,cpp}`, `tests/test_subsampling.cpp`. Modify CMake.

**NeMo math** (`ConvSubsampling`, `dw_striding`): input treated as `[B=1, 1, T, feat=80]`. Stage 1: Conv2d(1→conv_channels, k=3, stride=2, pad=1) + ReLU. Stages 2..(log2(factor)-1): depthwise Conv2d(ch→ch, k=3, stride=2, pad=1, groups=ch) + pointwise Conv2d(ch→ch, k=1) + ReLU. After 3 stages T→T/8, feat 80→10. Then flatten channels×freq and apply linear `pre_encode.out` (`[d_model, conv_channels*reduced_feat]`) → `[T/8, d_model]`. Weight names: `encoder.pre_encode.conv.*` (the conv stack) and `encoder.pre_encode.out.{weight,bias}`. Inspect the real shapes via `parakeet-cli info` and a quick `gguf` dump to map each conv index.

- [ ] **Step 1: Failing parity test** `tests/test_subsampling.cpp`: load baseline `mel` as input, run `pk::Subsampling::forward`, compare output to baseline `subsampling_out` (atol 1e-2, rtol 1e-2). (Mirror the structure of `test_mel.cpp`.)
- [ ] **Step 2: Interface** `src/subsampling.hpp`: `class Subsampling { Subsampling(const ModelLoader&); void forward(const std::vector<float>& mel, int n_mels, int T, std::vector<float>& out, int& Tout, int& d_model) const; };` (builds a ggml graph via `pk::run_graph`).
- [ ] **Step 3: Implement** using `ggml_conv_2d` / depthwise (`ggml_conv_2d_dw` if available; else depthwise via grouped conv), `ggml_relu`, reshape/permute, and the final `ggml_mul_mat` + bias for `pre_encode.out`. Map each `encoder.pre_encode.conv.N.*` weight by inspecting shapes.
- [ ] **Step 4: Build + run** `ctest -R test_subsampling`. Expected OK. Debug tip: if shapes mismatch, dump the conv weight shapes from the GGUF and confirm the Conv2d in/out/groups per stage.
- [ ] **Step 5: Commit** — `git commit -m "feat: dw_striding subsampling with parity"`

---

## Task 6: Relative-position attention — parity vs `l0_attn_out`

**Files:** Create `src/relpos_attention.{hpp,cpp}`, `tests/test_relpos_attention.cpp`. Modify CMake.

**NeMo math** (`RelPositionMultiHeadAttention.forward`, verbatim tensor suffixes under `encoder.layers.0.self_attn.`): `linear_q,linear_k,linear_v,linear_out,linear_pos` (linear_pos has no bias) and parameters `pos_bias_u,pos_bias_v` (shape `[n_heads, d_head]`). Algorithm:
```
q = linear_q(x); k = linear_k(x); v = linear_v(x)          # [T, d_model] -> heads [H, T, dk]
p = linear_pos(pos_emb)                                      # [2T-1, d_model] -> [H, 2T-1, dk]
q_u = q + pos_bias_u ; q_v = q + pos_bias_v                  # broadcast per head
ac  = q_u @ k^T                                              # [H, T, T]
bd  = q_v @ p^T                                              # [H, T, 2T-1]
bd  = rel_shift(bd)                                          # -> [H, T, T]  (Transformer-XL shift)
scores = (ac + bd) / sqrt(dk)                                # no mask for offline full-context
attn   = softmax(scores) @ v                                 # [H, T, dk]
out    = linear_out(concat_heads(attn))                      # [T, d_model]
```
`rel_shift` (match `multi_head_attention.py` exactly): pad last dim by 1 at the front, reshape `[H, 2T, T]`→slice→reshape back to `[H, T, 2T-1]`, take first `T` columns. Implement with `ggml_pad`/`ggml_view`/`ggml_reshape` to reproduce the same indexing.

- [ ] **Step 1: Failing parity test**: input = baseline `enc_pre_layers` after the layer's `norm_self_att` (NOTE: the baseline `l0_attn_out` is the self_attn module output, whose input is `norm_self_att(layer_input)`. To isolate attention, apply `norm_self_att` to `enc_pre_layers` first using that layer's `encoder.layers.0.norm_self_att.{weight,bias}` LayerNorm, then run attention, then compare to `l0_attn_out`). Also load `pos_emb` from baseline. atol/rtol 2e-2 (attention accumulates more error).
- [ ] **Step 2: Interface** `src/relpos_attention.hpp`: a function that, given the loader, a layer index, an input `[T, d_model]` tensor and `pos_emb`, returns the attention output `[T, d_model]`.
- [ ] **Step 3: Implement** the graph. Get `n_heads`, `d_head=d_model/n_heads` from config.
- [ ] **Step 4: Build + run** `ctest -R test_relpos_attention`. Expected OK. Debug: if `ac` looks right but totals are off, the bug is almost always in `rel_shift` indexing — print a small `[H=1,T=4]` slice and compare to a Python `rel_shift` on the same `bd`.
- [ ] **Step 5: Commit** — `git commit -m "feat: relative-position MHA with parity"`

---

## Task 7: Conformer layer — parity vs `enc_layer_0`

**Files:** Create `src/conformer.{hpp,cpp}`, `tests/test_conformer.cpp`. Modify CMake. Reuses `relpos_attention`.

**NeMo `ConformerLayer.forward`** (residual scaling `fc=0.5`; all LayerNorm; SiLU; verbatim suffixes under `encoder.layers.<i>.`):
```
r = x
x = norm_feed_forward1(r); x = feed_forward1(x); r = r + 0.5*x          # FFN1
x = norm_self_att(r);      x = self_attn(x,pos_emb); r = r + x          # MHSA
x = norm_conv(r);          x = conv(x);             r = r + x          # Conv module
x = norm_feed_forward2(r); x = feed_forward2(x); r = r + 0.5*x          # FFN2
out = norm_out(r)
```
- `feed_forwardN` = `linear1`(d→ff) → SiLU → `linear2`(ff→d). Suffixes `...feed_forward1.linear1.{weight,bias}` etc.
- `conv` (`ConformerConvolution`): `pointwise_conv1`(d→2d,k1) → GLU(dim=channel) → `depthwise_conv`(d→d,k=conv_kernel,groups=d, pad=(k-1)/2) → `batch_norm` (inference: affine from running stats; suffixes `...conv.batch_norm.{weight,bias,running_mean,running_var}`) → SiLU → `pointwise_conv2`(d→d,k1). Conv operates on `[d_model, T]` (transpose in/out around it).

- [ ] **Step 1: Failing parity test**: input = baseline `enc_pre_layers`, `pos_emb`; run full conformer layer 0; compare to `enc_layer_0` (atol/rtol 3e-2). Also assert the intermediate `conv` output matches `l0_conv_out` if you expose it (optional).
- [ ] **Step 2: Interface** `src/conformer.hpp`: `class ConformerLayer { ConformerLayer(const ModelLoader&, int idx); ggml_tensor* forward(ggml_context*, ggml_tensor* x, ggml_tensor* pos_emb) const; };` plus a convenience runner for the test.
- [ ] **Step 3: Implement** FFN, conv module (with batch_norm folded as `(x-mean)/sqrt(var+eps)*g+b`, eps=1e-5 — confirm against NeMo), GLU via `ggml_view`+`ggml_silu`/`ggml_sigmoid`+`ggml_mul`, depthwise conv via `ggml_conv_1d_dw`.
- [ ] **Step 4: Build + run** `ctest -R test_conformer`. Expected OK.
- [ ] **Step 5: Commit** — `git commit -m "feat: conformer layer with parity"`

---

## Task 8: Full encoder — parity vs `encoder_out` (+ mid/last layers)

**Files:** Create `src/encoder.{hpp,cpp}`, `tests/test_encoder.cpp`. Modify CMake.

**Flow:** mel `[80,T]` → subsampling → (if `xscaling`) `* sqrt(d_model)` → `RelPositionalEncoding` produces `pos_emb` `[2T'-1, d_model]` → run N `ConformerLayer`s → output transposed to `[d_model, T']` (NeMo encoder returns `[B, d_model, T']`). Build the pos encoding to match `RelPositionalEncoding` (trace `multi_head_attention.py`); the `xscaling` factor is applied to `x` (not pos_emb).

- [ ] **Step 1: Failing parity test** `tests/test_encoder.cpp`: from baseline `mel` → full encoder → compare to `encoder_out` (atol/rtol 5e-2). Additionally compare the captured mid/last layer outputs to `enc_layer_mid`/`enc_layer_last` for localization (print all three, fail if the final differs).
- [ ] **Step 2: Interface** `src/encoder.hpp`: `class Encoder { Encoder(const ModelLoader&); void forward(const std::vector<float>& mel, int n_mels, int T, std::vector<float>& enc_out, int& d_model, int& Tout) const; };` (output row-major `[d_model, Tout]`).
- [ ] **Step 3: Implement** assembling subsampling + pos-enc + the layer stack via one ggml graph (size `mem_bytes` generously, scale with T). Compute pos_emb in C++ or in-graph to match NeMo.
- [ ] **Step 4: Build + run** `ctest -R test_encoder`. Expected OK. If only the final differs while layer 0 matched, bisect with the mid/last comparisons.
- [ ] **Step 5: Commit** — `git commit -m "feat: FastConformer encoder with end-to-end parity"`

---

## Task 9: CTC head — parity vs `ctc_logits`

**Files:** Create `src/ctc_decoder.{hpp,cpp}`, `tests/test_ctc.cpp`. Modify CMake.

**NeMo `ConvASRDecoder`**: a single `Conv1d(d_model, vocab+1, k=1)` then `log_softmax(-1)`. Hybrid weight name `ctc_decoder.decoder_layers.0.{weight,bias}` (pure-CTC uses `decoder.decoder_layers.0.*`). Input `[d_model, T']` → logits `[T', vocab+1]`. Confirm the `ctc_logits` baseline orientation (documented by Task 7-Phase0) and match it.

- [ ] **Step 1: Failing parity test**: from baseline `encoder_out` → CTC head → compare to baseline `ctc_logits` (atol/rtol 5e-2). Note whether baseline stores pre- or post-log_softmax; match it.
- [ ] **Step 2: Interface** `src/ctc_decoder.hpp`: `class CTCDecoder { CTCDecoder(const ModelLoader&); void forward(const std::vector<float>& enc, int d_model, int T, std::vector<float>& logits, int& vocab_plus_1) const; };`
- [ ] **Step 3: Implement** as `ggml_mul_mat` (1×1 conv == linear over channels) + bias, optional log_softmax.
- [ ] **Step 4: Build + run** `ctest -R test_ctc`. Expected OK.
- [ ] **Step 5: Commit** — `git commit -m "feat: CTC head with parity"`

---

## Task 10: Tokenizer (id → text detok)

**Files:** Create `src/tokenizer.{hpp,cpp}`, `tests/test_tokenizer.cpp`. Modify CMake. Also EXTEND `scripts/gen_nemo_baseline.py` to dump a small fixture: an int32 tensor `detok_ids` and store the expected decoded string as a KV (`baseline.detok_text`) so the test has ground truth.

**Detok rule** (SentencePiece BPE): concatenate pieces for the ids, replace the `▁` (U+2581) meta-space with a regular space, strip a single leading space. Pieces come from GGUF KV `parakeet.tokenizer.pieces`.

- [ ] **Step 1: Extend baseline dumper**: pick a few token ids (e.g. `[10, 25, 100, 3, 7]`), set `cap["detok_ids"]=np.array(ids,int32)`, and `w.add_string("baseline.detok_text", m.tokenizer.ids_to_text(ids))`. Regenerate.
- [ ] **Step 2: Failing test** `tests/test_tokenizer.cpp`: read `parakeet.tokenizer.pieces` from the GGUF (extend `ModelLoader` to expose `tokenizer_pieces` — add a getter `const std::vector<std::string>& tokenizer_pieces()`; populate it in `ModelLoader::load` by reading the string array KV), read `detok_ids` + `baseline.detok_text` from baseline, run `pk::detokenize(pieces, ids)`, assert exact string match.
- [ ] **Step 3: Implement** `src/tokenizer.hpp`: `std::string detokenize(const std::vector<std::string>& pieces, const std::vector<int32_t>& ids);`
- [ ] **Step 4: Build + run** `ctest -R test_tokenizer`. Expected exact match. (If NeMo applies extra normalization in `ids_to_text`, match it — trace `SentencePieceTokenizer.ids_to_text`.)
- [ ] **Step 5: Commit** — `git commit -m "feat: BPE detokenizer + tokenizer KV loading"`

---

## Task 11: CTC greedy search + end-to-end transcribe

**Files:** Create `src/search.{hpp,cpp}`, `tests/test_transcribe.cpp`. Modify `include/parakeet.h`, `src/parakeet.cpp`, `examples/cli/main.cpp`, CMake.

**CTC greedy**: argmax per frame → collapse consecutive equal ids → drop blank (`blank_id == vocab_size`) → detokenize.

- [ ] **Step 1: Failing test** `tests/test_transcribe.cpp`: call `pk::transcribe(gguf_path, "tests/fixtures/clip.wav")` → returns a string. Assert it equals the NeMo reference. Store the reference as a baseline KV: EXTEND `gen_nemo_baseline.py` to `w.add_string("baseline.ctc_text", hyps[0].text if hasattr else str(hyps[0]))` (use the CTC head: set `m.cur_decoder='ctc'` for the hybrid before transcribe, or decode the CTC logits with the model's ctc decoding). Read `baseline.ctc_text` in the test and compare. (If the tone clip yields empty text, also assert the token-id sequence matches baseline `ctc_argmax_ids` collapsed — that's the real determinism check.)
- [ ] **Step 2: Implement search** `src/search.hpp`: `std::vector<int32_t> ctc_greedy(const std::vector<float>& logits, int T, int vocab_plus_1, int blank_id);`
- [ ] **Step 3: Orchestrator** — add to `include/parakeet.h`:
```cpp
// Transcribe a wav file. Returns 0 on success; writes UTF-8 text to *out (caller frees with parakeet_free_string).
int parakeet_transcribe_file(const char* model_path, const char* wav_path, char** out);
void parakeet_free_string(char* s);
```
Implement `pk::transcribe(model, wav)` in `src/parakeet.cpp`: `ModelLoader` → `MelFrontend` → `Encoder` → `CTCDecoder` → `ctc_greedy` → `detokenize`. Wire the C API to it.
- [ ] **Step 4: CLI** — add `transcribe` subcommand: `parakeet-cli transcribe --model m.gguf --input clip.wav` prints text.
- [ ] **Step 5: Build + run**
```bash
cmake --build build -j
PARAKEET_TEST_GGUF=$PARAKEET_TEST_GGUF PARAKEET_TEST_BASELINE=$PARAKEET_TEST_BASELINE ctest --test-dir build -R test_transcribe --output-on-failure
./build/bin/parakeet-cli transcribe --model /tmp/pk110m.gguf --input tests/fixtures/clip.wav
```
Expected: token-id/text parity with NeMo. Then run the WHOLE suite: `ctest --test-dir build --output-on-failure` (model tests need the env vars).
- [ ] **Step 6: Commit** — `git commit -m "feat: CTC greedy + end-to-end transcribe (CLI + C API)"`

---

## Task 12: Real-speech sanity check

**Files:** none (verification + optional fixture).

- [ ] **Step 1:** Download a short real English clip (a public LibriSpeech sample, or record a few words) at 16k mono, e.g. `tests/fixtures/speech.wav`. Run both:
```bash
.venv/bin/python -c "from nemo.collections.asr.models import ASRModel; m=ASRModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m'); m.cur_decoder='ctc'; print(m.transcribe(['tests/fixtures/speech.wav']))"
./build/bin/parakeet-cli transcribe --model /tmp/pk110m.gguf --input tests/fixtures/speech.wav
```
Expected: the C++ transcript matches NeMo's CTC transcript (word-for-word, or ≥ the agreed WER threshold). This is the real proof the pipeline works on speech, not just tensor parity on a tone.
- [ ] **Step 2:** If they diverge while all tensor-parity tests pass, the bug is in search/detok or normalization at the boundaries — re-check `ctc_greedy` collapse and the per-feature normalization mask. Commit the fixture + a note in `docs/parity.md` recording the reference transcript.

---

## Phase 1 done-when

- `ctest --test-dir build --output-on-failure` (with `PARAKEET_TEST_GGUF` + `PARAKEET_TEST_BASELINE` set): `test_fft`, `test_mel`, `test_subsampling`, `test_relpos_attention`, `test_conformer`, `test_encoder`, `test_ctc`, `test_tokenizer`, `test_transcribe` all PASS; `test_smoke`/`test_audio_io` still pass.
- `parakeet-cli transcribe --model <ctc-or-hybrid gguf> --input <speech.wav>` produces text matching NeMo's CTC head.
- All parity diffs are within tolerance (tighten tolerances toward 1e-3 where achievable and record final values in `docs/parity.md`).

---

## Handoff to Phase 2 (sequence gate)

When all Phase 1 done-when items pass and are committed, the Phase 2 plan
(`docs/superpowers/plans/2026-05-28-phase-2-transducer-core.md`, written next)
picks up. Phase 2 adds the prediction network (LSTM) + joint network with tensor
parity on the joint output (using the hybrid checkpoint's TDT joint). Phase 2's
entry gate re-runs this "Phase 1 done-when" checklist; do not start Phase 2
until it passes. Per the chaining convention (see Phase 0 plan), every phase
plan opens with Orientation + previous-phase entry gate and closes with
done-when + handoff.
