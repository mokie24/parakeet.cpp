# parakeet.cpp Phase 3 — TDT Duration-Aware Greedy Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the TDT (Token-and-Duration Transducer) duration-aware greedy decoding loop that drives the prediction net + joint frame-by-frame (advancing the time index by the predicted duration), producing end-to-end TDT transcription — validated to match NeMo on real speech, including the north-star `parakeet-tdt-0.6b-v2` and multilingual `-v3`.

**Architecture:** A greedy loop ports NeMo's `GreedyTDTInfer._greedy_decode`. It needs a **stateful single-step** prediction net (carry LSTM `h,c` across steps, commit state only when a non-blank is emitted) and the existing `pk::Joint` (called per (t,u)). Decoder validation is **behavioral** (token-id / WER parity vs NeMo), not tensor parity — the encoder/joint are already tensor-parity-verified. The orchestrator picks TDT vs CTC by the GGUF `parakeet.arch`.

**Tech Stack:** C++17, ggml v0.13.0, `.venv` NeMo, ctest. Anchors: `parakeet-tdt_ctc-110m` (TDT head), then `parakeet-tdt-0.6b-v2` and `parakeet-tdt-0.6b-v3`.

**Reference (read first):**
- Spec `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md` (§2.1, §2 TDT, §9 A+C strategy).
- Phase 2 plan + code: `pk::PredictionNet` (`src/prediction.hpp`), `pk::Joint` (`src/joint.hpp`, raw logits `[T,U,1030]`, `num_durations()`/`vocab_size()`), `pk::Encoder`, `pk::detokenize`, `pk::ModelLoader`, `pk::transcribe` (CTC path, `src/parakeet.cpp`).
- NeMo math: `/home/mudler/_git/NeMo/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` → `GreedyTDTInfer._greedy_decode` (≈ lines 2655-2745) and `_pred_step`/`_joint_step`.

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## The TDT greedy algorithm (ported from NeMo GreedyTDTInfer)
```
durations = cfg.tdt_durations          # [0,1,2,3,4]
blank = vocab_size                      # 1024
t = 0; hyp = []; h = c = 0; last_token = None (SOS); committed_state = zeros
while t < T:
    symbols_added = 0
    need_loop = True
    while need_loop and symbols_added < max_symbols:
        last_label = SOS(=blank/zero-embed) if no token emitted yet else last_token
        g, h', c' = pred_step(last_label, committed_state)     # single LSTM step
        logits = joint(enc[t], g)                              # raw [1030]
        token_logits = logits[:vocab+1]                        # [1025]
        dur_logits   = logits[vocab+1:]                        # [5]
        k   = argmax(token_logits)        # argmax of raw == argmax of softmax
        d_k = argmax(dur_logits)
        skip = durations[d_k]
        if k != blank:
            hyp.append(k); last_token = k; committed_state = (h',c')   # commit ONLY on non-blank
        symbols_added += 1
        t += skip
        need_loop = (skip == 0)           # if duration 0, re-loop at same frame (multi-token)
    if skip == 0: skip = 1                # infinite-loop guard
    if symbols_added == max_symbols: t += 1
text = detokenize(pieces, hyp)
```
Notes:
- **State commit only on non-blank.** On blank, `committed_state`/`last_token` are unchanged; `g,h',c'` are discarded.
- `max_symbols` (a.k.a. max_symbols_per_step) — match NeMo's effective value for this model (check the model's `cfg.decoding.greedy.max_symbols`; if unset, NeMo's default is 10). Read it or default to 10.
- The first step uses SOS: NeMo's `_pred_step(self._SOS, None)` returns the prediction-net output for the zero SOS input with zero initial state — the same SOS handling as Phase 2's `add_sos`.
- Argmax over RAW token/duration logits (NeMo log_softmaxes duration only for confidence; argmax is invariant). No softmax needed for greedy.

---

## Orientation — run BEFORE any task (fresh sub-agent)
1. Read the spec + this plan + skim Phase 1/2 plans. Committed code is the source of truth.
2. Verify env: `git rev-parse --is-inside-work-tree`→true; `.venv/bin/python -c "import nemo,gguf,torch"`; `/home/mudler/_git/NeMo` exists.
3. Resume point: `git log --oneline -25`; start at first incomplete task.

## Entry gate — Phase 2 must be complete
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
ctest --test-dir build --output-on-failure
```
Expect **18/18 PASS** (incl. `test_prediction`, `test_joint`, `test_transducer_core`). If not, STOP and finish Phase 2.

---

## File structure created in this phase
```
src/
  prediction.{hpp,cpp}   # EXTEND: stateful single-step API (h,c carry)
  tdt.{hpp,cpp}          # TDT greedy decoding loop
  parakeet.cpp           # EXTEND: transcribe() routes CTC vs TDT by arch
examples/cli/main.cpp    # EXTEND: --decoder ctc|tdt (default: by arch)
tests/
  test_prediction_step.cpp   # stateful step parity
  test_tdt_greedy.cpp        # token-id parity vs NeMo on a known case
  test_transcribe_tdt.cpp    # end-to-end TDT WER vs NeMo on speech.wav
scripts/gen_nemo_baseline.py # EXTEND: dump baseline.tdt_text + tdt token ids
```

---

## Task 1: Stateful single-step prediction net

**Files:** Modify `src/prediction.hpp` / `src/prediction.cpp`; create `tests/test_prediction_step.cpp`.

The greedy loop needs to advance the LSTM one token at a time, carrying `(h,c)`, committing only on non-blank. Add a stateful API alongside the existing full-sequence `forward`.

- [ ] **Step 1: Add the stateful API.** `struct PredState { std::vector<float> h, c; };` and
  `void PredictionNet::step(int32_t token_id, bool is_sos, const PredState& in, std::vector<float>& g, PredState& out) const;`
  — one LSTM step: input embedding is the zero SOS vector if `is_sos` else `embed.weight[token_id]`; run the gates from `in.{h,c}` → produce `g` (=`h'`, the `[hidden]` output) and `out.{h',c'}`. A zero state helper `PredState PredictionNet::zero_state() const`.
- [ ] **Step 2: Failing parity test** `tests/test_prediction_step.cpp`: starting from `zero_state()`, step through the sequence `[SOS, ids[0], ids[1], ids[2], ids[3]]` (using `pred_input_ids` from `$PARAKEET_TEST_BASELINE`), collecting each `g` → `[5, 640]`, and assert it equals baseline `pred_out` (atol/rtol 2e-3). This proves the stateful stepping reproduces the full-sequence prediction net. LABEL model, WORKING_DIRECTORY.
- [ ] **Step 3: Implement** `step` (refactor the existing LSTM recurrence into a single-step helper that `forward` also uses, to avoid duplication). Keep `forward` working (test_prediction must still pass).
- [ ] **Step 4: Build + run** `ctest -R "test_prediction_step|test_prediction"`. Both pass. 
- [ ] **Step 5: Commit** — `git commit -m "feat: stateful single-step prediction net"`

---

## Task 2: TDT greedy decoding loop

**Files:** Create `src/tdt.hpp`, `src/tdt.cpp`, `tests/test_tdt_greedy.cpp`. Modify CMake.

- [ ] **Step 1: Interface** `src/tdt.hpp`:
  `std::vector<int32_t> pk::tdt_greedy(const PredictionNet& pred, const Joint& joint, const std::vector<float>& enc, int T, int enc_hidden, const std::vector<int32_t>& durations, int blank_id, int max_symbols);`
  returning the emitted token-id sequence. `enc` row-major `[T, enc_hidden]`.
- [ ] **Step 2: Implement** the algorithm above exactly. Per inner step: `pred.step(...)`, `joint.forward(enc_frame (T=1), pred_g (U=1), logits, V_plus)` → split `logits[:vocab+1]` / `logits[vocab+1:]`, argmax each, `skip=durations[d_k]`, emit + commit state on non-blank, `t+=skip`, `need_loop = skip==0`, the `skip==0→1` guard, and the `symbols_added==max_symbols → t+=1` guard. (Calling `Joint::forward` with T=U=1 per step is fine for correctness; do not optimize yet.)
- [ ] **Step 3: Failing token-parity test** `tests/test_tdt_greedy.cpp`: To get a deterministic NeMo reference token sequence, EXTEND `scripts/gen_nemo_baseline.py` to also dump, on `tests/fixtures/speech.wav` (the real clip), NeMo's TDT greedy token-id sequence as `tdt_token_ids` (int32) — run the model's RNNT/TDT head greedy decode and capture the hypothesis `y_sequence`. Since the existing baseline is built from `clip.wav` (tones → empty), produce a SECOND baseline for the speech clip OR add a dedicated dumper invocation. (Recommended: add a `--audio` run for `speech.wav` producing `/tmp/baseline_speech.gguf` with `tdt_token_ids`, and have the test read it via a `PARAKEET_TEST_BASELINE_SPEECH` env var; register the test to skip 77 if unset.) Then in the test: run the full C++ path encoder→tdt_greedy on `speech.wav` and assert the emitted token ids equal `tdt_token_ids` exactly.
- [ ] **Step 4: Build + run** `ctest -R test_tdt_greedy`. Expect exact token-id match. Debug: if tokens diverge, dump the per-frame (k, d_k, skip) from both NeMo (add a hook/print in a throwaway `.venv` script) and C++ and diff the first divergence — usual culprits: state-commit-on-blank, SOS handling, the duration index→skip mapping, or max_symbols. Do NOT loosen to "close enough" — token ids must match.
- [ ] **Step 5: Commit** — `git commit -m "feat: TDT duration-aware greedy decoding"`

---

## Task 3: End-to-end TDT transcribe (orchestrator + CLI + C API)

**Files:** Modify `src/parakeet.cpp`, `include/parakeet.h`, `examples/cli/main.cpp`; create `tests/test_transcribe_tdt.cpp`. Modify `scripts/gen_nemo_baseline.py`.

- [ ] **Step 1: Route by arch.** Extend `pk::transcribe(model, wav)` so that for `arch ∈ {tdt, hybrid_tdt_ctc, rnnt, hybrid_rnnt_ctc}` it runs encoder → `tdt_greedy` (durations from config; for pure RNNT durations is effectively `[1]`-like — but those archs are out of scope to validate here, just don't crash) → detokenize; for `arch ∈ {ctc, hybrid_*ctc with CTC selected}` it runs the existing CTC path. Add an optional explicit selector. Keep the CTC path (and `test_transcribe_speech`) working — for a hybrid, default to the TDT head (matches NeMo's default `cur_decoder`), but allow forcing CTC.
- [ ] **Step 2: CLI** add `--decoder ctc|tdt` to `parakeet-cli transcribe` (default: chosen by arch). Keep `info`.
- [ ] **Step 3: NeMo reference.** Run NeMo's TDT head on `speech.wav`: `m.change_decoding_strategy(decoder_type='rnnt')` (the transducer head; for this hybrid that's the TDT decoder) → `text = m.transcribe(['tests/fixtures/speech.wav'])[0].text`. Record this exact string. Add it as the expected value in `tests/test_transcribe_tdt.cpp` (a committed constant, like `test_transcribe_speech` does for CTC) OR store it as `baseline.tdt_text` in the speech baseline and read it.
- [ ] **Step 4: Test** `tests/test_transcribe_tdt.cpp`: assert `pk::transcribe($PARAKEET_TEST_GGUF, "tests/fixtures/speech.wav")` (TDT path) equals the NeMo TDT reference (target: exact match / WER 0; if a tiny WER remains, report it and diagnose — do not loosen silently). LABEL model, WORKING_DIRECTORY.
- [ ] **Step 5: Build + run + manual check**
```
cmake --build build -j
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf ctest --test-dir build -R test_transcribe_tdt --output-on-failure
./build/examples/cli/parakeet-cli transcribe --model /tmp/pk110m.gguf --input tests/fixtures/speech.wav --decoder tdt
```
Run the FULL suite (no regressions; the CTC speech test still passes).
- [ ] **Step 6: Commit** — `git commit -m "feat: end-to-end TDT transcribe (CLI + C API routing by arch)"`

---

## Task 4: North-star validation — parakeet-tdt-0.6b-v2 and -v3

**Files:** `docs/parity.md` (update); possibly a small committed manifest. No core code changes expected (metadata-driven).

- [ ] **Step 1: Convert + transcribe the north-star TDT model.** Download/convert `nvidia/parakeet-tdt-0.6b-v2` (TDT-only, ~0.6B; the converter handles it — `arch=tdt`):
```
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt-0.6b-v2 --output /tmp/tdt06v2.gguf
./build/examples/cli/parakeet-cli info /tmp/tdt06v2.gguf        # confirm arch=tdt, durations, dims
./build/examples/cli/parakeet-cli transcribe --model /tmp/tdt06v2.gguf --input tests/fixtures/speech.wav --decoder tdt
```
Get the NeMo reference: load `nvidia/parakeet-tdt-0.6b-v2`, `m.transcribe(['tests/fixtures/speech.wav'])[0].text`. Compare. Compute WER. Expect a clean transcript matching NeMo (this is the headline result).
- [ ] **Step 2: Multilingual -v3.** Same for `nvidia/parakeet-tdt-0.6b-v3` (larger/different BPE vocab, 25 EU langs — architecture identical, so it should "just work" via the metadata-driven loader/tokenizer). Transcribe `speech.wav` (English) and compare to NeMo; optionally a non-English clip if easily available.
- [ ] **Step 3: Diagnose any divergence.** If the 0.6B TDT model diverges from NeMo while the 110m hybrid matched, likely causes are model-config differences the loader must honor (e.g. `xscaling` True on this model — verify `info` shows it and the encoder applies it; conv_norm_type; different d_model/layers — all metadata-driven, so a mismatch indicates a hardcoded assumption to fix). Record findings.
- [ ] **Step 4: Record results in `docs/parity.md`** — the NeMo vs C++ transcripts and WER for `parakeet-tdt_ctc-110m` (TDT), `parakeet-tdt-0.6b-v2`, and `-v3`; note the per-stage tensor-parity still holds. Commit: `git commit -m "test: validate parakeet-tdt-0.6b-v2/-v3 end-to-end vs NeMo"`.

---

## Phase 3 done-when
- `ctest --test-dir build --output-on-failure` (env vars set): `test_prediction_step`, `test_tdt_greedy`, `test_transcribe_tdt` PASS with all earlier tests.
- `parakeet-cli transcribe --decoder tdt` on `speech.wav` matches NeMo's TDT transcript (WER 0 / documented).
- **`parakeet-tdt-0.6b-v2` transcribes real speech matching NeMo** (north-star headline), and `-v3` works.
- `docs/parity.md` updated with TDT + north-star results.

---

## Handoff to Phase 4 (sequence gate)
When Phase 3 done-when passes and is committed, Phase 4 (productionize) picks up:
quantization (Q8_0/Q4_K via a CLI `quantize` + converter quant types), the flat
`parakeet_capi.h` for dlopen/LocalAI, an HF model-publishing script, and the
full CI closed-loop job. Phase 4's entry gate re-runs this "Phase 3 done-when".
Per the chaining convention (Phase 0 plan), every phase plan opens with
Orientation + previous-phase entry gate and closes with done-when + handoff.
