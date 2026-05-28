# parakeet.cpp Phase 2 — Transducer Core (prediction net + joint) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the RNN-Transducer core — the LSTM prediction network and the joint network — with tensor parity against NeMo, so Phase 3 can add the (TDT duration-aware) greedy decoding loop on top.

**Architecture:** Two new components built bottom-up with parity tests against new ground-truth tensors dumped from NeMo's `RNNTDecoder.predict` and `RNNTJoint.joint`. The prediction net is a hand-rolled embedding + single-layer LSTM (ggml has no LSTM op). The joint net is two linear projections summed, activated (ReLU), and a final linear to `vocab+1+num_durations` logits. Both run on the same hybrid checkpoint (`parakeet-tdt_ctc-110m`) whose joint is a TDT joint (durations `[0,1,2,3,4]`). Pure-RNNT checkpoints (durations empty) use the same code.

**Tech Stack:** C++17, ggml v0.13.0, the `.venv` NeMo env, ctest. Anchor `nvidia/parakeet-tdt_ctc-110m`.

**Reference (read before starting):**
- Spec `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md` (§2.1 families, §2 decoders).
- Phase 1 plan + the existing code: `pk::ModelLoader`, `pk::run_graph`, `pk::Encoder` (gives `encoder_out`), `pktest::load_baseline`/`compare`, `scripts/gen_nemo_baseline.py`.
- NeMo math: `/home/mudler/_git/NeMo/nemo/collections/asr/modules/rnnt.py` → `RNNTDecoder` (`predict`, the prednet), `RNNTJoint` (`joint`, `forward`).

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; no skipped hooks; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## Validated transducer tensors (from the GGUF, verbatim NeMo names)
```
decoder.prediction.embed.weight                 [1025, 640]   # Embedding(vocab+1, pred_hidden), padding_idx=blank=1024
decoder.prediction.dec_rnn.lstm.weight_ih_l0    [2560, 640]   # 4*640 x 640
decoder.prediction.dec_rnn.lstm.weight_hh_l0    [2560, 640]
decoder.prediction.dec_rnn.lstm.bias_ih_l0      [2560]
decoder.prediction.dec_rnn.lstm.bias_hh_l0      [2560]
joint.enc.weight  [640, 512]   joint.enc.bias  [640]          # encoder projection 512->640
joint.pred.weight [640, 640]   joint.pred.bias [640]          # prediction projection 640->640
joint.joint_net.2.weight [1030, 640]  joint.joint_net.2.bias [1030]   # output: 1024 vocab + 1 blank + 5 durations
```
`pred_hidden=joint_hidden=640`, `enc_hidden=512`, `vocab=1024`, `blank=1024`, `num_durations=5`. `joint_net` is `Sequential(0=ReLU, 1=Dropout, 2=Linear)` — only index 2 has weights; activation is **ReLU applied to the summed projections** before the output linear.

---

## Orientation — run BEFORE any task (fresh sub-agent, zero context)
1. Read the spec + this plan + skim the Phase 1 plan. The committed code is the source of truth for existing interfaces.
2. Verify env: `git rev-parse --is-inside-work-tree`→true; `.venv/bin/python -c "import nemo,gguf,torch"` works; `/home/mudler/_git/NeMo` exists.
3. Resume point: `git log --oneline -25`; start at the first incomplete task (each ends in a recognizable commit).

## Entry gate — Phase 1 must be complete (do not start otherwise)
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
ctest --test-dir build --output-on-failure
```
Expect **15/15 PASS** (incl. `test_encoder`, `test_ctc`, `test_transcribe_speech` = WER 0 on real speech). If anything fails, STOP and finish Phase 1.

---

## File structure created in this phase
```
src/
  prediction.hpp / prediction.cpp   # embedding + 1-layer LSTM prediction net
  joint.hpp / joint.cpp             # enc/pred projections + ReLU + output linear
tests/
  test_prediction.cpp
  test_joint.cpp
scripts/gen_nemo_baseline.py        # EXTEND: dump pred-net + joint ground truth
```

---

## Task 1: Extend baseline dumper for the transducer core

**Files:** Modify `scripts/gen_nemo_baseline.py`, `tests/python/check_baseline.py`.

Dump deterministic ground truth for the prediction net and joint. TRACE `RNNTDecoder.predict` and `RNNTJoint.joint`/`forward` in `/home/mudler/_git/NeMo/nemo/collections/asr/modules/rnnt.py` for exact signatures.

- [ ] **Step 1: Dump prediction-net output for a fixed token sequence**

Pick a deterministic input label sequence, e.g. `pred_input_ids = [120, 7, 300, 42]` (avoid blank=1024). Run NeMo's prediction net on it. `RNNTDecoder.predict(targets, target_length, states=None)` (verify the exact arg names) returns `(g, states)` where `g` is `[B, U(+1 if add_sos), pred_hidden]`. Use the model's `m.decoder`. Capture `g` squeezed → store tensor `pred_out`. Also store the input as an int32 tensor `pred_input_ids`, and record (in a comment / KV) whether `add_sos` prepended a step (so the C++ matches the exact output length and SOS handling). The SOS step uses the zero embedding (Embedding `padding_idx=blank`); confirm `embed.weight[1024]` is all zeros (padding_idx rows are zeroed) — if not, capture the actual SOS embedding behavior.

- [ ] **Step 2: Dump joint output**

Compute `joint_out = m.joint.joint(enc, pred)` (verify method name; `RNNTJoint.joint(f, g)` takes encoder output `[B,T,enc_hidden]` and decoder output `[B,U,pred_hidden]` and returns `[B,T,U,vocab+1+num_durations]`). Use the REAL `encoder_out` (transpose to `[B,T,enc_hidden]` as the joint expects) and the `pred_out` from Step 1. To keep it small, you may slice to a small `T×U` (e.g. first 4 encoder frames × the U from Step 1) — but record the slice used. Store squeezed `joint_out` (e.g. `[T_slice, U, 1030]`) and also store `joint_enc_frames` (the int count of encoder frames used) so the C++ test feeds the matching encoder slice. NOTE: confirm whether NeMo applies `log_softmax` in `joint` (the TDT joint typically returns RAW logits; the token-vs-duration split is `[:vocab+1]` and `[vocab+1:]`). Record raw-vs-logsoftmax in `docs/conversion.md`.

- [ ] **Step 3: Verify + register**

Regenerate `/tmp/baseline.gguf`; confirm `pred_out`, `pred_input_ids`, `joint_out` appear with sane shapes (`pred_out` ~`[U(+1), 640]`, `joint_out` ~`[T_slice, U(+1), 1030]`). Extend `tests/python/check_baseline.py` required set. Re-run twice for byte-identical determinism.

- [ ] **Step 4: Commit** — `git commit -m "feat: dump prediction-net + joint ground truth for transducer parity"`

---

## Task 2: Prediction network (embedding + 1-layer LSTM)

**Files:** Create `src/prediction.hpp`, `src/prediction.cpp`, `tests/test_prediction.cpp`. Modify CMake.

**NeMo math (verify against `RNNTDecoder` in rnnt.py):**
- Embedding lookup: `e[u] = embed.weight[id[u]]` (`[pred_hidden=640]`); the SOS step uses the zero (padding_idx) embedding — match Step-1 behavior from the dumper.
- Single-layer LSTM, PyTorch convention. Weights `weight_ih_l0 [4H,H]`, `weight_hh_l0 [4H,H]`, `bias_ih_l0 [4H]`, `bias_hh_l0 [4H]`, H=640. Gate order in PyTorch is **[input, forget, cell, output]** stacked in the 4H dim. Per step with input `x_t [H]`, prev `h,c [H]`:
  ```
  z = W_ih · x_t + b_ih + W_hh · h + b_hh        # [4H]
  i = sigmoid(z[0:H]); f = sigmoid(z[H:2H]); g = tanh(z[2H:3H]); o = sigmoid(z[3H:4H])
  c' = f * c + i * g
  h' = o * tanh(c')
  ```
  Initial `h0 = c0 = 0`. Process the U(+1) steps sequentially. Output is the sequence of `h'` → `[U(+1), H]`.
- The LSTM is sequential (state carried) so it does NOT map to a single ggml matmul over time; implement the per-step recurrence (you can do the `W_ih·x` for all steps as one matmul, but the `W_hh·h` term must be sequential). A clean approach: precompute `Wx = X · W_ih^T + b_ih` for all steps via `pk::run_graph` (one matmul), then do the recurrent gate loop in plain C++ (sigmoid/tanh on 640-vectors) carrying h,c. Either all-ggml or hybrid is fine — the parity test is the gate.

- [ ] **Step 1: Failing parity test** `tests/test_prediction.cpp`: read `pred_input_ids` from `$PARAKEET_TEST_BASELINE`, run `pk::PredictionNet::forward(ids)`, compare to baseline `pred_out` (atol/rtol 2e-3). Mirror existing tests (env, WORKING_DIRECTORY, LABEL model).
- [ ] **Step 2: Interface** `src/prediction.hpp`: `class PredictionNet { PredictionNet(const ModelLoader&); void forward(const std::vector<int32_t>& ids, bool add_sos, std::vector<float>& out, int& U_out, int& hidden) const; };` (match the dumper's add_sos handling; expose state-carry later for Phase 3 greedy — for now full-sequence is enough).
- [ ] **Step 3: Implement** embedding + LSTM per the math. Read H from `cfg.pred_hidden`.
- [ ] **Step 4: Build + run** `ctest -R test_prediction`. Expect OK. Debug: if off, check gate order (PyTorch [i,f,g,o]), the bias_ih+bias_hh sum, the SOS embedding, and that you carry state across steps. Compare step-0 `h'` against a Python LSTM-cell recompute to localize.
- [ ] **Step 5: Commit** — `git commit -m "feat: LSTM prediction network with parity"`

---

## Task 3: Joint network

**Files:** Create `src/joint.hpp`, `src/joint.cpp`, `tests/test_joint.cpp`. Modify CMake.

**NeMo math (verify against `RNNTJoint.joint`):**
- `enc_proj[t] = enc.weight · enc[t] + enc.bias`  (512→640), for each encoder frame t.
- `pred_proj[u] = pred.weight · pred[u] + pred.bias` (640→640), for each prediction step u.
- `f[t,u] = ReLU(enc_proj[t] + pred_proj[u])`  (broadcast add → `[T,U,640]`).
- `logits[t,u] = joint_net.2.weight · f[t,u] + joint_net.2.bias`  (640→1030).
- Output `[T, U, 1030]`. The TDT split: `logits[..., :vocab+1]` are token logits (incl. blank at 1024), `logits[..., vocab+1:]` are the `num_durations=5` duration logits. Match whether NeMo returns raw logits (per Task 1 finding).

- [ ] **Step 1: Failing parity test** `tests/test_joint.cpp`: read baseline `encoder_out` (slice to `joint_enc_frames`), `pred_out`, and target `joint_out`; run `pk::Joint::forward(enc_slice, pred)`; compare to `joint_out` (atol/rtol 5e-3). LABEL model, WORKING_DIRECTORY.
- [ ] **Step 2: Interface** `src/joint.hpp`: `class Joint { Joint(const ModelLoader&); void forward(const std::vector<float>& enc, int T, int enc_hidden, const std::vector<float>& pred, int U, int pred_hidden, std::vector<float>& logits, int& V_plus) const; };` (output row-major `[T, U, V_plus]`, `V_plus = vocab+1+num_durations`). Also expose `num_durations()` / `vocab_size()` for Phase 3.
- [ ] **Step 3: Implement** via `pk::run_graph` (two matmuls + broadcast-add + relu + output matmul + bias). Read dims from config; `num_durations = cfg.tdt_durations.size()`.
- [ ] **Step 4: Build + run** `ctest -R test_joint`. Expect OK. Debug: check the broadcast-add orientation and that ReLU is applied to the SUM (not the projections), and the output split. Do NOT loosen tolerance.
- [ ] **Step 5: Commit** — `git commit -m "feat: joint network with parity"`

---

## Task 4: Transducer-core integration check

**Files:** Modify `tests/test_joint.cpp` or add a small `tests/test_transducer_core.cpp` (your call; keep it focused).

- [ ] **Step 1:** Add an assertion that composing `PredictionNet.forward(pred_input_ids)` → `Joint.forward(encoder_out_slice, pred)` reproduces baseline `joint_out` end-to-end (i.e. feed the C++ prediction output into the C++ joint, not the baseline `pred_out`). This catches interface-seam bugs between the two components. atol/rtol 5e-3.
- [ ] **Step 2: Build + run + commit** — `git commit -m "test: transducer-core integration parity"`. Re-run the full suite.

---

## Phase 2 done-when
- `ctest --test-dir build --output-on-failure` (with env vars): `test_prediction`, `test_joint` (+ integration) PASS alongside all Phase 0/1 tests.
- Prediction-net and joint outputs match NeMo within tolerance; the composed pred→joint reproduces `joint_out`.
- `docs/parity.md` updated with the prediction-net and joint parity numbers.

---

## Handoff to Phase 3 (sequence gate)
When all Phase 2 done-when items pass and are committed, the Phase 3 plan
(`docs/superpowers/plans/2026-05-28-phase-3-tdt-greedy.md`, written next) picks
up: the duration-aware TDT greedy decoding loop that drives `PredictionNet` +
`Joint` frame-by-frame (advancing the time index by the predicted duration),
producing end-to-end TDT transcription → validated on `parakeet-tdt_ctc-110m`
(TDT head), then `parakeet-tdt-0.6b-v2` and `-v3`. Phase 3's entry gate re-runs
this "Phase 2 done-when" checklist. Per the chaining convention (Phase 0 plan),
every phase plan opens with Orientation + previous-phase entry gate and closes
with done-when + handoff.
