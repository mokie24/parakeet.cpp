# parakeet.cpp — Timestamps + Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Expose per-token and per-word **timestamps** and **confidence** from the C++ engine, validated against NeMo's `transcribe(timestamps=True)` + per-token confidence — matching NeMo's word offsets and the `max_prob` confidence method.

**Architecture:** The greedy decoders (CTC/RNNT/TDT) already know, at each emission, the encoder frame `t` and the logits — so per-token `{id, frame, confidence}` is a small extension. Time = `frame × hop × subsampling_factor / sample_rate` (= 0.08 s/frame for these models). Words are formed by grouping tokens on the `▁` (U+2581) word-start marker. Confidence = softmax probability of the emitted token (NeMo's `max_prob` method); word confidence = the aggregate NeMo uses (verify: mean vs min). Results surface through a `pk::Transcription` struct, the C-API (JSON), the CLI (`--timestamps`/`--json`), and the streaming path.

**Tech Stack:** C++17, ggml, `.venv` NeMo, ctest. Branch `feat/timestamps-confidence` (also carries the `max_symbols` metadata fix). Anchor `nvidia/parakeet-tdt_ctc-110m` (hybrid → validate both TDT and CTC heads); `/tmp/eou.gguf` for streaming.

**Reference:** NeMo timestamp logic — `parts/submodules/rnnt_decoding.py` / `ctc_decoding.py` (`compute_timestamps`, word offset grouping by `▁`), `parts/utils/asr_confidence_utils.py` (confidence methods; `max_prob`). Existing code: `src/rnnt.cpp` (`rnnt_decode_frames` has `emit_frames`), `src/tdt.cpp`, `src/search.cpp` (`ctc_greedy`), `src/tokenizer.cpp`, `src/model.cpp`, `src/streaming.cpp`, `src/parakeet_capi.cpp`, `examples/cli/main.cpp`.

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## Orientation + Entry gate
Read spec/this plan; verify env; resume via `git log`. Entry gate: build + `ctest` (32 tests, 29 pass + 3 big-model skips) with the standard env vars. Convert the anchor (`/tmp/pk110m.gguf`) + `/tmp/eou.gguf`; gen baselines.

---

## Task 1: NeMo timestamp + confidence baseline
**Files:** `scripts/gen_nemo_baseline.py`.
- [ ] Extend the dumper to capture, for `tests/fixtures/speech.wav` on `parakeet-tdt_ctc-110m`, with the TRANSDUCER head (and separately the CTC head): NeMo's per-token `{token_id, frame, confidence}` and per-word `{word_text, start_time, end_time, confidence}`. Use `m.transcribe([clip], timestamps=True, return_hypotheses=True)` with confidence enabled via the decoding config set to the **`max_prob`** method (set `m.change_decoding_strategy(...)` / `decoding.confidence_cfg` so `preserve_frame_confidence=True`, `confidence_cfg.method_cfg.name='max_prob'`; VERIFY the exact config path in `asr_confidence_utils.py`/the decoding cfg). Dump: int32 `ts_token_ids`, int32 `ts_token_frames`, f32 `ts_token_conf`, and KV/JSON `baseline.words_json` = `[{"w":..,"start":..,"end":..,"conf":..}, ...]`. Also record `frame_sec` (= window_stride*subsampling_factor). Produce `/tmp/baseline_ts.gguf`. Confirm the dumped word texts concatenate to the known transcript and frames are monotonic. Commit: `git commit -m "feat: dump NeMo token/word timestamps + max_prob confidence baseline"`.

## Task 2: Per-token frame + confidence in the greedy decoders
**Files:** `src/rnnt.{hpp,cpp}`, `src/tdt.{hpp,cpp}`, `src/search.{hpp,cpp}`.
- [ ] Define `struct TokenInfo { int32_t id; int32_t frame; float conf; };`. Extend each greedy decoder to optionally output `std::vector<TokenInfo>` alongside the id list:
  - **CTC** (`ctc_greedy`): at each emitted (collapsed, non-blank) frame, `frame = t`, `conf = exp(logit[p])` (CTC logits are log-softmax) — confirm the CTC logits orientation. 
  - **RNNT** (`rnnt_decode_frames`/`rnnt_greedy`): reuse the existing `emit_frames`; add `conf = softmax(joint_logits)[k]` at the emission step (compute softmax over the `V_plus` vector for the chosen `k`).
  - **TDT** (`tdt_greedy`): `frame = t`; `conf = softmax(token_logits[:vocab+1])[k]` (the token slice, not durations).
  Keep the existing id-only return paths working (overload or optional out-param). 
- [ ] Build + confirm existing decode tests still pass. Commit: `git commit -m "feat: per-token frame+confidence (max_prob) in CTC/RNNT/TDT greedy"`.

## Task 3: Word grouping + Transcription result + parity
**Files:** `src/transcription.{hpp,cpp}` (new) or extend `src/model.{hpp,cpp}`; `tests/test_timestamps.cpp`.
- [ ] `struct Word { std::string text; float start, end, conf; }; struct Transcription { std::string text; std::vector<Word> words; std::vector<TokenInfo> tokens; };`
- [ ] Word grouping: a token whose piece begins with `▁` starts a new word (the leading `▁` of the utterance too). `word.start = frame_sec * first_token_frame`; `word.end = frame_sec * (last_token_frame + 1)` (or the next word's start — VERIFY against NeMo's word-offset convention and match it); `word.conf` = the NeMo aggregate (verify mean vs min in `asr_confidence_utils.py` `aggregation`). `word.text` = detokenized tokens of the word.
- [ ] `pk::Model::transcribe_with_timestamps(pcm/​path, decoder) -> Transcription`.
- [ ] `tests/test_timestamps.cpp`: on `/tmp/pk110m.gguf` + `tests/fixtures/speech.wav`, assert: token ids/frames == `ts_token_*`; per-token conf within 1e-3 of `ts_token_conf`; word texts exact, word start/end within `frame_sec` (1 frame), word conf within 1e-3 of `baseline.words_json`. Skip 77 unless `PARAKEET_TEST_GGUF`+`PARAKEET_TEST_BASELINE_TS` set. LABEL model, WORKING_DIRECTORY. Validate both TDT and CTC heads.
- [ ] Commit: `git commit -m "feat: word grouping + Transcription (timestamps+confidence), parity vs NeMo"`.

## Task 4: API + CLI + streaming exposure
**Files:** `include/parakeet_capi.h` + `src/parakeet_capi.cpp`, `examples/cli/main.cpp`, `src/streaming.{hpp,cpp}`, `tests/test_capi_timestamps.cpp`; docs.
- [ ] **C-API:** `char* parakeet_capi_transcribe_path_json(parakeet_ctx*, const char* wav, int decoder)` returning malloc'd JSON `{"text":..,"words":[{"w","start","end","conf"}],"tokens":[{"id","t","conf"}]}` (no-throw boundary, freed by `parakeet_capi_free_string`). (A streaming variant can attach word timestamps to finalized text via the existing event mechanism.)
- [ ] **CLI:** `parakeet-cli transcribe ... --timestamps` prints `word  [start-end]  (conf)` lines; `--json` prints the JSON. Keep plain output default.
- [ ] **Streaming:** surface per-word timestamps + confidence in the streaming session alongside EOU events (reuse the per-token frame/conf; word boundaries as tokens finalize).
- [ ] `tests/test_capi_timestamps.cpp`: JSON transcribe on the anchor → parse → assert text + a couple of word times/confs match the Task 3 result. Skip-if-absent.
- [ ] Update `README.md`/`AGENTS.md` (timestamps+confidence usage, the `max_prob` confidence method, the new C-API symbol) and `docs/parity.md` (timestamp/confidence parity numbers). Commit: `git commit -m "feat: timestamps+confidence via C-API/CLI/streaming + docs"`.

---

## Done-when
- Token frames + confidence and word start/end + confidence match NeMo (`timestamps=True`, `max_prob`) within tolerance, for both TDT and CTC heads.
- C-API JSON + CLI `--timestamps`/`--json` + streaming word timestamps work; tests pass; full suite green (new tests skip cleanly without their env vars).
- Docs updated. Then merge the branch (incl. the `max_symbols` fix) to master + push.
