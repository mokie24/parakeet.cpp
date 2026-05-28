# parakeet.cpp — Parity report (Phase 1)

This document records the numerical and end-to-end parity of the C++/ggml
inference path against the NeMo reference implementation.

**Anchor checkpoint:** `nvidia/parakeet-tdt_ctc-110m` — the **CTC head** of the
hybrid TDT/CTC model (selected via `m.change_decoding_strategy(decoder_type='ctc')`).
NeMo version 2.7.3. All comparisons are CPU, batch size 1, deterministic
(CTC greedy decode).

---

## Real-speech end-to-end check (Task 12)

The decisive proof: a real English utterance transcribed by the full C++
pipeline (mel → subsampling → FastConformer encoder → CTC head → CTC greedy
decode → BPE detokenize) compared word-for-word against NeMo's CTC-head
transcript of the same clip.

**Fixture:** `tests/fixtures/speech.wav` — LibriSpeech dev-clean sample
`2086-149220-0033`, 16 kHz mono, 16-bit PCM, ~7.4 s. Obtained from the public
NeMo tutorial mirror
`https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav`
(already 16 kHz mono; re-encoded to canonical 16-bit PCM via soundfile).

This clip is substantially harder than the synthetic-tone clip used for the
per-stage tensor-parity tests: it is ~3.7x longer and contains real speech, so
it exercises per-feature normalization and valid-length masking at a different
sequence length `T`, and a non-trivial greedy collapse over a real token stream
— exactly the length-dependent code paths a single fixed-length tone clip could
not have validated.

| Source | Transcript |
| --- | --- |
| **NeMo CTC reference** | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` |
| **C++ `parakeet-cli transcribe`** | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` |

**Result:** byte-for-byte identical. **WER = 0.0** (0 edits over 23 reference
words).

Reproduce:

```bash
# NeMo CTC reference
.venv/bin/python -c "from nemo.collections.asr.models import ASRModel; \
m=ASRModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m'); \
m.change_decoding_strategy(decoder_type='ctc'); \
print(m.transcribe(['tests/fixtures/speech.wav'])[0].text)"

# C++ pipeline
./build/examples/cli/parakeet-cli transcribe --model /tmp/pk110m.gguf \
    --input tests/fixtures/speech.wav
```

A committed regression test asserts this match deterministically:
`tests/test_transcribe_speech.cpp` (ctest `test_transcribe_speech`, label
`model`). It runs `pk::transcribe` on the committed `speech.wav` and asserts the
output equals the stored NeMo CTC reference string.

---

## Per-stage tensor parity (synthetic-tone clip)

Each stage diffs its C++ output against the matching tensor dumped from NeMo by
`scripts/gen_nemo_baseline.py` (stored in `baseline.gguf`). Numbers below are
the measured `max|diff|` from the ctest suite on `tests/fixtures/clip.wav`.

| Stage | ctest | max\|diff\| | Notes |
| --- | --- | --- | --- |
| Log-mel front end | `test_mel` | **4.13e-05** | `mel` `[80, T]` |
| dw_striding subsampling (÷8) | `test_subsampling` | **6.35e-03** | `subsampling_out`; largest-magnitude stage |
| Relative-position MHA (layer 0) | `test_relpos_attention` | **1.71e-03** | `l0_attn_out` |
| Conformer conv submodule | `test_conformer` | **1.58e-03** | `l0_conv_out` |
| Conformer layer 0 (full) | `test_conformer` | **5.34e-05** | `enc_layer_0` |
| Relative positional encoding | `test_encoder` | **1.55e-06** | `pos_emb` `[2T'-1, d_model]` |
| Encoder (full, last layer) | `test_encoder` | **2.40e-05** | `encoder_out` `[d_model, T']` |
| CTC head (log-softmax logits) | `test_ctc` | **1.72e-05** | `ctc_logits` `[T', vocab+1]` |

Notes:
- Diffs are absolute. The subsampling stage carries the largest absolute diff
  (6.3e-3) because its outputs have large magnitudes (O(1e3)); the relative
  error there is ~1e-6. Downstream stages (encoder, CTC) re-normalize and the
  absolute diff collapses back to O(1e-5).
- The CTC `exp`-row-sums equal 1.0 (the baseline stores post-`log_softmax`
  logits, and the C++ head matches that orientation).
- All stage diffs are well within the plan's tolerances and the greedy argmax /
  collapse is unaffected, which is why the end-to-end transcript is exact.

## Phase 2 — transducer core parity (max abs diff vs NeMo)

| Component | max\|diff\| | tolerance |
| --- | --- | --- |
| prediction net (embedding + 1-layer LSTM) | 1.5e-06 | 2e-3 |
| joint network (enc/pred proj → ReLU → linear, raw logits) | 1.1e-05 | 5e-3 |
| composed pred→joint (integration) | 1.3e-05 | 5e-3 |

- The joint emits **raw** logits `[T, U, 1030]` = 1024 vocab + 1 blank + 5 TDT
  durations `[0,1,2,3,4]`; the token/duration split + separate log_softmax is a
  Phase 3 (greedy) concern.
- The prediction net prepends a literal zero SOS step (`add_sos`), so 4 input
  ids → 5 hidden states. PyTorch LSTM gate order `[i,f,g,o]`, both biases summed.

## Test suite status

`ctest --test-dir build --output-on-failure` (with `PARAKEET_TEST_GGUF` and
`PARAKEET_TEST_BASELINE` exported): all 18 tests pass — the 15 Phase 0/1 tests
plus `test_prediction`, `test_joint`, `test_transducer_core`.
