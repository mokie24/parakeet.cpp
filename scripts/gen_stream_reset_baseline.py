#!/usr/bin/env python3
"""Dump NeMo cache-aware streaming decode WITH end-of-utterance reset to a gguf.

Reference for the multi-utterance streaming regression (issue #13: "streaming
stops at first [EOU]"). The realtime EOU model emits <EOU>/<EOB> to mark the end
of an utterance; NeMo's reference streaming driver
(examples/voice_agent/.../nemo/streaming_asr.py NemoStreamingASRService.transcribe)
calls reset_state() whenever <EOU>/<EOB> appears in a chunk, so the NEXT utterance
decodes from a fresh decoder state. Without that reset the decoder stays
conditioned on <EOU> and goes silent after the first utterance.

To exercise a MID-STREAM EOU (the existing speech.wav fires <EOU> only on the
final streaming tail, which is dropped) this script builds a two-utterance clip
from the supplied --audio by concatenating it with a short silence gap and a
second copy, then runs NeMo's canonical cache-aware streaming loop (identical to
gen_stream_baseline.py — the schedule pk::run_stream_over_pcm mirrors) WITH the
reset-on-EOU behavior, accumulating the full token sequence across resets.

Stored:
* ``mel``               ``[n_mels, T]`` f32   the two-utterance clip mel
                                              (feat-major inner=T)
* ``reset_token_ids``   ``[L]`` int32         full streaming token ids across the
                                              whole clip WITH reset-on-EOU (incl
                                              the <EOU>/<EOB> specials)
* ``reset.eou_count``   uint32                number of <EOU>/<EOB> events
* ``reset.eou_id`` / ``reset.eob_id`` int32   the special token ids
* ``reset.first_eou_index`` uint32            index in reset_token_ids of the
                                              FIRST <EOU>/<EOB> (so the C++ test
                                              can assert tokens exist AFTER it)

Exit codes (ctest convention): 0 ok, 2 deps/model unavailable, 1 fail.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover
    print(f"stream-reset-baseline: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import torch
    import soundfile as sf
    from omegaconf import open_dict
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )
except ImportError as e:  # pragma: no cover
    print(f"stream-reset-baseline: missing dependency: {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet_realtime_eou_120m-v1",
                    help="HF id or local .nemo")
    ap.add_argument("--audio", required=True, help="16k mono wav clip (one utterance)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gap-secs", type=float, default=0.6,
                    help="silence between the two copies")
    args = ap.parse_args()

    try:
        if pathlib.Path(args.model).exists():
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    enc = m.encoder
    if not hasattr(enc, "cache_aware_stream_step"):
        print("PARAKEET_MODEL_UNAVAILABLE: not a streaming model", file=sys.stderr)
        sys.exit(2)
    enc.setup_streaming_params()
    sc = enc.streaming_cfg

    # Non-batched greedy with carried partial_hypotheses (same as gen_stream_baseline).
    import copy as _copy
    _dcfg = _copy.deepcopy(m.cfg.decoding)
    with open_dict(_dcfg):
        _dcfg.strategy = "greedy"
        _dcfg.compute_timestamps = False
        _dcfg.preserve_alignments = False
        if "greedy" in _dcfg:
            _dcfg.greedy.preserve_frame_confidence = False
    m.change_decoding_strategy(_dcfg)

    # Resolve <EOU>/<EOB> ids.
    def tid(tok):
        try:
            r = m.tokenizer.tokens_to_ids([tok])
            return int(r[0]) if r else -1
        except Exception:
            return -1
    eou_id, eob_id = tid("<EOU>"), tid("<EOB>")
    specials = {x for x in (eou_id, eob_id) if x >= 0}

    # Build the two-utterance clip: [clip, gap silence, clip].
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        print(f"PARAKEET_BASELINE_BAD_AUDIO: expected 16k mono, got sr={sr}",
              file=sys.stderr)
        sys.exit(1)
    gap = np.zeros(int(round(args.gap_secs * 16000)), dtype=np.float32)
    clip = np.concatenate([wav, gap, wav]).astype(np.float32)
    wav_t = torch.from_numpy(np.ascontiguousarray(clip)).float().unsqueeze(0)
    len_t = torch.tensor([wav_t.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        feats, feat_len = m.preprocessor(input_signal=wav_t, length=len_t)
    n_mels = int(feats.shape[1])
    mel_np = feats.detach().cpu().float().numpy()[0]  # [n_mels, T]

    # Canonical cache-aware streaming loop (== gen_stream_baseline.py) WITH
    # reset-on-EOU (NeMo reset_state: blank hypothesis + fresh encoder cache).
    sb = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False,
                                        pad_and_drop_preencoded=False)
    sb.append_processed_signal(feats, stream_id=-1)
    clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)
    previous_hypotheses = None
    collected = []
    seg_prev_len = 0
    n_resets = 0
    for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(sb)):
        drop = sc.drop_extra_pre_encoded if step_num != 0 else 0
        keep_all = sb.is_buffer_empty()
        with torch.no_grad():
            e, el, clc, clt, clcl = enc.cache_aware_stream_step(
                processed_signal=chunk_audio, processed_signal_length=chunk_lengths,
                cache_last_channel=clc, cache_last_time=clt,
                cache_last_channel_len=clcl,
                keep_all_outputs=keep_all, drop_extra_pre_encoded=drop)
        with torch.inference_mode():
            dec = m.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=e, encoded_lengths=el, return_hypotheses=True,
                partial_hypotheses=previous_hypotheses)
        ys = dec[0].y_sequence
        ys = ys.tolist() if isinstance(ys, torch.Tensor) else list(ys)
        new = [int(t) for t in ys[seg_prev_len:]]
        collected.extend(new)
        if any(t in specials for t in new):
            n_resets += 1
            previous_hypotheses = None            # blank hypothesis (SOS)
            seg_prev_len = 0
            clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)  # fresh cache
        else:
            previous_hypotheses = dec
            seg_prev_len = len(ys)

    reset_ids = np.array(collected, dtype=np.int32)
    eou_count = int(sum(1 for t in collected if t in specials))
    first_eou_index = next((i for i, t in enumerate(collected) if t in specials), -1)

    non_special = [t for t in collected if t not in specials]
    text = m.tokenizer.ids_to_text(non_special)
    print(f"reset-baseline: T={mel_np.shape[1]} tokens={len(collected)} "
          f"eou/eob={eou_count} resets={n_resets} first_eou_index={first_eou_index}")
    print(f"  reset_token_ids: {collected}")
    print(f"  text (specials stripped): {text!r}")
    if eou_count < 1:
        print("PARAKEET_RESET_BASELINE_WARN: no <EOU>/<EOB> fired; the clip does "
              "not exercise the reset path.", file=sys.stderr)
    if first_eou_index < 0 or first_eou_index >= len(collected) - 1:
        print("PARAKEET_RESET_BASELINE_WARN: no tokens AFTER the first EOU; the "
              "regression (continue-after-EOU) is not exercised.", file=sys.stderr)

    w = gguf.GGUFWriter(args.output, "parakeet-stream-reset-baseline")
    w.add_tensor("mel", np.ascontiguousarray(mel_np, dtype=np.float32))
    w.add_tensor("reset_token_ids", np.ascontiguousarray(reset_ids))
    w.add_uint32("reset.eou_count", eou_count)
    w.add_int32("reset.eou_id", int(eou_id))
    w.add_int32("reset.eob_id", int(eob_id))
    w.add_uint32("reset.first_eou_index", int(first_eou_index))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: mel={mel_np.shape} reset_tokens={len(collected)}")


if __name__ == "__main__":
    main()
