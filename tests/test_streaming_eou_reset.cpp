#include "streaming.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

// MODEL: nvidia/parakeet_realtime_eou_120m-v1 (cache-aware streaming FastConformer)
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// Multi-utterance streaming decode parity / regression for issue #13
// ("streaming stops at first [EOU]"). The realtime EOU model emits <EOU>/<EOB> at
// the end of each utterance; NeMo's reference streaming driver
// (examples/voice_agent/.../nemo/streaming_asr.py NemoStreamingASRService ->
// reset_state on <EOU>/<EOB>) resets the decoder so the NEXT utterance decodes
// from a fresh state. pk::StreamingSession does the same (decoder reset on EOU in
// feed_mel_chunk). Without that reset the prediction net stays conditioned on
// <EOU> and the joint scores blank forever, so the stream goes silent after the
// first utterance.
//
// The baseline (scripts/gen_stream_reset_baseline.py) builds a TWO-utterance clip
// (speech.wav + silence + speech.wav) so an <EOU> fires MID-STREAM, then runs
// NeMo's cache-aware streaming loop WITH reset-on-EOU and stores the full token
// sequence (reset_token_ids) plus the index of the first <EOU>. We drive
// pk::StreamingSession over the SAME chunk schedule as test_streaming_decode and
// assert our streamed token ids match NeMo's reset reference EXACTLY.
//
// Skips (77) unless PARAKEET_TEST_GGUF_EOU + PARAKEET_TEST_BASELINE_EOU_RESET set.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_EOU_RESET");
    if (!gguf || !base) {
        std::fprintf(stderr,
            "test_streaming_eou_reset: PARAKEET_TEST_GGUF_EOU and/or "
            "PARAKEET_TEST_BASELINE_EOU_RESET not set; skip (streaming EOU model is "
            "a large download, not in CI)\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "[eou_reset] load failed %s\n", gguf);
        return 1;
    }
    if (!ml.config().streaming.present) {
        std::fprintf(stderr, "[eou_reset] model has no streaming config\n");
        return 1;
    }

    // NeMo reset-on-EOU reference token ids + the index of the first <EOU>.
    std::vector<int32_t> ref_ids;
    if (!pktest::load_baseline_i32(base, "reset_token_ids", ref_ids)) {
        std::fprintf(stderr, "[eou_reset] reset_token_ids not found in %s\n", base);
        return 1;
    }
    const uint32_t ref_eou_count = pktest::pktest_read_u32(base, "reset.eou_count");
    const uint32_t first_eou_idx = pktest::pktest_read_u32(base, "reset.first_eou_index");

    // mel [n_mels, T] row-major (feat-major inner=T) from the baseline.
    std::vector<float> mel;
    std::vector<int64_t> mshape;
    if (!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if (mshape.size() != 2) {
        std::fprintf(stderr, "[eou_reset] mel rank=%zu\n", mshape.size());
        return 1;
    }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    pk::StreamingSession sess(ml);
    const int chunk0     = sess.chunk_size_first();
    const int chunk_main = sess.chunk_size();
    const int pre_cache  = sess.pre_encode_cache_size();

    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    // Same chunk schedule as test_streaming_decode / run_stream_over_pcm.
    std::vector<int32_t> stream_ids;
    int n_chunks = 0, buffer_idx = 0;
    bool first = true;
    int eou_event_chunks = 0;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;
        int chunk_hi = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        std::vector<int32_t> emitted = sess.feed_mel_chunk(win, win_frames, is_last);
        stream_ids.insert(stream_ids.end(), emitted.begin(), emitted.end());
        if (sess.last_chunk_had_eou()) ++eou_event_chunks;
        ++n_chunks;
        buffer_idx += shift;
        first = false;
    }

    // Locate <EOU>/<EOB> ids from the tokenizer pieces (don't hardcode 1024/1025).
    int eou_id = -1, eob_id = -1;
    {
        const auto& pieces = ml.config().tokenizer_pieces;
        for (int i = 0; i < (int)pieces.size(); ++i) {
            if (pieces[i] == "<EOU>") eou_id = i;
            else if (pieces[i] == "<EOB>") eob_id = i;
        }
    }
    auto is_special = [&](int32_t t) { return t == eou_id || t == eob_id; };
    auto strip_specials = [&](const std::vector<int32_t>& v) {
        std::vector<int32_t> out;
        for (int32_t t : v) if (!is_special(t)) out.push_back(t);
        return out;
    };

    std::fprintf(stderr,
        "[eou_reset] n_chunks=%d emitted %zu tokens, eou-firing chunks=%d "
        "(ref tokens=%zu eou_count=%u first_eou_index=%u eou_id=%d)\n",
        n_chunks, stream_ids.size(), eou_event_chunks, ref_ids.size(),
        ref_eou_count, first_eou_idx, eou_id);

    // Premise guard: the reference MUST contain a MID-STREAM EOU with tokens after
    // it, otherwise this clip does not exercise the regression at all.
    if (ref_eou_count < 1 || first_eou_idx + 1 >= ref_ids.size()) {
        std::fprintf(stderr,
            "[eou_reset] FAIL: reference has no mid-stream EOU followed by more "
            "tokens (eou_count=%u first_eou_index=%u of %zu) — the baseline does "
            "not exercise continue-after-EOU; regenerate it.\n",
            ref_eou_count, first_eou_idx, ref_ids.size());
        return 1;
    }

    // The session's accumulated record must equal the per-chunk concat.
    if (stream_ids != sess.tokens()) {
        std::fprintf(stderr,
            "[eou_reset] INTERNAL: per-chunk concat (%zu) != session.tokens() (%zu)\n",
            stream_ids.size(), sess.tokens().size());
        return 1;
    }

    // The core issue-#13 guarantee: non-EOU TRANSCRIPT tokens are emitted AFTER
    // the first mid-stream <EOU>. With the old (no-reset) decoder this set is empty
    // (the stream goes silent after the first utterance); with the reset it holds
    // the whole second utterance.
    bool any_after_eou = false;
    for (size_t i = first_eou_idx + 1; i < stream_ids.size(); ++i)
        if (!is_special(stream_ids[i])) { any_after_eou = true; break; }
    if (!any_after_eou) {
        std::fprintf(stderr,
            "[eou_reset] FAIL (issue #13 REGRESSION): no transcript tokens emitted "
            "after the first <EOU> — the decoder was not reset on end-of-utterance, "
            "so the stream went silent after the first utterance.\n");
        return 1;
    }

    // Transcript parity: the NON-special token sequence (what the user sees) must
    // match NeMo's reset-on-EOU streaming decode EXACTLY. The <EOU>/<EOB> SPECIALS
    // are compared separately below — the only legitimate difference there is a
    // single trailing end-of-clip <EOU> on the final utterance, the documented
    // streaming-tail artifact (see test_streaming_decode): NeMo's offline-clip
    // driver resets the encoder cache on EOU and its degraded final-tail frame
    // drops that trailing <EOU>, whereas our decoder-only reset keeps the encoder
    // cache and may still emit it. That trailing special never changes the
    // transcript, and the real-time NeMo service would emit it once more audio
    // arrives, so it is explicitly out of scope here.
    std::vector<int32_t> got_ns = strip_specials(stream_ids);
    std::vector<int32_t> ref_ns = strip_specials(ref_ids);
    if (got_ns != ref_ns) {
        std::fprintf(stderr,
            "[eou_reset] TRANSCRIPT MISMATCH vs NeMo reset reference\n"
            "  got non-special=%zu expect non-special=%zu\n",
            got_ns.size(), ref_ns.size());
        size_t minlen = std::min(got_ns.size(), ref_ns.size());
        for (size_t i = 0; i < minlen; ++i) {
            if (got_ns[i] != ref_ns[i]) {
                std::fprintf(stderr, "  first diff at index %zu: got=%d expect=%d\n",
                             i, got_ns[i], ref_ns[i]);
                break;
            }
        }
        return 1;
    }

    // Full-stream check: identical to the reference up to an OPTIONAL trailing run
    // of specials (the end-of-clip <EOU> discussed above). Anything else — a
    // missing/duplicated mid-stream token, or a special appearing mid-transcript —
    // is a real decode bug and fails.
    bool tail_only_specials = stream_ids.size() >= ref_ids.size() &&
        std::equal(ref_ids.begin(), ref_ids.end(), stream_ids.begin());
    if (tail_only_specials)
        for (size_t i = ref_ids.size(); i < stream_ids.size(); ++i)
            if (!is_special(stream_ids[i])) { tail_only_specials = false; break; }

    std::fprintf(stderr,
        "[eou_reset] PASS — transcript (%zu non-special tokens) == NeMo cache-aware "
        "streaming decode WITH reset-on-EOU, EXACT, across %d chunks; the second "
        "utterance is fully recovered after the mid-stream <EOU> (issue #13).%s\n",
        ref_ns.size(), n_chunks,
        stream_ids.size() == ref_ids.size()
            ? ""
            : (tail_only_specials
                   ? " (+1 trailing end-of-clip <EOU>, streaming-tail artifact)"
                   : " (WARN: unexpected extra tokens)"));
    return tail_only_specials || stream_ids == ref_ids ? 0 : 1;
}
