#include "encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

// T5.2: end-to-end equivalence + padding invariance for the fused batched
// encoder (forward_batch).
//
// Build a ragged B=2 batch where item0 is the full baseline mel (length T0)
// and item1 is the first T1=(3*T0)/4 frames, zero-padded up to T0. Run both
// clips standalone via Encoder::forward, run the batch via forward_batch, and
// assert each item's VALID output region matches its standalone result. A
// divergence on item1 (the shorter, zero-padded clip) or any perturbation of
// item0 by the shorter neighbor signals pad leakage in the batched encoder,
// NOT a test bug. Tolerance 5e-2/5e-2 mirrors test_encoder.cpp (error
// accumulates over the 17 conformer layers).
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "model load failed\n"); return 1; }

    // Baseline "mel" is [n_mels, T0] row-major (feat-major inner = T0).
    std::vector<float> mel; std::vector<int64_t> ms;
    if (!pktest::load_baseline(base, "mel", mel, ms)) return 1;
    if (ms.size() != 2) { std::fprintf(stderr, "mel rank=%zu\n", ms.size()); return 1; }
    const int n_mels = (int)ms[0];
    const int T0     = (int)ms[1];
    const int T1     = (T0 * 3) / 4;

    pk::Encoder enc(ml);

    // --- Standalone references ---------------------------------------------
    std::vector<float> e0, e1; int dm = 0, to0 = 0, to1 = 0;
    enc.forward(mel, n_mels, T0, e0, dm, to0);

    std::vector<float> mel1((size_t)n_mels * T1);
    for (int m = 0; m < n_mels; ++m)
        for (int t = 0; t < T1; ++t)
            mel1[(size_t)m * T1 + t] = mel[(size_t)m * T0 + t];
    int dm1 = 0;
    enc.forward(mel1, n_mels, T1, e1, dm1, to1);

    // --- Batched (T_max=T0; item1 zero-padded) -----------------------------
    pk::MelBatch mb;
    mb.B = 2; mb.n_mels = n_mels; mb.T_max = T0; mb.valid_T = { T0, T1 };
    mb.data.assign((size_t)2 * n_mels * T0, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < T0; ++t) mb.data[((size_t)0 * n_mels + m) * T0 + t] = mel[(size_t)m * T0 + t];
        for (int t = 0; t < T1; ++t) mb.data[((size_t)1 * n_mels + m) * T0 + t] = mel1[(size_t)m * T1 + t];
    }
    std::vector<std::vector<float>> eo; int dmb = 0, tob = 0; std::vector<int> vt;
    enc.forward_batch(mb, eo, dmb, tob, vt);

    if (dmb <= 0 || tob <= 0 || eo.size() != 2 || vt.size() != 2) {
        std::fprintf(stderr, "bad batched output dmb=%d tob=%d |eo|=%zu |vt|=%zu\n",
                     dmb, tob, eo.size(), vt.size());
        return 1;
    }

    // Valid-length plumbing sanity. forward() reports Tout = each clip's OWN Tp,
    // so to0 (full clip, T0) and to1 (shorter clip, T1) generally differ. The
    // batch is padded to T_max=T0, so the batched tob == to0 (item0's Tp).
    // vt[b] is the per-item NON-PAD valid frame count (= subsampling
    // valid_out_len). For a clip whose length is not a clean power-of-two
    // multiple the offline "T-1" convention can leave one trailing pad output
    // frame, so vt[b] <= tob. The comparison below slices to vt[b] columns so
    // only genuine (non-pad) frames are checked.
    std::fprintf(stderr,
        "[encbatch] dm=%d to0=%d to1=%d | dmb=%d tob=%d vt={%d,%d}\n",
        dm, to0, to1, dmb, tob, vt[0], vt[1]);
    if (dmb != dm || tob != to0)
        std::fprintf(stderr, "[encbatch] WARN shape mismatch: dmb=%d(dm=%d) tob=%d(to0=%d)\n",
                     dmb, dm, tob, to0);
    if (vt[0] < 1 || vt[0] > tob || vt[1] < 1 || vt[1] > to1)
        std::fprintf(stderr, "[encbatch] WARN vt out of range: vt={%d,%d} tob=%d to1=%d\n",
                     vt[0], vt[1], tob, to1);

    // enc_out is channels-first [d_model, Tout] (full[c*Tout + t]); slice the
    // first Tvalid columns from each row.
    auto slice_cols = [&](const std::vector<float>& full, int Tfull, int Tvalid) {
        std::vector<float> s((size_t)dmb * Tvalid);
        for (int c = 0; c < dmb; ++c)
            for (int t = 0; t < Tvalid; ++t)
                s[(size_t)c * Tvalid + t] = full[(size_t)c * Tfull + t];
        return s;
    };
    // forward_batch compacts each eo[b] to its own valid_Tout[b] columns, so its
    // full row width is vt[b] (not the padded tob). The standalone references
    // keep their own full Tout (to0 / to1).
    std::vector<float> b0   = slice_cols(eo[0], vt[0], vt[0]);
    std::vector<float> b1   = slice_cols(eo[1], vt[1], vt[1]);
    std::vector<float> ref0 = slice_cols(e0, to0, vt[0]);
    std::vector<float> ref1 = slice_cols(e1, to1, vt[1]);

    bool a = pktest::compare(b0, ref0, "encbatch.item0", 5e-2f, 5e-2f);
    bool b = pktest::compare(b1, ref1, "encbatch.item1", 5e-2f, 5e-2f);
    return (a && b) ? 0 : 1;
}
