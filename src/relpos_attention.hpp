#pragma once
#include "model_loader.hpp"
#include "graph_builder.hpp"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// Relative-position multi-head self-attention (Transformer-XL style) as used by
// the FastConformer encoder — NeMo RelPositionMultiHeadAttention.
//
// IMPORTANT (module boundary): this mirrors NeMo's `self_attn(...)` exactly. The
// input `x` is the ALREADY-normalized attention input, i.e. NeMo's
// `norm_self_att(residual)` (note: `residual` already includes FFN1, so it is
// NOT `norm_self_att(enc_pre_layers)`). The norm itself lives in the conformer
// layer (next task), NOT here. q, k and v are all projected from this same `x`.
//
// Layout convention (matches the rest of the port and the baseline GGUF):
//   x       row-major [T, d_model]      (d_model fastest)
//   pos_emb row-major [2T-1, d_model]   (d_model fastest)
//   out     row-major [T, d_model]      (d_model fastest)
//
// `valid_len` is the number of non-padding frames. Frames >= valid_len are
// padding: their key columns get -inf in the scores (zero attention weight) and
// their query rows are fully masked, so each padded output row reduces to
// linear_out.bias — exactly matching NeMo's att_mask handling. Pass valid_len ==
// T to disable masking.
class RelPosAttention {
public:
    RelPosAttention(const ModelLoader& ml, int layer_idx);

    // GRAPH-BUILDER: append MHSA ops to a SHARED graph `ctx`. `xt` is the
    // normalized attention input tensor [D, T] (ggml ne0=D fastest) and `pe` is
    // the positional-encoding tensor [D, pos_len], both ALREADY in the graph.
    // Returns the attention output [D, T]. Host-built additive masks are fed via
    // pk::graph_input_tensor and registered into `pool` (must outlive compute).
    // Reused by the fused conformer layer and the unit test.
    //
    // When att_left/att_right >= 0, an additional SYMMETRIC sliding-window mask
    // is applied: query qi may attend to key kj only if -att_left <= qi-kj <=
    // att_right (NeMo rel_pos_local_attn). Defaults (-1, -1) = full context.
    ggml_tensor* build_graph(ggml_context* ctx, ggml_tensor* xt, int T,
                             ggml_tensor* pe, int pos_len, int valid_len,
                             GraphInputPool& pool,
                             int att_left = -1, int att_right = -1) const;

    // LOCAL (banded / Longformer) GRAPH-BUILDER. Appends NeMo rel_pos_local_attn
    // ops to a SHARED graph. `xt` is [D, T] and `pe` is the LOCAL positional
    // encoding [D, att_left+att_right+1]. Each query attends only to keys within
    // [t-att_left, t+att_right] via pad-and-shift, so peak memory is
    // O(T * window) not O(T^2). Returns the attention output [D, T].
    ggml_tensor* build_graph_local(ggml_context* ctx, ggml_tensor* xt, int T,
                                   ggml_tensor* pe, int pos_len, int valid_len,
                                   int att_left, int att_right,
                                   GraphInputPool& pool) const;

    // Batched GRAPH-BUILDER. `xt` is [D, T, B]; `pe` is [D, pos_len] (shared
    // across the batch). `valid_len` is per item (size B). Returns [D, T, B].
    ggml_tensor* build_graph_batched(ggml_context* ctx, ggml_tensor* xt, int T,
                                     int B, ggml_tensor* pe, int pos_len,
                                     const std::vector<int>& valid_len,
                                     GraphInputPool& pool) const;

    // x: [T, d_model]; pos_emb: [2T-1, d_model]; out: [T, d_model].
    void forward(const std::vector<float>& x, int T,
                 const std::vector<float>& pos_emb, int pos_len,
                 int valid_len,
                 std::vector<float>& out) const;

    // Local (banded / Longformer) attention — NeMo rel_pos_local_attn. Each
    // query qi attends only to keys in [qi-att_left, qi+att_right]; pos_emb is
    // the LOCAL positional encoding [att_left+att_right+1, d_model] (i.e. 2W+1
    // for a symmetric [W,W] window), NOT the full 2T-1. Output matches NeMo's
    // Longformer self_attn while bounding memory to O(T * window) instead of
    // O(T^2). x: [T, d_model]; out: [T, d_model].
    void forward_local(const std::vector<float>& x, int T,
                       const std::vector<float>& pos_emb, int pos_len,
                       int valid_len, int att_left, int att_right,
                       std::vector<float>& out) const;
private:
    const ModelLoader& ml_;
    int layer_idx_;
    int d_model_;
    int n_heads_;
    int d_head_;
    // Chunked-limited attention (att_context_style=="chunked_limited"). When set,
    // an extra additive -inf window mask is applied to the scores (see forward()).
    bool chunked_limited_ = false;
    int att_left_  = -1;   // att_context_size[0] (left limit in frames)
    int att_right_ = -1;   // att_context_size[1] (right/lookahead in frames)
};

} // namespace pk
