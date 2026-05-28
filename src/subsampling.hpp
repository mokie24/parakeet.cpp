#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {

// dw_striding (÷8) convolutional subsampling front end of the FastConformer
// encoder (NeMo ConvSubsampling, dw_striding branch). Builds one ggml graph.
class Subsampling {
public:
    explicit Subsampling(const ModelLoader& ml);
    // mel: row-major [n_mels, T] (feat-major inner = T) — i.e. mel[m*T + t].
    // out: row-major [Tout, d_model] (time-major) matching baseline subsampling_out.
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& out, int& Tout, int& d_model) const;
private:
    const ModelLoader& ml_;
    int conv_channels_;   // C
    int d_model_;         // out features
};

} // namespace pk
