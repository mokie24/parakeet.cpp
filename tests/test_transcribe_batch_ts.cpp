#include "model.hpp"
#include "audio_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
static bool words_close(const std::vector<pk::Word>& a, const std::vector<pk::Word>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].text != b[i].text) return false;
        if (std::fabs(a[i].start - b[i].start) > 1e-3f) return false;
        if (std::fabs(a[i].end   - b[i].end)   > 1e-3f) return false;
    }
    return true;
}
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    auto model = pk::Model::load(gguf);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }
    pk::Audio a;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", a) || a.samples.empty()) {
        std::fprintf(stderr, "wav load failed\n"); return 1;
    }
    std::vector<float> half(a.samples.begin(), a.samples.begin() + (a.samples.size()*3)/4);
    pk::Transcription r0 = model->transcribe_with_timestamps(a.samples, 16000);
    pk::Transcription r1 = model->transcribe_with_timestamps(half, 16000);
    auto batch = model->transcribe_pcm_batch_with_timestamps({a.samples, half}, 16000);
    if (batch.size() != 2) { std::fprintf(stderr, "size %zu\n", batch.size()); return 1; }
    bool ok = batch[0].text == r0.text && batch[1].text == r1.text
           && words_close(batch[0].words, r0.words) && words_close(batch[1].words, r1.words);
    std::fprintf(stderr, "item0 text %s words %zu/%zu; item1 text %s words %zu/%zu -> %s\n",
        (batch[0].text==r0.text?"OK":"DIFF"), batch[0].words.size(), r0.words.size(),
        (batch[1].text==r1.text?"OK":"DIFF"), batch[1].words.size(), r1.words.size(),
        ok?"OK":"FAIL");
    return ok ? 0 : 1;
}
