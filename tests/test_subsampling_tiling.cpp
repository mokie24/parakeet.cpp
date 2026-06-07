#include "subsampling.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if(!gguf){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    pk::Subsampling sub(ml);
    // f(x)=(x-1)/2+1 applied 3x for non-causal; spot-check.
    struct { int T, Tp; } cases[] = {{100, 13}, {1000, 125}, {307848, 38481}};
    for (auto& c : cases) {
        int got = sub.subsample_len(c.T);
        if (got != c.Tp){ std::fprintf(stderr,"T=%d got=%d want=%d\n",c.T,got,c.Tp); return 1; }
    }
    std::printf("subsample_len ok\n");
    return 0;
}
