// Device selection via PARAKEET_DEVICE (issue #17).
//
// Runs on any build, including the CPU-only one used in CI: it exercises the
// env-var parsing and fallback paths that don't need a GPU to be present.
//   - PARAKEET_DEVICE=cpu        -> CPU backend.
//   - PARAKEET_DEVICE=<unknown>  -> no such device, falls back to CPU.
//   - unset                      -> CPU on a CPU-only build (auto-pick).
// On a GPU build the unset/auto case may select a GPU/integrated-GPU device, so
// we don't assert a specific name there.
#include "backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int failures = 0;

static void expect_cpu(const char* label, const std::string& name) {
    if (name != "cpu") {
        std::fprintf(stderr, "FAIL [%s]: expected device 'cpu', got '%s'\n",
                     label, name.c_str());
        ++failures;
    } else {
        std::printf("ok [%s]: device = %s\n", label, name.c_str());
    }
}

int main() {
    // Forcing CPU must always yield the CPU backend.
    setenv("PARAKEET_DEVICE", "cpu", 1);
    {
        pk::Backend b(1);
        expect_cpu("PARAKEET_DEVICE=cpu", b.device_name());
    }

    // An unknown device name must not crash; it falls back to CPU.
    setenv("PARAKEET_DEVICE", "definitely-not-a-real-device-9000", 1);
    {
        pk::Backend b(1);
        expect_cpu("PARAKEET_DEVICE=<unknown>", b.device_name());
    }

    // Case-insensitive "CPU" is also honored as a CPU force.
    setenv("PARAKEET_DEVICE", "CPU", 1);
    {
        pk::Backend b(1);
        expect_cpu("PARAKEET_DEVICE=CPU", b.device_name());
    }

    unsetenv("PARAKEET_DEVICE");
    return failures == 0 ? 0 : 1;
}
