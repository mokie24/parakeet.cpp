#include "model_fetch.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace pkserver {

const char* kCollectionRepo = "mudler/parakeet-cpp-gguf";

const std::vector<std::pair<std::string, std::string>>& model_aliases() {
    static const std::vector<std::pair<std::string, std::string>> kAliases = {
        {"tdt_ctc-110m",      "tdt_ctc-110m-f16.gguf"},
        {"tdt_ctc-110m-q4_k", "tdt_ctc-110m-q4_k.gguf"},
        {"tdt_ctc-1.1b",      "tdt_ctc-1.1b-f16.gguf"},
        {"tdt-0.6b-v2",       "tdt-0.6b-v2-f16.gguf"},
        {"tdt-0.6b-v3",       "tdt-0.6b-v3-f16.gguf"},
        {"tdt-1.1b",          "tdt-1.1b-f16.gguf"},
        {"ctc-0.6b",          "ctc-0.6b-f16.gguf"},
        {"ctc-1.1b",          "ctc-1.1b-f16.gguf"},
        {"rnnt-0.6b",         "rnnt-0.6b-f16.gguf"},
        {"rnnt-1.1b",         "rnnt-1.1b-f16.gguf"},
        {"eou-120m",          "realtime_eou_120m-v1-f16.gguf"},
    };
    return kAliases;
}

static std::string collective_url(const std::string& filename) {
    return std::string("https://huggingface.co/") + kCollectionRepo +
           "/resolve/main/" + filename;
}

static std::string basename_of(const std::string& url) {
    auto pos = url.find_last_of('/');
    return pos == std::string::npos ? url : url.substr(pos + 1);
}

// Allow only filesystem-safe cache names so we never write outside the cache or
// craft a surprising curl argument.
static bool safe_name(const std::string& n) {
    if (n.empty() || n == "." || n == "..") return false;
    for (char c : n) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
        if (!ok) return false;
    }
    return true;
}

bool resolve_model(const std::string& arg, ModelSource& out, std::string& err) {
    std::error_code ec;
    if (fs::exists(arg, ec) && fs::is_regular_file(arg, ec)) {
        out.kind = ModelSource::kLocalPath;
        out.value = arg;
        return true;
    }
    if (arg.rfind("http://", 0) == 0 || arg.rfind("https://", 0) == 0) {
        out.kind = ModelSource::kUrl;
        out.value = arg;
        out.cache_name = basename_of(arg);
        if (!safe_name(out.cache_name)) {
            err = "unsafe model filename in URL: " + out.cache_name;
            return false;
        }
        return true;
    }
    for (const auto& a : model_aliases()) {
        if (a.first == arg) {
            out.kind = ModelSource::kUrl;
            out.value = collective_url(a.second);
            out.cache_name = a.second;
            return true;
        }
    }
    if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".gguf" &&
        safe_name(arg)) {
        out.kind = ModelSource::kUrl;
        out.value = collective_url(arg);
        out.cache_name = arg;
        return true;
    }
    err = "unknown model '" + arg +
          "'. Pass a local .gguf path, an http(s):// URL, a <name>.gguf in " +
          kCollectionRepo + ", or one of the aliases:";
    for (const auto& a : model_aliases()) err += " " + a.first;
    return false;
}

std::string default_cache_dir() {
    if (const char* c = std::getenv("PARAKEET_CACHE_DIR"); c && *c) return c;
    std::string base;
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
        base = x;
    } else if (const char* h = std::getenv("HOME"); h && *h) {
        base = std::string(h) + "/.cache";
    } else {
        base = ".cache";
    }
    return base + "/parakeet.cpp/models";
}

static bool have_tool(const char* tool) {
    std::string cmd = std::string("command -v ") + tool + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

std::string fetch_model(const ModelSource& src, const std::string& cache_dir) {
    if (src.kind == ModelSource::kLocalPath) return src.value;

    fs::create_directories(cache_dir);
    fs::path final_path = fs::path(cache_dir) / src.cache_name;
    if (fs::exists(final_path)) return final_path.string();

    fs::path part = final_path;
    part += ".part";

    // src.value and src.cache_name are validated by resolve_model (https scheme,
    // safe filename). Quote them anyway.
    std::string cmd;
    if (have_tool("curl")) {
        cmd = "curl -fSL --retry 3 -o '" + part.string() + "' '" + src.value + "'";
    } else if (have_tool("wget")) {
        cmd = "wget -O '" + part.string() + "' '" + src.value + "'";
    } else {
        throw std::runtime_error(
            "no curl or wget on PATH to download " + src.value +
            "; download it manually and pass the local path to --model");
    }

    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::error_code ec;
        fs::remove(part, ec);
        throw std::runtime_error("download failed (exit " + std::to_string(rc) +
                                 ") for " + src.value);
    }
    std::error_code ec;
    fs::rename(part, final_path, ec);
    if (ec) {
        throw std::runtime_error("could not finalize download at " +
                                 final_path.string() + ": " + ec.message());
    }
    return final_path.string();
}

} // namespace pkserver
