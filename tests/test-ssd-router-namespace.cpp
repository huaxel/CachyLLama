// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Unit test: update_args renders a preset-provided --cache-ssd verbatim.
//
// The router namespaces a router-level (shared) --cache-ssd per model at
// preset-merge time (server_models::load_models) so two models cannot
// overwrite each other's checkpoints through shared conversation/user
// directories. A cache-ssd set in the model's own preset is the
// operator's path and must pass through unchanged — appending the model
// name to it would silently migrate the on-disk layout and orphan every
// existing checkpoint under the configured path. update_args performs no
// namespacing of its own; this test guards that invariant.
#undef NDEBUG

#include "server-models.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static bool has_flag(const std::vector<std::string> & args, const std::string & flag) {
    for (const auto & arg : args) {
        if (arg == flag) return true;
    }
    return false;
}

static std::string find_arg_value(const std::vector<std::string> & args, const std::string & flag) {
    for (size_t i = 0; i + 1 < args.size(); i++) {
        if (args[i] == flag) return args[i + 1];
    }
    assert(!"flag not found in rendered args");
    return {};
}

int main() {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    // 1. preset-provided cache-ssd renders verbatim, no suffix appended
    {
        server_model_meta meta;
        meta.name = "Qwen3.6-35B-A3B-MTP";
        meta.port = 1234;
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/mnt/cache/qwen3_6-35b-a3b-mtp");
        meta.update_args(ctx, "/bin/llama-server");

        assert(has_flag(meta.args, "--cache-ssd"));
        assert(find_arg_value(meta.args, "--cache-ssd") == "/mnt/cache/qwen3_6-35b-a3b-mtp");
    }

    // 2. re-render (model reload) is stable: still verbatim
    {
        server_model_meta meta;
        meta.name = "m";
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/root");
        meta.update_args(ctx, "/bin/llama-server");
        meta.update_args(ctx, "/bin/llama-server");
        assert(find_arg_value(meta.args, "--cache-ssd") == "/root");
    }

    // 3. without --cache-ssd nothing is rendered
    {
        server_model_meta meta;
        meta.name = "m";
        meta.update_args(ctx, "/bin/llama-server");
        assert(!has_flag(meta.args, "--cache-ssd"));
    }

    printf("test-ssd-router-namespace: OK\n");
    return 0;
}
