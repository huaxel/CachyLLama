// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Unit test for router-mode SSD cache root namespacing.
//
// Router mode propagates the router's --cache-ssd PATH to every model worker,
// but the cache directories under the root are keyed only by conversation/
// user hash, never by model, so two workers sharing one root overwrite each
// other's checkpoints. update_args must therefore render each worker's own
// root as <path>/<sanitized model name>. The append must be idempotent
// because update_args re-renders on model reload.
#undef NDEBUG

#include "server-models.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static bool has_flag(const std::vector<std::string> & args, const std::string & flag) {
    for (const auto & arg : args) {
        if (arg == flag) {
            return true;
        }
    }
    return false;
}

static std::string find_arg_value(const std::vector<std::string> & args, const std::string & flag) {
    for (size_t i = 0; i + 1 < args.size(); i++) {
        if (args[i] == flag) {
            return args[i + 1];
        }
    }
    assert(!"flag not found in rendered args");
    return {};
}

int main() {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    // 1. shared router root is namespaced with the sanitized model name
    {
        server_model_meta meta;
        meta.name = "model one";
        meta.port = 1234;
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/tmp/cache-root");
        meta.update_args(ctx, "/bin/llama-server");

        assert(has_flag(meta.args, "--cache-ssd"));
        assert(find_arg_value(meta.args, "--cache-ssd") == "/tmp/cache-root/model_one");
    }

    // 2. re-render (model reload) must not append the name twice
    {
        server_model_meta meta;
        meta.name = "qwen";
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/root");
        meta.update_args(ctx, "/bin/llama-server");
        meta.update_args(ctx, "/bin/llama-server");
        assert(find_arg_value(meta.args, "--cache-ssd") == "/root/qwen");
    }

    // 3. trailing slash in the operator path must not double up
    {
        server_model_meta meta;
        meta.name = "m";
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/root/");
        meta.update_args(ctx, "/bin/llama-server");
        assert(find_arg_value(meta.args, "--cache-ssd") == "/root/m");
    }

    // 4. path-unsafe characters in the model name are mapped to '_'
    {
        server_model_meta meta;
        meta.name = "a/b c:d";
        meta.preset.set_option(ctx, "LLAMA_ARG_CACHE_SSD", "/root");
        meta.update_args(ctx, "/bin/llama-server");
        assert(find_arg_value(meta.args, "--cache-ssd") == "/root/a_b_c_d");
    }

    // 5. without --cache-ssd nothing is rendered or namespaced
    {
        server_model_meta meta;
        meta.name = "m";
        meta.update_args(ctx, "/bin/llama-server");
        assert(!has_flag(meta.args, "--cache-ssd"));
    }

    printf("test-ssd-router-namespace: OK\n");
    return 0;
}
